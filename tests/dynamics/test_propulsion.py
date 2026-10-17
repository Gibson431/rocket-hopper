import numpy as np
import pytest

from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.propulsion import propulsion
from hopper.sim.integrator import integrate
from hopper.types import GIM, GIMR, N_STATE, PROP, THR, Command, make_state

G0 = 9.80665


def cmd(throttle=0.0, gimbal=(0.0, 0.0), roll=0.0):
    return Command(throttle=throttle, gimbal=np.array(gimbal), roll_torque=roll)


def out(cfg, x, c):
    return propulsion(x, c, cfg, mass_properties(x[PROP], cfg))


def state(prop=20.0, thr=0.0, gim=(0.0, 0.0), gimr=(0.0, 0.0)):
    return make_state([0, 0, 0], [0, 0, 0], None, [0, 0, 0], prop, thr, gim, gimr)


def run_actuators(cfg, x0, c, t1, dt=1e-3):
    def f(t, x):
        return out(cfg, x, c).xdot_act

    return integrate(f, x0, 0.0, t1, dt, renorm=None, record=True)


def test_steady_thrust_is_thrust_max_times_fraction(cfg):
    x = state(thr=0.7)
    o = out(cfg, x, cmd(0.7))
    assert o.force_B == pytest.approx([0, 0, cfg.engine.thrust_max * 0.7], abs=1e-9)
    assert o.xdot_act[THR] == pytest.approx(0.0, abs=1e-12)


def test_no_thrust_and_decaying_lag_state_when_out_of_propellant(cfg):
    x = state(prop=0.0, thr=1.0)
    o = out(cfg, x, cmd(1.0))
    assert np.all(o.force_B == 0.0)
    assert o.xdot_act[THR] == pytest.approx(-1.0 / cfg.engine.throttle_tau)
    assert o.xdot_act[PROP] == 0.0


def test_zero_command_means_engine_off(cfg):
    o = out(cfg, state(thr=0.5), cmd(0.0))
    assert o.xdot_act[THR] == pytest.approx(-0.5 / cfg.engine.throttle_tau)


def test_command_below_minimum_is_clamped_up_to_minimum(cfg):
    o = out(cfg, state(thr=0.0), cmd(0.1))
    assert o.xdot_act[THR] == pytest.approx(cfg.engine.thrust_min_frac / cfg.engine.throttle_tau)


def test_command_above_one_is_clamped_to_full(cfg):
    o = out(cfg, state(thr=0.0), cmd(3.0))
    assert o.xdot_act[THR] == pytest.approx(1.0 / cfg.engine.throttle_tau)


def test_throttle_step_reaches_63_percent_after_one_time_constant(cfg):
    _, traj = run_actuators(cfg, state(thr=0.0), cmd(1.0), cfg.engine.throttle_tau)
    assert traj[-1, THR] == pytest.approx(1 - np.exp(-1), rel=0.01)


def test_mass_flow_follows_isp(cfg):
    x = state(thr=1.0)
    o = out(cfg, x, cmd(1.0))
    assert o.xdot_act[PROP] == pytest.approx(-cfg.engine.thrust_max / (cfg.engine.isp * G0))


def test_small_gimbal_step_settles_to_command_within_limits(cfg):
    target = 0.02
    x, traj = run_actuators(cfg, state(thr=0.5), cmd(0.5, (target, -target)), 1.0)
    assert x[GIM] == pytest.approx([target, -target], abs=1e-4)
    assert x[GIMR] == pytest.approx([0.0, 0.0], abs=1e-3)


def test_large_gimbal_step_respects_angle_and_rate_limits(cfg):
    dt = 1e-3
    _, traj = run_actuators(cfg, state(thr=0.5), cmd(0.5, (1.0, -1.0)), 1.0, dt)
    gmax, rmax = cfg.engine.gimbal_max, cfg.engine.gimbal_rate_max
    ang = traj[:, GIM]
    # RK4 may overshoot the limit by at most one step at the rate limit.
    assert np.max(np.abs(ang)) <= gmax + rmax * dt
    assert np.max(np.abs(np.diff(ang, axis=0))) / dt <= rmax * (1 + 1e-9)
    # It does get there, and the commanded value was clamped (not the raw 1.0).
    assert np.abs(ang[-1]) == pytest.approx([gmax, gmax], abs=1e-4)


def test_gimbal_angle_rate_derivative_is_clamped_to_rate_limit(cfg):
    x = state(thr=0.5, gimr=(5.0, -5.0))
    o = out(cfg, x, cmd(0.5, (0.1, -0.1)))
    assert o.xdot_act[GIM] == pytest.approx(
        [cfg.engine.gimbal_rate_max, -cfg.engine.gimbal_rate_max]
    )


def test_no_outward_acceleration_when_at_rate_limit(cfg):
    rmax = cfg.engine.gimbal_rate_max
    x = state(thr=0.5, gim=(0.0, 0.0), gimr=(rmax, 0.0))
    o = out(cfg, x, cmd(0.5, (cfg.engine.gimbal_max, 0.0)))
    assert o.xdot_act[GIMR][0] == 0.0


def test_thrust_direction_uses_angles_clamped_to_limit(cfg):
    gmax = cfg.engine.gimbal_max
    o = out(cfg, state(thr=1.0, gim=(0.0, 3 * gmax)), cmd(1.0))
    assert o.force_B[0] / cfg.engine.thrust_max == pytest.approx(np.sin(gmax))


@pytest.mark.parametrize(
    "gx,gy", [(0.0, 0.0), (0.1, 0.0), (0.0, -0.1), (0.07, 0.12), (-0.13, 0.05)]
)
def test_thrust_vector_magnitude_independent_of_gimbal(cfg, gx, gy):
    o = out(cfg, state(thr=0.8, gim=(gx, gy)), cmd(0.8))
    assert np.linalg.norm(o.force_B) == pytest.approx(cfg.engine.thrust_max * 0.8, rel=1e-12)


def test_positive_gy_gives_negative_torque_about_y(cfg):
    o = out(cfg, state(thr=1.0, gim=(0.0, 0.1)), cmd(1.0))
    assert o.torque_B[1] < 0
    assert o.torque_B[0] == pytest.approx(0.0, abs=1e-9)
    com_z = mass_properties(20.0, cfg).com_z
    assert o.torque_B[1] == pytest.approx(-com_z * o.force_B[0])


def test_positive_gx_gives_negative_torque_about_x(cfg):
    o = out(cfg, state(thr=1.0, gim=(0.1, 0.0)), cmd(1.0))
    assert o.torque_B[0] < 0
    assert o.torque_B[1] == pytest.approx(0.0, abs=1e-9)


def test_zero_gimbal_gives_zero_torque(cfg):
    o = out(cfg, state(thr=1.0), cmd(1.0))
    assert np.all(o.torque_B == 0.0)


def test_roll_torque_is_pure_z_scaled_and_clamped(cfg):
    x = state(thr=0.0)
    full = out(cfg, x, cmd(0.0, roll=1.0)).torque_B
    assert full == pytest.approx([0, 0, cfg.rcs.max_roll_torque])
    half = out(cfg, x, cmd(0.0, roll=-0.5)).torque_B
    assert half == pytest.approx([0, 0, -0.5 * cfg.rcs.max_roll_torque])
    over = out(cfg, x, cmd(0.0, roll=7.0)).torque_B
    assert over == pytest.approx([0, 0, cfg.rcs.max_roll_torque])


def test_actuator_derivative_is_zero_outside_actuator_slots(cfg):
    o = out(cfg, state(thr=0.5, gim=(0.1, 0.1)), cmd(1.0, (0.05, 0.05)))
    assert o.xdot_act.shape == (N_STATE,)
    assert np.all(o.xdot_act[:13] == 0.0)
