import numpy as np
import pytest
from hopper_control.attitude import AttitudeController, AttitudeGains
from hopper_control.attitude_test_controller import AttitudeHoldController
from hopper_control.interface import StateFeedbackController
from hopper_core.math3d import quat_error, quat_from_axis_angle
from hopper_core.types import make_state

from hopper.config import VehicleConfig
from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.scenario_glue import control_params_from_config

IDENTITY = np.array([1.0, 0.0, 0.0, 0.0])
ZERO3 = np.zeros(3)
PROP_MASS = 10.0


@pytest.fixture
def params(cfg: VehicleConfig):
    return control_params_from_config(cfg)


@pytest.fixture
def gains():
    return AttitudeGains.from_bandwidth(wn=[3.0, 3.0, 1.5], zeta=0.8)


@pytest.fixture
def ctrl(params, gains):
    return AttitudeController(params, gains)


def update(ctrl, params, q=IDENTITY, omega=ZERO3, q_des=IDENTITY, thrust=1000.0, prop=PROP_MASS):
    m = params.dry_mass + prop
    com_z = params.mass_properties(m)[0]
    return ctrl.update(q_des, ZERO3, q, omega, thrust, m, com_z, 0.01)


# --- unit ---------------------------------------------------------------------------------


def test_zero_error_gives_zero_torque_and_zero_command(ctrl, params):
    out = update(ctrl, params)
    assert np.allclose(out.torque_des, 0.0, atol=1e-12)
    assert np.allclose(out.gimbal, 0.0, atol=1e-12)
    assert out.roll_torque == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_small_positive_error_gives_positive_torque_about_that_axis(ctrl, params, axis):
    e = np.zeros(3)
    e[axis] = 0.05
    q = quat_from_axis_angle(e, -0.05)  # vehicle is 0.05 rad short of q_des about the axis
    out = update(ctrl, params, q=q)
    assert out.torque_des[axis] > 0.0
    others = np.delete(out.torque_des, axis)
    assert np.allclose(others, 0.0, atol=1e-9)


def test_rate_damping_opposes_rotation(ctrl, params):
    out = update(ctrl, params, omega=np.array([0.1, -0.2, 0.3]))
    assert np.all(np.sign(out.torque_des) == np.array([-1, 1, -1]))


def test_torque_matches_inertia_times_pd_law(ctrl, params, gains):
    m = params.dry_mass + PROP_MASS
    inertia = params.mass_properties(m)[1]
    q = quat_from_axis_angle([1, 0, 0], -0.1)
    omega = np.array([0.02, 0.0, 0.0])
    out = update(ctrl, params, q=q, omega=omega)
    expected = inertia[0] * (gains.kp[0] * 0.1 - gains.kd[0] * 0.02)
    assert out.torque_des[0] == pytest.approx(expected, rel=1e-9)


def test_inertia_is_scheduled_from_estimated_mass(ctrl, params):
    q = quat_from_axis_angle([0, 1, 0], -0.05)
    heavy = update(ctrl, params, q=q, prop=20.0).torque_des[1]
    light = update(ctrl, params, q=q, prop=0.0).torque_des[1]
    assert heavy > light > 0.0


def test_error_takes_the_shortest_path(ctrl, params):
    q = quat_from_axis_angle([0, 0, 1], np.radians(179.0))
    a = update(ctrl, params, q=q).torque_des[2]
    b = update(ctrl, params, q=-q).torque_des[2]
    assert a == pytest.approx(b)
    assert a < 0.0  # 179 deg about +z: shortest way back is negative


def test_gimbal_held_at_zero_below_thrust_floor(ctrl, params):
    q = quat_from_axis_angle([1, 0, 0], -0.1)
    out = update(ctrl, params, q=q, thrust=0.0)
    assert np.all(out.gimbal == 0.0)
    assert np.isfinite(out.roll_torque)


def test_roll_uses_rcs_even_when_engine_is_off(ctrl, params):
    q = quat_from_axis_angle([0, 0, 1], -0.3)
    out = update(ctrl, params, q=q, thrust=0.0)
    assert out.roll_torque > 0.0
    assert np.all(out.gimbal == 0.0)


def test_roll_command_is_clipped_to_unit_range(ctrl, params):
    q = quat_from_axis_angle([0, 0, 1], -np.radians(120))
    assert update(ctrl, params, q=q).roll_torque == 1.0
    assert (
        update(ctrl, params, q=quat_from_axis_angle([0, 0, 1], np.radians(120))).roll_torque == -1.0
    )


def test_saturated_gimbal_stays_within_limits_and_is_flagged(ctrl, params):
    q = quat_from_axis_angle([1, 1, 0], -np.radians(60))
    out = update(ctrl, params, q=q)
    lim = params.gimbal_max
    assert out.saturated
    assert np.all(np.abs(out.gimbal) <= lim + 1e-12)
    assert np.max(np.abs(out.gimbal)) == pytest.approx(lim)


def test_saturation_preserves_the_gimbal_direction(ctrl, params):
    q = quat_from_axis_angle([2, 1, 0], -np.radians(60))
    out = update(ctrl, params, q=q)
    # gx = -tau_x/(z T), gy = -tau_y/(z T) in the small-angle limit, so gy/gx = tau_y/tau_x = 1/2
    assert out.gimbal[1] / out.gimbal[0] == pytest.approx(0.5, rel=0.05)


def test_output_is_finite_for_degenerate_inputs(ctrl, params):
    out = update(ctrl, params, thrust=1e-12)
    assert np.all(np.isfinite(out.gimbal))


# --- torque-to-gimbal mapping inverts the thrust-vector geometry ----------------------------------


@pytest.mark.parametrize("thrust", [1000.0, 1700.0, 2500.0])
@pytest.mark.parametrize("tau", [(30.0, -20.0), (-45.0, 10.0), (5.0, 60.0)])
def test_gimbal_command_produces_the_demanded_torque_about_the_com(cfg, params, ctrl, thrust, tau):
    mp = mass_properties(PROP_MASS, cfg)
    gx, gy = ctrl.torque_to_gimbal(np.array(tau), thrust, mp.com_z)
    # Thrust direction of a gimbal (gx about x_B, gy about y_B); the pivot is com_z below the CoM.
    force = thrust * np.array([np.cos(gx) * np.sin(gy), -np.sin(gx), np.cos(gx) * np.cos(gy)])
    torque = np.cross([0.0, 0.0, -mp.com_z], force)
    assert torque[0] == pytest.approx(tau[0], rel=1e-9)
    assert torque[1] == pytest.approx(tau[1], rel=1e-9)


def test_control_params_match_vehicle_config(cfg, params):
    assert params.thrust_max == cfg.engine.thrust_max
    assert params.gimbal_max == cfg.engine.gimbal_max
    for prop in (0.0, 7.0, 20.0):
        mp = mass_properties(prop, cfg)
        com_z, inertia = params.mass_properties(params.dry_mass + prop)
        assert com_z == pytest.approx(mp.com_z)
        assert np.allclose(inertia, np.diag(mp.inertia))


def test_gains_from_bandwidth():
    g = AttitudeGains.from_bandwidth(wn=3.0, zeta=0.7)
    assert np.allclose(g.kp, 9.0)
    assert np.allclose(g.kd, 2 * 0.7 * 3.0)


def test_hold_controller_satisfies_state_feedback_protocol(params, gains):
    assert isinstance(AttitudeHoldController(params, gains, throttle=0.5), StateFeedbackController)


# --- closed loop ----------------------------------------------------------------------------


def run_closed_loop(cfg, params, gains, q0, omega0=(0, 0, 0), t_end=5.0, delay=0.0, q_des=None):
    ctrl = AttitudeHoldController(params, gains, throttle=0.45, q_des=q_des)
    x0 = make_state([0, 0, 50], [0, 0, 0], q0, omega0, PROP_MASS, thr=0.45)
    sim = DrakeSimulation(
        cfg,
        ctrl,
        Environment.default(),
        x0,
        np.random.default_rng(0),
        command_delay=delay,
        log_decimation=10,
    )
    return sim.run(t_end).log


def tilt_deg(log):
    # tilt of z_B from vertical, from the logged quaternion
    q = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy()
    cos_tilt = 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)
    return np.degrees(np.arccos(np.clip(cos_tilt, -1, 1)))


def test_closed_loop_removes_10_degree_tilt_without_large_overshoot(cfg, params, gains):
    q0 = quat_from_axis_angle([1, 1, 0], np.radians(10))
    log = run_closed_loop(cfg, params, gains, q0)
    tilt = tilt_deg(log)
    assert tilt[0] == pytest.approx(10.0, abs=0.01)
    assert tilt[log.t >= 5.0 - 1e-9].max() < 0.5
    assert tilt[log.t >= 4.0].max() < 0.5
    # No overshoot: the tilt never grows beyond 30% of the initial error after it first
    # passes within 1 degree (tilt is a magnitude, so overshoot shows as re-growth).
    first_small = np.argmax(tilt < 1.0)
    assert tilt[first_small:].max() < 3.0


def test_closed_loop_removes_30_degree_roll_error_with_rcs(cfg, params, gains):
    q0 = quat_from_axis_angle([0, 0, 1], np.radians(30))
    log = run_closed_loop(cfg, params, gains, q0, t_end=10.0)
    q = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy()[-1]
    err = quat_error(IDENTITY, q)
    assert np.degrees(np.linalg.norm(err)) < 1.0
    assert abs(log["omega_z"].iloc[-1]) < 0.02


def test_closed_loop_holds_a_nonzero_attitude_target(cfg, params, gains):
    q_des = quat_from_axis_angle([0, 1, 0], np.radians(5))
    # Short run: once the vehicle picks up speed, the aero torque (CP above the CoM) pushes the
    # PD loop off target by a growing offset (about 2 degrees at 10 s). No integral term here.
    log = run_closed_loop(cfg, params, gains, IDENTITY, q_des=q_des, t_end=3.0)
    q = log[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy()[-1]
    assert np.degrees(np.linalg.norm(quat_error(q_des, q))) < 0.5


def test_closed_loop_stable_with_20ms_delay_and_actuator_lag(cfg, params, gains):
    q0 = quat_from_axis_angle([1, 1, 0], np.radians(10))
    log = run_closed_loop(cfg, params, gains, q0, delay=0.020, t_end=6.0)
    tilt = tilt_deg(log)
    assert tilt[log.t >= 5.0].max() < 1.0
    assert tilt.max() < 20.0


def test_closed_loop_with_300ms_delay_is_unstable(cfg, params, gains):
    # Measured margin (10 degree start, wn = 3 rad/s): settles to ~0.2 degrees with up to 100 ms of
    # delay, but the tilt grows to ~16 degrees at 200 ms and ~70 degrees at 300 ms. This documents
    # the margin; it is not a requirement that the loop survives such delay.
    q0 = quat_from_axis_angle([1, 1, 0], np.radians(10))
    log = run_closed_loop(cfg, params, gains, q0, delay=0.300, t_end=6.0)
    tilt = tilt_deg(log)
    assert np.all(np.isfinite(tilt))
    assert tilt[log.t >= 5.0].max() > 10.0  # worse than the initial error: diverging


def test_closed_loop_60_degree_tilt_saturates_without_nan_or_chatter(cfg, params, gains):
    q0 = quat_from_axis_angle([1, 0, 0], np.radians(60))
    log = run_closed_loop(cfg, params, gains, q0, t_end=8.0)
    assert np.all(np.isfinite(log.drop(columns="event").to_numpy(dtype=float)))
    lim = cfg.engine.gimbal_max
    assert log[["cmd_gim_x", "cmd_gim_y"]].abs().to_numpy().max() <= lim + 1e-12
    sign_changes = (np.diff(np.sign(log["cmd_gim_x"].to_numpy())) != 0).sum()
    assert sign_changes <= 6
    assert tilt_deg(log)[-1] < tilt_deg(log)[0]


# --- slew limit, reset, CoM scheduling in the hold controller --------------------------------


def test_gimbal_command_change_per_call_obeys_the_actuator_rate_limit(ctrl, params):
    dt = 0.01
    m = params.dry_mass + PROP_MASS
    com_z = params.mass_properties(m)[0]
    ctrl.update(IDENTITY, ZERO3, IDENTITY, ZERO3, 1000.0, m, com_z, dt)
    q = quat_from_axis_angle([1, 0, 0], -np.radians(20))
    first = ctrl.update(IDENTITY, ZERO3, q, ZERO3, 1000.0, m, com_z, dt)
    assert np.max(np.abs(first.gimbal)) == pytest.approx(params.gimbal_rate_max * dt)


def test_reset_clears_the_slew_limit_memory(ctrl, params):
    m = params.dry_mass + PROP_MASS
    com_z = params.mass_properties(m)[0]
    q = quat_from_axis_angle([1, 0, 0], -np.radians(20))
    ctrl.update(IDENTITY, ZERO3, IDENTITY, ZERO3, 1000.0, m, com_z, 0.01)
    ctrl.reset()
    out = ctrl.update(IDENTITY, ZERO3, q, ZERO3, 1000.0, m, com_z, 0.01)
    assert np.max(np.abs(out.gimbal)) > params.gimbal_rate_max * 0.01 * 2


def test_hold_controller_uses_the_center_of_mass_for_the_estimated_mass(cfg, params, gains):
    from hopper_core.types import StateEstimate

    mass = params.dry_mass + params.prop_mass_initial
    q = quat_from_axis_angle([1, 0, 0], -np.radians(2))
    est = StateEstimate(0.0, ZERO3, ZERO3, q, ZERO3, mass)
    hold = AttitudeHoldController(params, gains, throttle=0.5)
    cmd = hold.step_state(est, 0.0)

    ref = AttitudeController(params, gains)
    com_z = params.mass_properties(mass)[0]
    assert com_z != pytest.approx(params.dry_com_z, rel=0.01)
    want = ref.update(IDENTITY, ZERO3, q, ZERO3, 0.5 * params.thrust_max, mass, com_z, 0.0)
    assert np.allclose(cmd.gimbal, want.gimbal, rtol=1e-12)


def test_saturated_flag_is_set_when_roll_torque_exceeds_the_rcs(ctrl, params):
    q = quat_from_axis_angle([0, 0, 1], -np.radians(90))
    out = update(ctrl, params, q=q)
    assert out.saturated
    assert np.all(out.gimbal == 0.0)


def test_vehicle_without_rcs_commands_no_roll_and_is_not_flagged_saturated(cfg, gains):
    p = control_params_from_config(cfg.model_copy(update={"rcs": None}))
    ctrl = AttitudeController(p, gains)
    out = update(ctrl, p, q=quat_from_axis_angle([0, 0, 1], -0.3))
    assert out.roll_torque == 0.0
    assert not out.saturated
    assert np.all(np.isfinite(out.gimbal))
