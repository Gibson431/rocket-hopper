import numpy as np
import pytest

from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.math3d import quat_from_axis_angle
from hopper.sim.integrator import integrate
from hopper.types import OMEGA, PROP, VEL, Command, make_state

G0 = 9.80665


def cmd(throttle=0.0, gimbal=(0.0, 0.0), roll=0.0):
    return Command(throttle=throttle, gimbal=np.array(gimbal), roll_torque=roll)


def upright(prop=20.0, thr=0.0, vel=(0, 0, 0), omega=(0, 0, 0), **kw):
    return make_state([0, 0, 10.0], vel, None, omega, prop, thr, **kw)


def test_free_fall_with_engine_off_accelerates_at_g(cfg):
    xd = vehicle_derivative(0.0, upright(thr=0.0), cmd(0.0), cfg)
    assert xd[VEL] == pytest.approx([0, 0, -G0], abs=1e-12)


def test_hover_when_thrust_equals_weight(cfg):
    # Min throttle exceeds hover thrust for this vehicle, so lower it for the test.
    low = cfg.model_copy(update={"engine": cfg.engine.model_copy(update={"thrust_min_frac": 0.1})})
    mass = mass_properties(20.0, low).mass
    frac = mass * G0 / low.engine.thrust_max
    xd = vehicle_derivative(0.0, upright(thr=frac), cmd(frac), low)
    assert xd[VEL] == pytest.approx([0, 0, 0], abs=1e-9)


def test_tilted_thrust_rotates_into_inertial_frame(cfg):
    q = quat_from_axis_angle([0, 1, 0], np.pi / 2)  # z_B -> +x_I
    x = make_state([0, 0, 10], [0, 0, 0], q, [0, 0, 0], 20.0, thr=1.0)
    xd = vehicle_derivative(0.0, x, cmd(1.0), cfg, g=0.0)
    mass = mass_properties(20.0, cfg).mass
    assert xd[VEL] == pytest.approx([cfg.engine.thrust_max / mass, 0, 0], abs=1e-9)


def test_environment_hooks_add_force_and_torque(cfg):
    x = upright(thr=0.0)
    base = vehicle_derivative(0.0, x, cmd(), cfg)
    mp = mass_properties(20.0, cfg)
    f = np.array([10.0, -20.0, 30.0])
    tq = np.array([1.0, 2.0, 3.0])
    xd = vehicle_derivative(0.0, x, cmd(), cfg, env_force_I=f, env_torque_B=tq)
    assert xd[VEL] - base[VEL] == pytest.approx(f / mp.mass)
    assert xd[OMEGA] - base[OMEGA] == pytest.approx(np.linalg.solve(mp.inertia, tq))


def test_roll_torque_gives_pure_z_angular_acceleration(cfg):
    x = upright(thr=0.0)
    xd = vehicle_derivative(0.0, x, cmd(0.0, roll=1.0), cfg)
    izz = mass_properties(20.0, cfg).inertia[2, 2]
    assert xd[OMEGA] == pytest.approx([0, 0, cfg.rcs.max_roll_torque / izz], rel=1e-12)


def test_gimbal_tilt_produces_pitch_acceleration_with_correct_sign(cfg):
    x = upright(thr=1.0, gim=(0.0, 0.1))
    xd = vehicle_derivative(0.0, x, cmd(1.0, (0.0, 0.1)), cfg)
    assert xd[OMEGA][1] < 0


def test_actuator_derivatives_are_merged_into_result(cfg):
    x = upright(thr=0.0)
    xd = vehicle_derivative(0.0, x, cmd(1.0), cfg)
    assert xd[14] == pytest.approx(1.0 / cfg.engine.throttle_tau)


def test_rocket_equation_delta_v_with_no_gravity(cfg):
    throttle = 0.8
    x0 = upright(thr=throttle)
    m0 = mass_properties(x0[PROP], cfg).mass
    c = cmd(throttle)

    def f(t, x):
        return vehicle_derivative(t, x, c, cfg, g=0.0)

    x, _ = integrate(f, x0, 0.0, 10.0, 1e-3)
    m1 = mass_properties(x[PROP], cfg).mass
    assert m1 < m0
    expected = cfg.engine.isp * G0 * np.log(m0 / m1)
    assert x[VEL][2] == pytest.approx(expected, rel=1e-3)
    assert x[VEL][:2] == pytest.approx([0, 0], abs=1e-9)
