import numpy as np
import pytest

from hopper.dynamics.rigid_body import rigid_body_derivative
from hopper.math3d import quat_from_axis_angle, quat_to_dcm
from hopper.sim.integrator import integrate
from hopper.types import N_STATE, OMEGA, POS, PROP, QUAT, VEL, make_state

G = 9.81
MASS = 100.0
INERTIA = np.diag([20.0, 30.0, 8.0])
ZERO3 = np.zeros(3)


def propagate(x0, t1, dt, force=ZERO3, torque=ZERO3, mass=MASS, inertia=INERTIA):
    def f(t, x):
        return rigid_body_derivative(x, mass, inertia, force, torque)

    x, _ = integrate(f, x0, 0.0, t1, dt)
    return x


def tumble_state():
    q = quat_from_axis_angle([1.0, 2.0, 3.0], 0.7)
    return make_state([0, 0, 0], [0, 0, 0], q, [2.0, -1.5, 3.0], prop=0.0)


def test_derivative_has_full_state_length_and_zero_outside_rigid_block():
    x = make_state([1, 2, 3], [0, 0, 0], None, [0.1, 0.2, 0.3], prop=20.0, thr=0.5, gim=[0.1, 0.1])
    xd = rigid_body_derivative(x, MASS, INERTIA, ZERO3, ZERO3)
    assert xd.shape == (N_STATE,)
    assert xd[PROP] == 0.0
    assert np.all(xd[14:] == 0.0)


def test_free_fall_matches_analytic_position_and_velocity():
    x0 = make_state([0, 0, 100.0], [0, 0, 0], None, ZERO3, prop=0.0)
    x = propagate(x0, 5.0, 1e-3, force=np.array([0, 0, -MASS * G]))
    assert x[POS][2] == pytest.approx(100.0 - G * 25 / 2, abs=1e-9)
    assert x[VEL][2] == pytest.approx(-G * 5.0, abs=1e-9)


def test_constant_force_gives_uniform_acceleration_in_every_axis():
    v0 = np.array([1.0, -2.0, 0.5])
    p0 = np.array([3.0, 4.0, 5.0])
    force = np.array([200.0, -50.0, 80.0])
    x0 = make_state(p0, v0, None, ZERO3, prop=0.0)
    t = 3.0
    x = propagate(x0, t, 1e-3, force=force)
    assert x[POS] == pytest.approx(p0 + v0 * t + force * t**2 / (2 * MASS), abs=1e-9)
    assert x[VEL] == pytest.approx(v0 + force * t / MASS, abs=1e-9)


def test_force_is_interpreted_in_the_inertial_frame_not_the_body_frame():
    q = quat_from_axis_angle([1, 0, 0], np.pi / 2)
    x0 = make_state(ZERO3, ZERO3, q, ZERO3, prop=0.0)
    xd = rigid_body_derivative(x0, MASS, INERTIA, np.array([0, 0, 100.0]), ZERO3)
    assert xd[VEL] == pytest.approx([0, 0, 1.0])


def test_position_derivative_is_velocity():
    x0 = make_state(ZERO3, [1.0, 2.0, 3.0], None, ZERO3, prop=0.0)
    assert rigid_body_derivative(x0, MASS, INERTIA, ZERO3, ZERO3)[POS] == pytest.approx(
        [1.0, 2.0, 3.0]
    )


def test_symmetric_top_spin_about_symmetry_axis_stays_constant():
    inertia = np.diag([20.0, 20.0, 8.0])
    x0 = make_state(ZERO3, ZERO3, None, [0.0, 0.0, 5.0], prop=0.0)
    x = propagate(x0, 10.0, 1e-3, inertia=inertia)
    assert x[OMEGA] == pytest.approx([0.0, 0.0, 5.0], abs=1e-12)


def test_asymmetric_body_conserves_inertial_angular_momentum():
    x0 = tumble_state()

    def momentum(x):
        return quat_to_dcm(x[QUAT]) @ (INERTIA @ x[OMEGA])

    x = propagate(x0, 10.0, 1e-3)
    assert momentum(x) == pytest.approx(momentum(x0), abs=1e-9)
    assert np.linalg.norm(momentum(x0)) > 10.0  # the case is not trivially still


def test_torque_free_kinetic_energy_is_conserved():
    x0 = tumble_state()
    x = propagate(x0, 10.0, 1e-3)
    ke0 = 0.5 * x0[OMEGA] @ INERTIA @ x0[OMEGA]
    ke = 0.5 * x[OMEGA] @ INERTIA @ x[OMEGA]
    assert ke == pytest.approx(ke0, abs=1e-8)


def test_torque_free_tumble_actually_exchanges_angular_velocity_between_axes():
    x0 = tumble_state()
    x = propagate(x0, 2.0, 1e-3)
    assert np.linalg.norm(x[OMEGA] - x0[OMEGA]) > 0.1


def test_constant_torque_about_principal_axis_gives_linear_rate_and_quadratic_angle():
    tau, t = 12.0, 4.0
    x0 = make_state(ZERO3, ZERO3, None, ZERO3, prop=0.0)
    x = propagate(x0, t, 1e-3, torque=np.array([0.0, tau, 0.0]))
    i_yy = INERTIA[1, 1]
    assert x[OMEGA] == pytest.approx([0.0, tau * t / i_yy, 0.0], abs=1e-10)
    expected = quat_from_axis_angle([0, 1, 0], tau * t**2 / (2 * i_yy))
    assert x[QUAT] == pytest.approx(expected, abs=1e-10)


def test_quaternion_norm_stays_unit_after_ten_seconds_of_tumbling():
    x = propagate(tumble_state(), 10.0, 1e-3)
    assert np.linalg.norm(x[QUAT]) == pytest.approx(1.0, abs=1e-12)


def test_integration_converges_at_fourth_order_on_a_nonlinear_tumble():
    x0 = tumble_state()
    t1 = 5.0
    ref = propagate(x0, t1, 1e-3 / 4)

    def err(dt):
        x = propagate(x0, t1, dt)
        return np.linalg.norm(x[6:13] - ref[6:13])

    dt = 0.04
    e1, e2, e3 = err(dt), err(dt / 2), err(dt / 4)
    assert e1 / e2 > 10
    assert e2 / e3 > 10
    assert e3 < 1e-6
