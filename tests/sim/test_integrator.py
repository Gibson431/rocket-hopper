import numpy as np
import pytest

from hopper.sim.integrator import integrate, rk4_step
from hopper.types import QUAT


def decay(t, x):
    return -x


def test_rk4_step_matches_exponential_decay_to_fifth_order_local_error():
    dt = 0.1
    x = rk4_step(decay, 0.0, np.array([1.0]), dt, renorm=None)
    assert x[0] == pytest.approx(np.exp(-dt), abs=1e-6)


def test_rk4_step_passes_time_to_the_derivative():
    # x' = t  ->  x = t^2 / 2, exact for RK4
    x = rk4_step(lambda t, x: np.array([t]), 0.0, np.array([0.0]), 2.0, renorm=None)
    assert x[0] == pytest.approx(2.0, abs=1e-12)


def test_rk4_step_renormalises_only_the_quaternion_slice():
    x0 = np.zeros(19)
    x0[QUAT] = [1.0, 0.0, 0.0, 0.0]
    x0[0] = 5.0

    def f(t, x):
        xd = np.zeros_like(x)
        xd[QUAT] = [1.0, 0.0, 0.0, 0.0]  # grows the norm
        xd[0] = 1.0
        return xd

    x1 = rk4_step(f, 0.0, x0, 1.0)
    assert np.linalg.norm(x1[QUAT]) == pytest.approx(1.0, abs=1e-15)
    assert x1[0] == pytest.approx(6.0, abs=1e-12)


def test_rk4_step_does_not_mutate_its_input():
    x0 = np.array([1.0, 2.0])
    rk4_step(decay, 0.0, x0, 0.1, renorm=None)
    assert x0.tolist() == [1.0, 2.0]


def test_integrate_hits_end_time_exactly_when_not_divisible_by_dt():
    times = []

    def f(t, x):
        times.append(t)
        return np.array([1.0])  # x = t

    x, traj = integrate(f, np.array([0.0]), 0.0, 1.0, 0.3, renorm=None, record=True)
    assert x[0] == pytest.approx(1.0, abs=1e-12)
    assert traj[-1, 0] == pytest.approx(1.0, abs=1e-12)
    assert max(times) <= 1.0 + 1e-12


def test_integrate_trajectory_includes_initial_and_final_samples():
    x, traj = integrate(decay, np.array([1.0]), 0.0, 1.0, 0.25, renorm=None, record=True)
    assert traj.shape == (5, 1)
    assert traj[0, 0] == 1.0
    assert traj[-1, 0] == x[0]


def test_integrate_returns_no_trajectory_by_default():
    _, traj = integrate(decay, np.array([1.0]), 0.0, 1.0, 0.25, renorm=None)
    assert traj is None


def test_integrate_calls_callback_each_step_with_time_and_state():
    seen = []
    integrate(
        decay, np.array([1.0]), 0.0, 1.0, 0.4, renorm=None, callback=lambda t, x: seen.append(t)
    )
    assert seen == pytest.approx([0.4, 0.8, 1.0])


def test_integrate_zero_length_interval_returns_initial_state():
    x, _ = integrate(decay, np.array([3.0]), 2.0, 2.0, 0.1, renorm=None)
    assert x[0] == 3.0


def test_integrate_rejects_nonpositive_dt():
    with pytest.raises(ValueError):
        integrate(decay, np.array([1.0]), 0.0, 1.0, 0.0, renorm=None)


def test_integrate_does_not_mutate_initial_state():
    x0 = np.array([1.0])
    integrate(decay, x0, 0.0, 1.0, 0.1, renorm=None)
    assert x0[0] == 1.0


def test_integrate_takes_exactly_three_steps_for_a_divisible_span():
    seen = []
    integrate(
        decay, np.array([1.0]), 0.0, 0.3, 0.1, renorm=None, callback=lambda t, x: seen.append(t)
    )
    assert seen == pytest.approx([0.1, 0.2, 0.3])
    assert seen[-1] == 0.3
