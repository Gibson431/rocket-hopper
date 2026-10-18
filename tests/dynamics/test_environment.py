import numpy as np
import pytest

from hopper.dynamics.environment import Environment, Wind, atmosphere, gravity, pressure_to_altitude


def test_gravity_points_down_with_standard_magnitude():
    assert gravity(np.array([1.0, 2.0, 3.0])) == pytest.approx([0, 0, -9.80665], abs=1e-15)


def test_isa_sea_level_values():
    rho, temp, p, a = atmosphere(0.0)
    assert rho == pytest.approx(1.225, rel=1e-3)
    assert temp == pytest.approx(288.15, rel=1e-3)
    assert p == pytest.approx(101325.0, rel=1e-3)
    assert a == pytest.approx(340.29, rel=1e-3)


def test_isa_at_1000_m_matches_published_values():
    atm = atmosphere(1000.0)
    assert atm.density == pytest.approx(1.1117, rel=1e-2)
    assert atm.temperature == pytest.approx(281.65, rel=1e-2)
    assert atm.pressure == pytest.approx(89874.6, rel=1e-2)
    assert atm.speed_of_sound == pytest.approx(336.4, rel=1e-2)


@pytest.mark.parametrize("h", [0.0, 1.0, 250.0, 1000.0, 8000.0, 11000.0])
def test_pressure_to_altitude_inverts_atmosphere(h):
    assert pressure_to_altitude(atmosphere(h).pressure) == pytest.approx(h, abs=1e-6)


def test_zero_sigma_wind_returns_exactly_the_mean():
    mean = np.array([3.0, -2.0, 0.5])
    w = Wind(mean, 0.0, 2.0, np.random.default_rng(0))
    for _ in range(10):
        w.advance(0.01)
        assert np.array_equal(w.velocity(0.0, np.zeros(3)), mean)


def test_none_wind_is_zero():
    assert np.array_equal(Wind.none().velocity(1.0, np.zeros(3)), np.zeros(3))


def test_velocity_is_a_pure_read():
    w = Wind(np.zeros(3), 2.0, 1.0, np.random.default_rng(1))
    w.advance(0.01)
    assert np.array_equal(w.velocity(0.0, np.zeros(3)), w.velocity(5.0, np.ones(3)))


def test_gust_std_approaches_sigma():
    sigma, tau, dt = 2.0, 1.0, 0.05
    w = Wind(np.zeros(3), sigma, tau, np.random.default_rng(42))
    samples = []
    for _ in range(40000):
        w.advance(dt)
        samples.append(w.velocity(0.0, np.zeros(3)).copy())
    s = np.array(samples)
    assert s.std(axis=0) == pytest.approx([sigma] * 3, rel=0.1)


def test_gust_autocorrelation_decays_with_tau():
    sigma, tau, dt = 2.0, 1.0, 0.05
    w = Wind(np.zeros(3), sigma, tau, np.random.default_rng(7))
    s = []
    for _ in range(80000):
        w.advance(dt)
        s.append(w.velocity(0.0, np.zeros(3))[0])
    s = np.array(s)
    for lag_t, expected in ((tau, np.exp(-1)), (2 * tau, np.exp(-2))):
        k = round(lag_t / dt)
        rho = np.corrcoef(s[:-k], s[k:])[0, 1]
        assert rho == pytest.approx(expected, abs=0.05)


def test_reset_with_same_seed_reproduces_the_gust_sequence():
    w = Wind(np.zeros(3), 1.5, 1.0, np.random.default_rng(0))

    def run():
        out = []
        for _ in range(20):
            w.advance(0.01)
            out.append(w.velocity(0.0, np.zeros(3)).copy())
        return np.array(out)

    w.reset(5)
    a = run()
    w.reset(5)
    b = run()
    assert np.array_equal(a, b)
    w.reset(6)
    assert not np.array_equal(a, run())


def test_environment_default_has_no_wind_and_gravity_on():
    env = Environment.default()
    assert env.gravity_enabled
    assert np.array_equal(env.wind.velocity(0.0, np.zeros(3)), np.zeros(3))


@pytest.mark.parametrize("h,p", [(5000.0, 54019.9), (11000.0, 22632.1)])
def test_isa_pressure_matches_published_tables_tightly(h, p):
    assert atmosphere(h).pressure == pytest.approx(p, rel=2e-4)


def test_atmosphere_clamps_outside_troposphere():
    assert atmosphere(-50.0) == atmosphere(0.0)
    assert atmosphere(20000.0) == atmosphere(11000.0)
