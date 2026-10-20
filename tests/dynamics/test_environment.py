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


def test_none_wind_is_zero_mean_without_gusts():
    w = Wind.none()
    assert np.array_equal(w.mean_i, np.zeros(3)) and w.gust_sigma == 0.0


def test_environment_default_has_no_wind_and_gravity_on():
    env = Environment.default()
    assert env.gravity_enabled
    assert np.array_equal(env.wind.mean_i, np.zeros(3)) and env.wind.gust_sigma == 0.0


@pytest.mark.parametrize("h,p", [(5000.0, 54019.9), (11000.0, 22632.1)])
def test_isa_pressure_matches_published_tables_tightly(h, p):
    assert atmosphere(h).pressure == pytest.approx(p, rel=2e-4)


def test_atmosphere_clamps_outside_troposphere():
    assert atmosphere(-50.0) == atmosphere(0.0)
    assert atmosphere(20000.0) == atmosphere(11000.0)
