import numpy as np
import pytest

from hopper.dynamics.aero import aero_loads
from hopper.dynamics.mass_model import MassProperties
from hopper.math3d import quat_from_axis_angle

RHO = 1.2
ID = np.array([1.0, 0, 0, 0])
ZERO = np.zeros(3)


@pytest.fixture
def mp():
    return MassProperties(100.0, 0.8, np.eye(3))


def test_zero_relative_velocity_gives_zero_loads_without_nan(cfg, mp):
    f, t = aero_loads(ZERO, ID, ZERO, RHO, mp, cfg.aero)
    assert np.array_equal(f, ZERO) and np.array_equal(t, ZERO)


def test_axial_drag_opposes_velocity_with_quadratic_magnitude(cfg, mp):
    v = 30.0
    f, t = aero_loads(np.array([0, 0, v]), ID, ZERO, RHO, mp, cfg.aero)
    a = cfg.aero
    assert f[2] == pytest.approx(-0.5 * RHO * a.cd_axial * a.area_axial * v**2, rel=1e-9)
    assert f[:2] == pytest.approx([0, 0], abs=1e-12)
    assert t == pytest.approx(ZERO, abs=1e-12)  # pure axial flow: no moment


def test_lateral_drag_opposes_velocity_with_quadratic_magnitude(cfg, mp):
    v = 20.0
    f, _ = aero_loads(np.array([0, -v, 0]), ID, ZERO, RHO, mp, cfg.aero)
    a = cfg.aero
    assert f[1] == pytest.approx(0.5 * RHO * a.cd_lateral * a.area_lateral * v**2, rel=1e-9)
    assert f[[0, 2]] == pytest.approx([0, 0], abs=1e-12)


def test_drag_acts_on_velocity_relative_to_the_wind(cfg, mp):
    f1, _ = aero_loads(np.array([5.0, 0, 0]), ID, ZERO, RHO, mp, cfg.aero)
    f2, _ = aero_loads(np.array([15.0, 0, 0]), ID, np.array([10.0, 0, 0]), RHO, mp, cfg.aero)
    assert f2 == pytest.approx(f1, rel=1e-12)


def test_drag_is_computed_in_body_axes_for_a_tilted_vehicle(cfg, mp):
    # z_B -> +x_I, so inertial +x motion is axial flow.
    q = quat_from_axis_angle([0, 1, 0], np.pi / 2)
    v = 25.0
    f, _ = aero_loads(np.array([v, 0, 0]), q, ZERO, RHO, mp, cfg.aero)
    a = cfg.aero
    assert f == pytest.approx([-0.5 * RHO * a.cd_axial * a.area_axial * v**2, 0, 0], rel=1e-9)


def test_weathercock_turns_nose_into_relative_wind_when_cp_above_com(cfg, mp):
    assert cfg.aero.cp_z > mp.com_z
    # Wind blowing toward +x: relative velocity is -x. Nose should swing toward +x, i.e. +y torque.
    _, t = aero_loads(ZERO, ID, np.array([10.0, 0, 0]), RHO, mp, cfg.aero)
    assert t[1] > 0 and abs(t[0]) < 1e-12 and abs(t[2]) < 1e-12


def test_cp_below_com_flips_the_torque_sign(cfg, mp):
    low = cfg.aero.model_copy(update={"cp_z": 0.3})
    _, up = aero_loads(ZERO, ID, np.array([10.0, 0, 0]), RHO, mp, cfg.aero)
    _, down = aero_loads(ZERO, ID, np.array([10.0, 0, 0]), RHO, mp, low)
    assert up[1] > 0 > down[1]


def test_torque_equals_lever_arm_times_lateral_force(cfg, mp):
    v = 12.0
    f, t = aero_loads(np.array([v, 0, 0]), ID, ZERO, RHO, mp, cfg.aero)
    assert t[1] == pytest.approx((cfg.aero.cp_z - mp.com_z) * f[0], rel=1e-9)
