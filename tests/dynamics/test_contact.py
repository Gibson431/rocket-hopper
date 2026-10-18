import numpy as np
import pytest

from hopper.dynamics.contact import ContactOutput, contact_loads
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.math3d import quat_from_axis_angle, quat_rotate, quat_to_dcm
from hopper.sim.integrator import integrate
from hopper.types import OMEGA, POS, PROP, QUAT, VEL, Command, make_state

G0 = 9.80665
OFF = Command(throttle=0.0, gimbal=np.zeros(2), roll_torque=0.0)


def standing(cfg, lift=0.0, vel=(0, 0, 0), omega=(0, 0, 0), quat=None, prop=20.0, thr=0.0):
    """Upright at rest with the feet `lift` metres above the ground (negative = embedded)."""
    com_z = mass_properties(prop, cfg).com_z
    z = com_z + cfg.legs.length + lift
    return make_state([0, 0, z], vel, quat, omega, prop, thr)


def loads(cfg, x):
    return contact_loads(x, mass_properties(x[PROP], cfg), cfg.legs)


def tilt_of(x):
    return np.arccos(np.clip(quat_to_dcm(x[QUAT])[2, 2], -1, 1))


def simulate(cfg, x0, t1, cmd=OFF, record=False):
    return integrate(
        lambda t, x: vehicle_derivative(t, x, cmd, cfg), x0, 0.0, t1, cfg.sim.dt, record=record
    )


def test_no_contact_force_when_all_feet_are_above_ground(cfg):
    out = loads(cfg, standing(cfg, lift=0.01))
    assert isinstance(out, ContactOutput)
    assert not out.any_contact
    assert not out.base_strike
    assert np.all(out.force_I == 0.0)
    assert np.all(out.torque_B == 0.0)
    assert out.foot_forces.shape == (cfg.legs.count,)
    assert np.all(out.foot_forces == 0.0)


def test_static_deflection_force_is_stiffness_times_penetration(cfg):
    out = loads(cfg, standing(cfg, lift=-0.005))
    assert out.any_contact
    assert out.foot_forces == pytest.approx(
        [cfg.legs.stiffness * 0.005] * cfg.legs.count, rel=1e-12
    )
    assert out.force_I == pytest.approx([0, 0, 4 * cfg.legs.stiffness * 0.005], rel=1e-12)
    assert out.torque_B == pytest.approx([0, 0, 0], abs=1e-9)


def test_damping_adds_force_when_moving_down_and_never_pulls_when_separating(cfg):
    down = loads(cfg, standing(cfg, lift=-0.005, vel=(0, 0, -1.0)))
    assert down.foot_forces[0] == pytest.approx(
        cfg.legs.stiffness * 0.005 + cfg.legs.damping * 1.0, rel=1e-12
    )
    up = loads(cfg, standing(cfg, lift=-0.001, vel=(0, 0, 5.0)))
    assert np.all(up.foot_forces == 0.0)  # damping would exceed the spring: clamp, no tension
    assert np.all(up.force_I == 0.0)


def test_foot_geometry_matches_leg_circle_below_the_base(cfg):
    # Tilt about y by a small angle: the foot at +x goes down and is the only one touching.
    ang = 0.05
    q = quat_from_axis_angle([0, 1, 0], ang)
    com_z = mass_properties(20.0, cfg).com_z
    r, ln = cfg.legs.radius, cfg.legs.length
    foot_z = -(com_z + ln)
    p0 = np.array([r, 0.0, foot_z])
    z_foot0 = quat_rotate(q, p0)[2]
    z_com = -z_foot0 - 0.002  # lowest foot 2 mm into the ground
    x = make_state([0, 0, z_com], [0, 0, 0], q, [0, 0, 0], 20.0)
    out = loads(cfg, x)
    assert out.foot_forces[0] == pytest.approx(cfg.legs.stiffness * 0.002, rel=1e-9)
    assert np.all(out.foot_forces[1:] == 0.0)
    # Torque is r x F about the CoM (r from the CoM to the loaded foot), in the body frame.
    tau_i = np.cross(quat_rotate(q, p0), out.force_I)
    assert quat_rotate(q, out.torque_B) == pytest.approx(tau_i, abs=1e-9)


def test_friction_opposes_slip_and_is_bounded_by_coulomb(cfg):
    x = standing(cfg, lift=-0.005, vel=(2.0, 0, 0))
    out = loads(cfg, x)
    n_total = out.foot_forces.sum()
    assert out.force_I[0] == pytest.approx(-cfg.legs.friction * n_total, rel=1e-6)  # saturated
    assert out.force_I[1] == pytest.approx(0.0, abs=1e-9)
    slow = loads(cfg, standing(cfg, lift=-0.005, vel=(0.005, 0, 0)))
    assert -cfg.legs.friction * n_total < slow.force_I[0] < 0.0  # regularised, not saturated


def test_base_strike_when_engine_base_goes_below_ground(cfg):
    com_z = mass_properties(20.0, cfg).com_z
    q = quat_from_axis_angle([0, 1, 0], np.deg2rad(80))
    base_offset_z = quat_rotate(q, np.array([0, 0, -com_z]))[2]
    ok = make_state([0, 0, -base_offset_z + 0.01], [0, 0, 0], q, [0, 0, 0], 20.0)
    bad = make_state([0, 0, -base_offset_z - 0.01], [0, 0, 0], q, [0, 0, 0], 20.0)
    assert not loads(cfg, ok).base_strike
    assert loads(cfg, bad).base_strike
    assert not loads(cfg, standing(cfg, lift=0.0)).base_strike


def test_vehicle_derivative_includes_contact_and_exposes_info(cfg):
    x = standing(cfg, lift=-0.005)
    info: dict = {}
    xd = vehicle_derivative(0.0, x, OFF, cfg, info=info)
    out = loads(cfg, x)
    mass = mass_properties(20.0, cfg).mass
    assert xd[VEL][2] == pytest.approx(-G0 + out.force_I[2] / mass, rel=1e-9)
    assert isinstance(info["contact"], ContactOutput)
    assert info["contact"].force_I == pytest.approx(out.force_I)


def test_drop_settles_with_foot_forces_supporting_weight_and_no_lateral_drift(cfg):
    x, _ = simulate(cfg, standing(cfg, lift=0.001), 4.0)
    weight = mass_properties(20.0, cfg).mass * G0
    out = loads(cfg, x)
    assert out.foot_forces.sum() == pytest.approx(weight, rel=0.01)
    assert np.linalg.norm(x[POS][:2]) < 1e-4
    assert np.linalg.norm(x[VEL]) < 1e-3
    assert tilt_of(x) < 1e-4


def test_resting_normal_force_is_steady_at_default_dt(cfg):
    x, _ = simulate(cfg, standing(cfg, lift=0.001), 3.0)
    _, traj = simulate(cfg, x, 0.5, record=True)
    forces = np.array([loads(cfg, row).foot_forces for row in traj])
    assert np.ptp(forces, axis=0).max() / forces.mean() < 0.01


def test_sideways_push_while_standing_decelerates_and_stops(cfg):
    x0 = standing(cfg, lift=-0.0082, vel=(0.3, 0, 0))
    x, traj = simulate(cfg, x0, 3.0, record=True)
    assert np.linalg.norm(x[VEL][:2]) < 1e-3
    assert 0.0 < x[0] < 0.1  # slid a little: v^2 / (2 mu g) = 0.0076 m, plus compliance
    assert max(tilt_of(r) for r in traj) < np.deg2rad(10)


def test_spinning_in_place_on_the_ground_decays_without_a_numerical_limit_cycle(cfg):
    x0 = standing(cfg, lift=-0.0082, omega=(0, 0, 0.5))
    x, _ = simulate(cfg, x0, 1.0)
    assert abs(x[OMEGA][2]) < 1e-4


def test_tilting_past_footprint_angle_tips_over_but_a_small_tilt_recovers(cfg):
    com_z = mass_properties(20.0, cfg).com_z
    cg_height = com_z + cfg.legs.length
    limit = np.arctan(cfg.legs.radius / cg_height)

    def tilted(angle):
        q = quat_from_axis_angle([0, 1, 0], angle)
        foot = quat_rotate(q, np.array([cfg.legs.radius, 0.0, -cg_height]))
        return make_state([0, 0, -foot[2] - 0.003], [0, 0, 0], q, [0, 0, 0], 20.0)

    _, traj = simulate(cfg, tilted(limit + np.deg2rad(5)), 3.0, record=True)
    tilts = np.array([tilt_of(r) for r in traj])
    assert tilts[-1] > np.deg2rad(60)
    assert tilts[-1] > tilts[len(tilts) // 2] > tilts[0]  # keeps growing

    _, traj = simulate(cfg, tilted(limit - np.deg2rad(8)), 3.0, record=True)
    assert max(tilt_of(r) for r in traj) < limit


def test_landing_from_a_small_drop_never_gains_mechanical_energy(cfg):
    mass_p = mass_properties(20.0, cfg)
    x0 = standing(cfg, lift=0.2, vel=(0, 0, -1.0), omega=(0.0, 0.0, 0.0))
    # A little tilt so rotation and friction are exercised too.
    x0[QUAT] = quat_from_axis_angle([1, 1, 0], 0.03)
    k = cfg.legs.stiffness
    foot_drop = mass_p.com_z + cfg.legs.length

    def energy(x):
        rot = quat_to_dcm(x[QUAT])
        z_feet = [
            (x[POS] + rot @ foot)[2]
            for foot in (
                np.array([cfg.legs.radius * np.cos(a), cfg.legs.radius * np.sin(a), -foot_drop])
                for a in 2 * np.pi * np.arange(cfg.legs.count) / cfg.legs.count
            )
        ]
        spring = 0.5 * k * sum(min(z, 0.0) ** 2 for z in z_feet)
        kin = 0.5 * mass_p.mass * x[VEL] @ x[VEL] + 0.5 * x[OMEGA] @ mass_p.inertia @ x[OMEGA]
        return kin + mass_p.mass * G0 * x[2] + spring

    _, traj = simulate(cfg, x0, 3.0, record=True)
    e = np.array([energy(r) for r in traj])
    assert (e[1:] - e[0]).max() < 0.05
    assert e[-1] < e[0] - 100.0  # most of the drop energy is dissipated


def test_thrust_above_weight_lifts_off_and_contact_goes_to_zero(cfg):
    full = Command(throttle=1.0, gimbal=np.zeros(2), roll_torque=0.0)
    x0 = standing(cfg, lift=-0.0082, thr=1.0)
    x, _ = simulate(cfg, x0, 1.0, cmd=full)
    assert not loads(cfg, x).any_contact
    assert x[VEL][2] > 5.0
    assert x[PROP] < 20.0
