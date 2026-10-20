import numpy as np
import pytest
from pydrake.all import DiagramBuilder

from hopper.drake.legs import ContactObserver, add_legs
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_state_vector
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import quat_from_axis_angle, quat_to_dcm
from hopper.types import make_state

G = 9.80665


def build(cfg):
    builder = DiagramBuilder()
    plant, scene_graph = build_vehicle_plant(cfg, builder, finalize=False)
    legs = add_legs(plant, scene_graph, cfg)
    plant.Finalize()
    obs = builder.AddSystem(ContactObserver(plant, legs))
    builder.Connect(plant.get_state_output_port(), obs.get_plant_state_input_port())
    builder.Connect(plant.get_contact_results_output_port(), obs.get_contact_results_input_port())
    return builder.Build(), plant, obs


def foot_offsets(cfg):
    """Foot points from the gimbal pivot, body frame."""
    a = 2 * np.pi * np.arange(cfg.legs.count) / cfg.legs.count
    return np.stack(
        [
            cfg.legs.radius * np.cos(a),
            cfg.legs.radius * np.sin(a),
            np.full(cfg.legs.count, -cfg.legs.length),
        ],
        axis=1,
    )


def state_with_lowest_foot(cfg, depth, tilt=0.0, vel=(0, 0, 0), prop=20.0):
    """State vector whose lowest foot point is `depth` below the ground (negative: above)."""
    q = quat_from_axis_angle([0, 1, 0], tilt)
    mp = mass_properties(prop, cfg)
    low = (foot_offsets(cfg) @ quat_to_dcm(q).T)[:, 2].min()
    pivot = np.array([0.0, 0.0, -depth - low])
    return make_state(pivot + quat_to_dcm(q) @ [0, 0, mp.com_z], vel, q, [0, 0, 0], prop)


def outputs(cfg, x):
    diagram, plant, obs = build(cfg)
    ctx = diagram.CreateDefaultContext()
    set_state_vector(plant, plant.GetMyMutableContextFromRoot(ctx), x, cfg)
    oc = obs.GetMyContextFromRoot(ctx)
    return (
        obs.get_foot_forces_output_port().Eval(oc),
        obs.get_contact_count_output_port().Eval(oc)[0],
        obs.get_total_force_output_port().Eval(oc),
        obs.get_base_strike_output_port().Eval(oc)[0],
    )


def test_one_collision_geometry_per_leg(cfg):
    _, plant, _ = build(cfg)
    body = plant.GetBodyByName("hopper")
    assert len(plant.GetCollisionGeometriesForBody(body)) == cfg.legs.count


def test_add_legs_needs_a_scene_graph(cfg):
    plant, _ = build_vehicle_plant(cfg, None, finalize=False)
    with pytest.raises(ValueError):
        add_legs(plant, None, cfg)


def test_no_contact_in_flight(cfg):
    forces, n, total, strike = outputs(cfg, state_with_lowest_foot(cfg, -0.5))
    assert np.all(forces == 0.0) and n == 0 and np.all(total == 0.0) and strike == 0.0


def test_foot_force_is_the_series_stiffness_times_depth_at_rest(cfg):
    depth = 0.005
    forces, n, total, strike = outputs(cfg, state_with_lowest_foot(cfg, depth))
    expected = cfg.legs.stiffness * depth  # k_series = legs.stiffness (foot sphere is 10 mm)
    assert forces == pytest.approx(np.full(cfg.legs.count, expected), rel=0.02)
    assert n == cfg.legs.count
    assert total[2] == pytest.approx(forces.sum(), rel=1e-6)
    assert strike == 0.0


def test_only_the_lowest_foot_reports_force_and_keeps_its_index(cfg):
    # Positive rotation about y lowers the foot on +x, which is foot 0.
    forces, n, total, _ = outputs(cfg, state_with_lowest_foot(cfg, 0.004, tilt=0.15))
    assert n == 1
    # The tilted sphere's lowest point is 10 mm * (1 - cos tilt) below the foot point.
    depth = 0.004 + 0.01 * (1 - np.cos(0.15))
    assert forces[0] == pytest.approx(cfg.legs.stiffness * depth, rel=0.02)
    assert np.all(forces[1:] == 0.0)
    assert total[2] == pytest.approx(forces[0], rel=0.05)


def test_damping_adds_force_while_a_foot_is_moving_into_the_ground(cfg):
    still = outputs(cfg, state_with_lowest_foot(cfg, 0.005))[0].sum()
    sinking = outputs(cfg, state_with_lowest_foot(cfg, 0.005, vel=(0, 0, -1.0)))[0].sum()
    assert sinking > still * 1.5


def test_base_strike_when_the_gimbal_pivot_is_below_ground(cfg):
    mp = mass_properties(20.0, cfg)
    x = make_state([0, 0, mp.com_z - 0.05], [0, 0, 0], None, [0, 0, 0], 20.0)
    assert outputs(cfg, x)[3] == 1.0
    assert outputs(cfg, state_with_lowest_foot(cfg, 0.005))[3] == 0.0


def _drop(cfg, height, vz=0.0, t_end=3.0):
    diagram, plant, obs = build(cfg)
    sim = make_simulator(diagram)
    ctx = sim.get_mutable_context()
    pctx = plant.GetMyMutableContextFromRoot(ctx)
    set_state_vector(plant, pctx, state_with_lowest_foot(cfg, -height, vel=(0, 0, vz)), cfg)
    sim.Initialize()
    mp = mass_properties(20.0, cfg)
    zs, fs = [], []
    for k in range(int(t_end / 1e-3)):
        sim.AdvanceTo((k + 1) * 1e-3)
        s = plant.GetPositionsAndVelocities(pctx)
        zs.append(s[6] - cfg.legs.length)  # foot height of the (upright) vehicle
        fs.append(obs.get_foot_forces_output_port().Eval(obs.GetMyContextFromRoot(ctx)).sum())
    return np.array(zs), np.array(fs), s, mp


def test_drop_from_half_a_metre_settles_upright_without_bouncing(cfg):
    zs, fs, s, mp = _drop(cfg, 0.5)
    assert np.all(np.isfinite(zs)) and np.all(np.isfinite(s))
    lowest = int(np.argmin(zs))
    assert zs.max() <= 0.5 + 1e-6  # never above the drop height
    assert zs[lowest:].max() < 0.01 and zs[lowest + 20 :].max() < 0.0  # no rebound off the ground
    # Rest: weight carried by the feet at the series-spring depth m g / (n k).
    assert zs[-1] == pytest.approx(-mp.mass * G / (cfg.legs.count * cfg.legs.stiffness), abs=2e-4)
    assert fs[-1] == pytest.approx(mp.mass * G, rel=0.01)
    assert np.ptp(zs[-500:]) < 1e-5 and np.ptp(fs[-500:]) < 1.0  # no jitter
    assert abs(s[4]) < 1e-3 and abs(s[5]) < 1e-3  # no lateral drift


def test_hard_five_metre_per_second_landing_stays_finite_without_rebound(cfg):
    zs, fs, s, _ = _drop(cfg, 0.02, vz=-5.0, t_end=2.0)
    assert np.all(np.isfinite(zs)) and np.all(np.isfinite(fs)) and np.all(np.isfinite(s))
    lowest = int(np.argmin(zs))
    assert 0.0 < -zs[lowest] < 0.2  # bounded penetration
    assert zs[lowest + 20 :].max() < 0.0
    assert zs[-1] == pytest.approx(zs[-500:].mean(), abs=1e-4)


def test_sliding_friction_is_mu_times_the_normal_force(cfg):
    _, _, total, _ = outputs(cfg, state_with_lowest_foot(cfg, 0.005, vel=(1.0, 0, 0)))
    assert total[0] == pytest.approx(-cfg.legs.friction * total[2], rel=0.02)
    assert abs(total[1]) < 1e-6 * total[2]


def test_friction_ramps_up_over_the_stiction_tolerance(cfg):
    # Slip at a fifth of the tolerance gives a small fraction of full friction (a frictionless or
    # tolerance-free model gives 0 or the full mu N).
    _, _, total, _ = outputs(cfg, state_with_lowest_foot(cfg, 0.005, vel=(0.01, 0, 0)))
    ratio = -total[0] / total[2]
    assert 0.02 * cfg.legs.friction < ratio < 0.3 * cfg.legs.friction
