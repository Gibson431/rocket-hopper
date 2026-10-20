"""Meshcat replay and visual geometry. No browser: poses are read back from the scene graph."""

import math

import numpy as np
import pandas as pd
import pytest
from pydrake.all import DiagramBuilder, Meshcat, RigidTransform, Role, RotationMatrix

from hopper.analysis.logger import COLUMNS
from hopper.drake.diagram import build_diagram
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.simulation import DrakeSimulation
from hopper.drake.viz import Replay, record_replay, replay
from hopper.dynamics.environment import Environment, Wind
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import quat_to_dcm
from hopper.types import Command, make_state


@pytest.fixture(scope="module")
def meshcat():
    return Meshcat()


def synthetic_log(cfg, n: int = 121, t_end: float = 2.0) -> pd.DataFrame:
    t = np.linspace(0.0, t_end, n)
    df = pd.DataFrame(0.0, index=range(n), columns=COLUMNS[:-1])
    df["t"] = t
    df["pos_x"], df["pos_y"], df["pos_z"] = np.sin(t), 0.3 * t, 2.0 + t
    ang = 0.4 * t
    axis = np.array([1.0, 2.0, 0.5]) / np.linalg.norm([1.0, 2.0, 0.5])
    df["quat_w"] = np.cos(ang / 2)
    df["quat_x"], df["quat_y"], df["quat_z"] = (np.sin(ang / 2) * a for a in axis)
    df["prop"] = cfg.tank.prop_mass_initial * (1 - t / (2 * t_end))
    df["thr"] = 0.6
    df["thrust"] = 1500.0
    df["gim_x"], df["gim_y"] = 0.05, -0.04
    df["mass"] = [mass_properties(p, cfg).mass for p in df["prop"]]
    df["event"] = ""
    df.loc[n - 1, "event"] = "LANDED"
    df["mission_state"] = "HOVER"
    return df


def test_visual_geometry_has_no_proximity_role_and_adds_the_expected_shapes(cfg):
    builder = DiagramBuilder()
    plant, sg = build_vehicle_plant(cfg, builder, finalize=False)
    plant.Finalize()
    insp = sg.model_inspector()
    assert insp.NumGeometriesWithRole(Role.kProximity) == 0  # legs add the collision geometry
    # body cylinder, nose cone, one per leg, ground plane
    assert insp.NumGeometriesWithRole(Role.kIllustration) == 3 + cfg.legs.count


def test_visual_geometry_does_not_change_the_dynamics(cfg):
    def run(visual: bool):
        x0 = make_state([0, 0, 1.5], [0, 0, 0], None, [0, 0, 0], cfg.tank.prop_mass_initial)
        sim = DrakeSimulation(
            cfg, _Thrusting(), Environment(wind=Wind.none()), x0, np.random.default_rng(3)
        )
        sim.rig = build_diagram(
            cfg, sim.controller, Environment(wind=Wind.none()), np.random.default_rng(3),
            visual=visual,
        )  # fmt: skip
        return sim.run(0.3).log

    pd.testing.assert_frame_equal(run(True), run(False), check_exact=True)


class _Thrusting:
    def reset(self):
        pass

    def step_state(self, est, t):
        return Command(0.7, np.array([0.02, -0.01]), 0.0)


def test_live_mode_publishes_geometry_to_meshcat_while_running(cfg, meshcat):
    x0 = make_state([0, 0, 1.5], [0, 0, 0], None, [0, 0, 0], cfg.tank.prop_mass_initial)
    sim = DrakeSimulation(
        cfg, _Thrusting(), Environment(wind=Wind.none()), x0, np.random.default_rng(0),
        meshcat=meshcat,
    )  # fmt: skip
    sim.run(0.1)
    assert meshcat.HasPath("/drake/visualizer")


def test_replayed_poses_equal_the_logged_poses_to_1e9(cfg, meshcat):
    log = synthetic_log(cfg)
    seen = []
    rp = Replay(cfg, meshcat)
    rp.play(log, speed=math.inf, fps=None, on_frame=lambda k, pose: seen.append((k, pose)))
    assert [k for k, _ in seen] == list(range(len(log)))
    for k, pose in seen:
        row = log.iloc[k]
        q = row[["quat_w", "quat_x", "quat_y", "quat_z"]].to_numpy(float)
        rot = quat_to_dcm(q / np.linalg.norm(q))
        com = np.array([0.0, 0.0, mass_properties(row["prop"], cfg).com_z])
        np.testing.assert_allclose(pose.rotation().matrix(), rot, atol=1e-9)
        # the logged position is the CoM; the body frame sits at the gimbal pivot
        np.testing.assert_allclose(
            pose.translation() + rot @ com,
            row[["pos_x", "pos_y", "pos_z"]].to_numpy(float),
            atol=1e-9,
        )


def test_replay_frame_rate_limits_the_rows_shown(cfg, meshcat):
    log = synthetic_log(cfg, n=2001, t_end=2.0)  # 1 kHz log
    ks = []
    Replay(cfg, meshcat).play(log, speed=math.inf, fps=50, on_frame=lambda k, pose: ks.append(k))
    assert 95 <= len(ks) <= 105
    assert ks[0] == 0 and ks[-1] == len(log) - 1  # first and terminal rows always shown


def test_replay_draws_plume_trail_and_overlay(cfg, meshcat):
    log = synthetic_log(cfg)
    rp = replay(log, cfg, speed=math.inf, meshcat=meshcat)
    assert meshcat.HasPath(rp.plume_path)
    assert meshcat.HasPath(rp.trail_path)
    assert "HOVER" in rp.overlay_text and "LANDED" in rp.overlay_text and "2.00" in rp.overlay_text


def test_plume_length_scales_with_logged_thrust_and_vanishes_at_zero(cfg, meshcat):
    rp = Replay(cfg, meshcat)
    assert rp.plume_length(0.0) == 0.0
    assert rp.plume_length(2000.0) == pytest.approx(2 * rp.plume_length(1000.0))


def test_plume_runs_from_the_nozzle_to_a_tip_opposite_the_thrust_direction(cfg, meshcat):
    rp = Replay(cfg, meshcat)
    gx, gy = 0.05, -0.04
    d_b = np.array([np.cos(gx) * np.sin(gy), -np.sin(gx), np.cos(gx) * np.cos(gy)])
    rot = quat_to_dcm(np.array([0.9, 0.1, -0.3, 0.2]) / np.linalg.norm([0.9, 0.1, -0.3, 0.2]))
    pivot = np.array([1.0, 2.0, 3.0])
    x_wb = RigidTransform(RotationMatrix(rot), pivot)
    m = rp.plume_matrix(x_wb, (gx, gy), 1000.0)
    length = rp.plume_length(1000.0)
    np.testing.assert_allclose((m @ [0, 0, 1, 1])[:3], pivot, atol=1e-12)  # cone base at nozzle
    np.testing.assert_allclose(m[:3, 3], pivot - length * rot @ d_b, atol=1e-12)  # apex at tip


def test_replay_rejects_a_non_positive_speed(cfg, meshcat):
    with pytest.raises(ValueError):
        replay(synthetic_log(cfg), cfg, speed=0.0, meshcat=meshcat)


def test_record_replay_writes_a_non_empty_self_contained_html(cfg, tmp_path):
    out = tmp_path / "run.html"
    record_replay(synthetic_log(cfg, n=61), cfg, out)
    text = out.read_text()
    assert len(text) > 10_000
    assert "<html" in text.lower()
