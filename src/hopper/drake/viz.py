"""Meshcat visualisation: visual geometry for the plant and a replay of a logged run."""

import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from pydrake.all import (
    Box,
    Cylinder,
    Diagram,
    DiagramBuilder,
    Meshcat,
    MeshcatCone,
    MeshcatVisualizer,
    Rgba,
    RigidTransform,
    RotationMatrix,
)
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.legs import foot_points
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.state import BODY_NAME
from hopper.dynamics.mass_model import mass_properties

NOSE_LENGTH = 0.4  # m
LEG_RADIUS = 0.02  # m
LEG_ATTACH_Z = 0.3  # m above the pivot, where the struts meet the body
GROUND_SIZE = 40.0  # m
PLUME_PER_NEWTON = 1.0 / 2500.0  # m of plume per N of thrust (1 m at the default full thrust)
PLUME_RADIUS = 0.1  # m
TRAIL_EVERY = 10  # frames between trail updates
OVERLAY_PERIOD = 0.25  # s of log time between overlay updates

_BODY_COLOR = np.array([0.85, 0.85, 0.88, 1.0])
_NOSE_COLOR = np.array([0.8, 0.25, 0.2, 1.0])
_LEG_COLOR = np.array([0.3, 0.3, 0.35, 1.0])
_GROUND_COLOR = np.array([0.45, 0.55, 0.45, 1.0])
_PLUME_COLOR = Rgba(1.0, 0.6, 0.1, 0.8)


def body_radius(cfg: VehicleConfig) -> float:
    return math.sqrt(cfg.aero.area_axial / math.pi)


def body_length(cfg: VehicleConfig) -> float:
    return 2.0 * cfg.mass.dry_com_z  # the dry inertia is that of a uniform cylinder


def _align_z(v: np.ndarray) -> RotationMatrix:
    """Rotation taking +z to the direction of `v`."""
    z, v = np.array([0.0, 0.0, 1.0]), v / np.linalg.norm(v)
    axis = np.cross(z, v)
    s, c = np.linalg.norm(axis), z @ v
    if s < 1e-12:
        return RotationMatrix() if c > 0 else RotationMatrix.MakeXRotation(math.pi)
    k = axis / s
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return RotationMatrix(np.eye(3) + s * kx + (1 - c) * kx @ kx)  # Rodrigues


def add_visual_geometry(plant: MultibodyPlant, scene_graph, cfg: VehicleConfig) -> None:
    """Body cylinder, nose cone, one strut per leg and a ground plane; illustration role only, so
    nothing here takes part in contact or dynamics. Call before `plant.Finalize()`."""
    body = plant.GetBodyByName(BODY_NAME)
    r, length = body_radius(cfg), body_length(cfg)
    plant.RegisterVisualGeometry(
        body, RigidTransform([0, 0, length / 2]), Cylinder(r, length), "body_visual", _BODY_COLOR
    )
    flip = RotationMatrix.MakeXRotation(math.pi)  # the cone's apex is at its origin, +z is the base
    plant.RegisterVisualGeometry(
        body,
        RigidTransform(flip, [0, 0, length + NOSE_LENGTH]),
        MeshcatCone(NOSE_LENGTH, r, r),
        "nose_visual",
        _NOSE_COLOR,
    )
    for i, foot in enumerate(foot_points(cfg)):
        a = r * foot[:2] / np.linalg.norm(foot[:2]) if np.linalg.norm(foot[:2]) > 0 else foot[:2]
        top = np.array([a[0], a[1], LEG_ATTACH_Z])
        strut = foot - top
        plant.RegisterVisualGeometry(
            body,
            RigidTransform(_align_z(strut), (top + foot) / 2),
            Cylinder(LEG_RADIUS, float(np.linalg.norm(strut))),
            f"leg{i}_visual",
            _LEG_COLOR,
        )
    plant.RegisterVisualGeometry(
        plant.world_body(),
        RigidTransform([0, 0, -0.01]),
        Box(GROUND_SIZE, GROUND_SIZE, 0.02),
        "ground_visual",
        _GROUND_COLOR,
    )


def frame_indices(t: np.ndarray, fps: float | None) -> list[int]:
    """Rows to show at about `fps` frames per second of log time; first and last always shown."""
    n = len(t)
    if fps is None:
        return list(range(n))
    keep, last = [0], t[0]
    for k in range(1, n - 1):
        if t[k] - last >= 1.0 / fps - 1e-9:
            keep.append(k)
            last = t[k]
    if n > 1:
        keep.append(n - 1)
    return keep


class Replay:
    """The vehicle drawn from logged rows. Owns a plant + scene graph + Meshcat visualizer; the
    plume, trail and overlay are Meshcat objects outside the plant."""

    plume_path = "/replay/plume"
    trail_path = "/replay/trail"

    def __init__(self, cfg: VehicleConfig, meshcat: Meshcat | None = None):
        self.cfg = cfg
        self.meshcat = meshcat if meshcat is not None else Meshcat()
        builder = DiagramBuilder()
        self.plant, self.scene_graph = build_vehicle_plant(builder=builder, cfg=cfg)
        MeshcatVisualizer.AddToBuilder(builder, self.scene_graph, self.meshcat)
        self.diagram: Diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self._plant_ctx = self.plant.GetMyMutableContextFromRoot(self.context)
        self._body = self.plant.GetBodyByName(BODY_NAME)
        self.overlay_text = ""
        self._overlay_t = -math.inf
        self.recording = False
        self.meshcat.SetObject(
            self.plume_path, MeshcatCone(1.0, PLUME_RADIUS, PLUME_RADIUS), _PLUME_COLOR
        )

    # --- geometry of one row ---
    def body_pose(self, row) -> RigidTransform:
        """Pose of the body frame (gimbal pivot): the logged position is the CoM."""
        q = np.array([row["quat_w"], row["quat_x"], row["quat_y"], row["quat_z"]], dtype=float)
        rot = RotationMatrix(_rot_from_quat(q))
        com = np.array([0.0, 0.0, mass_properties(row["prop"], self.cfg).com_z])
        p = np.array([row["pos_x"], row["pos_y"], row["pos_z"]], dtype=float)
        return RigidTransform(rot, p - rot.matrix() @ com)

    def plume_length(self, thrust: float) -> float:
        return PLUME_PER_NEWTON * max(float(thrust), 0.0)

    def plume_matrix(self, x_wb: RigidTransform, gimbal, thrust: float) -> np.ndarray:
        """4x4 (with scale) taking the unit cone (apex at the origin, base at z = 1) to a plume with
        its base at the nozzle and its apex at the tip, opposite the thrust direction."""
        gx, gy = gimbal
        d_b = np.array([np.cos(gx) * np.sin(gy), -np.sin(gx), np.cos(gx) * np.cos(gy)])
        length = max(self.plume_length(thrust), 1e-6)
        rot = x_wb.rotation().matrix()
        z_axis = rot @ d_b  # from tip to base
        r_cone = _align_z(z_axis).matrix()
        m = np.eye(4)
        m[:3, :3] = r_cone @ np.diag([1.0, 1.0, length])
        m[:3, 3] = x_wb.translation() - length * z_axis
        return m

    # --- drawing ---
    def show(self, log: pd.DataFrame, k: int, trail: bool = True) -> RigidTransform:
        row = log.iloc[k]
        t = float(row["t"])
        pose = self.body_pose(row)
        self.context.SetTime(t)
        self.plant.SetFreeBodyPose(self._plant_ctx, self._body, pose)
        gimbal = (float(row["gim_x"]), float(row["gim_y"]))
        thrust = float(row["thrust"])
        m = self.plume_matrix(pose, gimbal, thrust)
        if self.recording:
            # a 4x4 matrix cannot be recorded; pose and scale are separate animation tracks
            length = np.linalg.norm(m[:3, 2])
            self.meshcat.SetTransform(
                self.plume_path,
                RigidTransform(RotationMatrix(m[:3, :3] / np.array([1, 1, length])), m[:3, 3]),
                t,
            )
            self.meshcat.SetProperty(self.plume_path, "scale", [1.0, 1.0, float(length)], t)
        else:
            self.meshcat.SetTransform(self.plume_path, m)
        self.meshcat.SetProperty(
            self.plume_path, "visible", thrust > 0.0, *(([t]) if self.recording else [])
        )
        self.diagram.ForcedPublish(self.context)
        if trail and not self.recording and (k % TRAIL_EVERY == 0 or k == len(log) - 1):
            self._set_trail(log, k)
        self._update_overlay(row, force=k == len(log) - 1)
        return pose

    def _set_trail(self, log: pd.DataFrame, k: int) -> None:
        pts = log[["pos_x", "pos_y", "pos_z"]].to_numpy(float)[: k + 1].T
        if pts.shape[1] >= 2:
            self.meshcat.SetLine(self.trail_path, pts, 2.0, Rgba(0.1, 0.4, 0.9, 1.0))

    def _update_overlay(self, row, force: bool = False) -> None:
        """Meshcat has no text overlay: the status is a button label in the controls panel."""
        t = float(row["t"])
        phase = str(row["mission_state"]) if "mission_state" in row.index else "-"
        event = str(row["event"]) if "event" in row.index and row["event"] else "RUNNING"
        text = f"t = {t:.2f} s | {phase} | {event}"
        old_phase = self.overlay_text.split(" | ")[1:2]
        if not force and t - self._overlay_t < OVERLAY_PERIOD and old_phase == [phase]:
            return
        if self.overlay_text:
            self.meshcat.DeleteButton(self.overlay_text, False)
        self.meshcat.AddButton(text)
        self.overlay_text, self._overlay_t = text, t

    def play(
        self,
        log: pd.DataFrame,
        speed: float = 1.0,
        fps: float | None = 60.0,
        on_frame: Callable[[int, RigidTransform], None] | None = None,
    ) -> None:
        """Show the rows at `fps` log-time frames per second, paced at `speed` times real time
        (`math.inf` does not wait)."""
        if not speed > 0:
            raise ValueError(f"speed must be > 0, got {speed}")
        t = log["t"].to_numpy(float)
        start = time.monotonic()
        for k in frame_indices(t, fps):
            if math.isfinite(speed):
                wait = (t[k] - t[0]) / speed - (time.monotonic() - start)
                if wait > 0:
                    time.sleep(wait)
            pose = self.show(log, k)
            if on_frame is not None:
                on_frame(k, pose)
        if self.recording:
            self._set_trail(log, len(log) - 1)


def _rot_from_quat(q: np.ndarray) -> np.ndarray:
    from hopper.math3d import quat_to_dcm

    return quat_to_dcm(q / np.linalg.norm(q))


def replay(
    log: pd.DataFrame,
    cfg: VehicleConfig,
    speed: float = 1.0,
    meshcat: Meshcat | None = None,
    fps: float | None = 60.0,
) -> Replay:
    """Play a logged run in Meshcat (a new server when `meshcat` is None)."""
    rp = Replay(cfg, meshcat)
    rp.play(log, speed=speed, fps=fps)
    return rp


def record_replay(
    log: pd.DataFrame, cfg: VehicleConfig, path: str | Path, fps: float = 30.0
) -> None:
    """Write a self-contained Meshcat HTML animation of the run (no server needed to view it)."""
    meshcat = Meshcat()
    rp = Replay(cfg, meshcat)
    meshcat.StartRecording(frames_per_second=fps, set_visualizations_while_recording=False)
    rp.recording = True
    rp.play(log, speed=math.inf, fps=fps)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    Path(path).write_text(meshcat.StaticHtml())
