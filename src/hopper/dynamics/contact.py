"""Leg/ground contact: spring-damper normal force with regularised Coulomb friction."""

from dataclasses import dataclass

import numpy as np

from hopper.config import LegsConfig
from hopper.dynamics.mass_model import MassProperties
from hopper.math3d import quat_to_dcm
from hopper.types import OMEGA, POS, QUAT, VEL

# Friction regularisation width. 0.01 m/s (the step's first choice) makes yaw friction stiff enough
# (4 mu N r^2 / (eps I_zz) ~ 7000 /s) to leave RK4 unstable at dt = 1 ms; 0.05 m/s is stable.
V_EPS = 0.05  # m/s


@dataclass(frozen=True)
class ContactOutput:
    force_I: np.ndarray  # noqa: N815  (3,) total contact force, inertial
    torque_B: np.ndarray  # noqa: N815  (3,) about the CoM, body frame
    foot_forces: np.ndarray  # (n,) normal force per foot (N)
    any_contact: bool
    base_strike: bool  # engine base (z = 0 on the vehicle) is below ground


def contact_loads(x: np.ndarray, mp: MassProperties, cfg: LegsConfig) -> ContactOutput:
    """Ground (plane z = 0) loads on the vehicle.

    Foot i sits at body position (from the CoM)
    `p_i = [r cos a_i, r sin a_i, -(com_z + length)]`, `r = cfg.radius`, `a_i = 2 pi i / count`:
    the engine base is `com_z` below the CoM and the feet are `length` below the base.
    With penetration `d = -z_foot > 0`, the normal force is `max(0, k d - c v_z)` (no tension).
    Friction opposes foot slip with magnitude `mu N tanh(|v_t| / V_EPS)`, V_EPS = 0.05 m/s.
    """
    rot = quat_to_dcm(x[QUAT])
    angles = 2 * np.pi * np.arange(cfg.count) / cfg.count
    p = np.stack(
        [
            cfg.radius * np.cos(angles),
            cfg.radius * np.sin(angles),
            np.full(cfg.count, -(mp.com_z + cfg.length)),
        ],
        axis=1,
    )  # (n, 3) body
    r_i = p @ rot.T  # CoM -> foot, inertial
    foot_pos = x[POS] + r_i
    foot_vel = x[VEL] + (np.cross(x[OMEGA], p)) @ rot.T

    depth = -foot_pos[:, 2]
    normal = np.where(
        depth > 0.0, np.maximum(0.0, cfg.stiffness * depth - cfg.damping * foot_vel[:, 2]), 0.0
    )

    v_t = foot_vel[:, :2]
    slip = np.linalg.norm(v_t, axis=1)
    safe = np.where(slip > 1e-12, slip, 1.0)
    # tanh(s/eps)/s -> 1/eps as s -> 0, so the friction force is continuous through zero slip.
    gain = np.where(slip > 1e-12, np.tanh(slip / V_EPS) / safe, 1.0 / V_EPS)
    f_t = -cfg.friction * normal[:, None] * gain[:, None] * v_t

    forces = np.column_stack([f_t, normal])  # (n, 3) inertial
    torque_i = np.cross(r_i, forces).sum(axis=0)

    base_z = x[POS][2] + (rot @ np.array([0.0, 0.0, -mp.com_z]))[2]
    return ContactOutput(
        force_I=forces.sum(axis=0),
        torque_B=rot.T @ torque_i,
        foot_forces=normal,
        any_contact=bool(np.any(normal > 0.0)),
        base_strike=bool(base_z < 0.0),
    )
