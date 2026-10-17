"""Full vehicle state derivative: rigid body + gravity + propulsion (+ optional environment)."""

import numpy as np

from hopper.config import VehicleConfig
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.propulsion import G0, propulsion
from hopper.dynamics.rigid_body import rigid_body_derivative
from hopper.math3d import quat_rotate
from hopper.types import PROP, QUAT, Command


def vehicle_derivative(
    t: float,
    x: np.ndarray,
    cmd: Command,
    cfg: VehicleConfig,
    env_force_I: np.ndarray | None = None,  # noqa: N803 (frame suffix is the project convention)
    env_torque_B: np.ndarray | None = None,  # noqa: N803
    g: float = G0,
) -> np.ndarray:
    """d(state)/dt of the 19-element state for the commanded engine.

    `env_force_I` (N, inertial) and `env_torque_B` (N m about the CoM, body) are the hooks for
    aero, wind and ground contact (S04, S05). `g` is flat constant gravity (m/s^2); tests set 0.

    Neglected (small for a slow burn): d(inertia)/dt and CoM motion terms from propellant
    depletion, and the momentum carried by the exhaust. `t` is unused (time-invariant dynamics).
    """
    mp = mass_properties(x[PROP], cfg)
    p = propulsion(x, cmd, cfg, mp)

    force_i = quat_rotate(x[QUAT], p.force_B) + np.array([0.0, 0.0, -mp.mass * g])
    torque_b = p.torque_B
    if env_force_I is not None:
        force_i = force_i + env_force_I
    if env_torque_B is not None:
        torque_b = torque_b + env_torque_B

    xd = rigid_body_derivative(x, mp.mass, mp.inertia, force_i, torque_b)
    # Rigid-body part leaves the actuator slots at 0, so the sum merges the two.
    return xd + p.xdot_act
