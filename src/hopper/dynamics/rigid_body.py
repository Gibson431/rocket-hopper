"""Rigid-body equations of motion for a body with given total force and torque."""

import numpy as np

from hopper.math3d import cross, quat_derivative
from hopper.types import N_STATE, OMEGA, POS, QUAT, VEL


def rigid_body_derivative(
    x: np.ndarray,
    mass: float,
    inertia: np.ndarray,
    force_I: np.ndarray,  # noqa: N803 (frame suffix is the project convention)
    torque_B: np.ndarray,  # noqa: N803
) -> np.ndarray:
    """State derivative for POS, VEL, QUAT, OMEGA; all other entries are 0.

    `force_I` is the total external force in the inertial frame (gravity included),
    `torque_B` the total torque about the CoM in the body frame. omega_dot ignores
    d(I)/dt (and mass flow); S03 owns the propellant dynamics.
    """
    omega = x[OMEGA]
    xd = np.zeros(N_STATE)
    xd[POS] = x[VEL]
    xd[VEL] = force_I / mass
    xd[QUAT] = quat_derivative(x[QUAT], omega)
    xd[OMEGA] = np.linalg.solve(inertia, torque_B - cross(omega, inertia @ omega))
    return xd
