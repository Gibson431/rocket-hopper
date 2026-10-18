"""Quasi-steady drag and weathercock moment on the body."""

import numpy as np

from hopper.config import AeroConfig
from hopper.dynamics.mass_model import MassProperties
from hopper.math3d import cross, quat_rotate, quat_rotate_inv


def aero_loads(
    vel_i: np.ndarray,
    quat: np.ndarray,
    wind_i: np.ndarray,
    rho: float,
    mp: MassProperties,
    cfg: AeroConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (force_I, torque_B about the CoM) from the velocity relative to the air.

    Axial and lateral drag act opposite their relative-velocity components, each scaled by the
    total speed (magnitude 0.5 rho Cd A v^2 for pure axial or lateral flow). The force acts at
    the centre of pressure; a CP above the CoM weathercocks the nose into the wind.
    """
    v_b = quat_rotate_inv(quat, vel_i - wind_i)
    speed = np.linalg.norm(v_b)
    k = -0.5 * rho * speed
    force_b = k * np.array(
        [
            cfg.cd_lateral * cfg.area_lateral * v_b[0],
            cfg.cd_lateral * cfg.area_lateral * v_b[1],
            cfg.cd_axial * cfg.area_axial * v_b[2],
        ]
    )
    r = np.array([0.0, 0.0, cfg.cp_z - mp.com_z])
    return quat_rotate(quat, force_b), cross(r, force_b)
