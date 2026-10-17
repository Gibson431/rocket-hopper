"""Propellant-dependent mass, CoM height and inertia."""

from dataclasses import dataclass

import numpy as np

from hopper.config import VehicleConfig


@dataclass(frozen=True)
class MassProperties:
    mass: float
    com_z: float  # CoM height above the base along z_B (m)
    inertia: np.ndarray  # (3, 3) about the CoM, body frame


def mass_properties(prop_mass: float, cfg: VehicleConfig) -> MassProperties:
    """Dry point mass plus a propellant cylinder filled from the tank bottom upward."""
    m_d = cfg.mass.dry_mass
    m_p = max(prop_mass, 0.0)
    if m_p == 0.0:
        return MassProperties(m_d, cfg.mass.dry_com_z, np.diag(cfg.mass.dry_inertia))

    tank = cfg.tank
    r = tank.radius
    h = m_p / tank.prop_mass_initial * tank.height
    z_p = tank.bottom_z + h / 2
    z_d = cfg.mass.dry_com_z

    mass = m_d + m_p
    com_z = (m_d * z_d + m_p * z_p) / mass

    i_prop = np.diag([m_p * (3 * r**2 + h**2) / 12] * 2 + [m_p * r**2 / 2])
    inertia = np.diag(cfg.mass.dry_inertia) + i_prop
    # Parallel axis along z only shifts the transverse axes.
    for m_i, z_i in ((m_d, z_d), (m_p, z_p)):
        inertia[0, 0] += m_i * (z_i - com_z) ** 2
        inertia[1, 1] += m_i * (z_i - com_z) ** 2
    return MassProperties(mass, com_z, inertia)
