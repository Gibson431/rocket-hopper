"""Vehicle parameters the controller is allowed to know (no simulator imports)."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ControlParams:
    thrust_max: float
    thrust_min_frac: float
    gimbal_max: float
    gimbal_rate_max: float
    max_roll_torque: float
    inertia_diag_dry: tuple[float, float, float]
    dry_mass: float
    dry_com_z: float
    prop_mass_initial: float
    tank_bottom_z: float
    tank_height: float
    tank_radius: float
    leg_length: float  # foot tips hang this far below the base

    def foot_height(self, mass: float, z_com: float) -> float:
        """Height of the foot tips above the ground for a CoM at `z_com` (upright vehicle)."""
        return z_com - self.mass_properties(mass)[0] - self.leg_length

    def mass_properties(self, mass: float) -> tuple[float, np.ndarray]:
        """CoM height above the base and diagonal inertia for an estimated total mass.

        Same uniform-cylinder model as the plant, driven by mass (what an estimator provides).
        """
        m_p = min(max(mass - self.dry_mass, 0.0), self.prop_mass_initial)
        i_dry = np.array(self.inertia_diag_dry)
        if m_p == 0.0:
            return self.dry_com_z, i_dry
        r = self.tank_radius
        h = m_p / self.prop_mass_initial * self.tank_height
        z_p = self.tank_bottom_z + h / 2
        m_tot = self.dry_mass + m_p
        com_z = (self.dry_mass * self.dry_com_z + m_p * z_p) / m_tot
        inertia = i_dry + np.array([m_p * (3 * r**2 + h**2) / 12] * 2 + [m_p * r**2 / 2])
        for m_i, z_i in ((self.dry_mass, self.dry_com_z), (m_p, z_p)):
            inertia[:2] += m_i * (z_i - com_z) ** 2
        return com_z, inertia
