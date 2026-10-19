"""Variable-mass updater: writes the full spatial inertia from the propellant mass."""

import numpy as np
from pydrake.all import RotationalInertia, SpatialInertia
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.dynamics.mass_model import mass_properties


def spatial_inertia(prop_mass: float, cfg: VehicleConfig) -> SpatialInertia:
    """Spatial inertia about the body origin (gimbal pivot) from the one mass model."""
    mp = mass_properties(prop_mass, cfg)
    return SpatialInertia.MakeFromCentralInertia(
        mp.mass,
        np.array([0.0, 0.0, mp.com_z]),
        RotationalInertia(mp.inertia[0, 0], mp.inertia[1, 1], mp.inertia[2, 2]),
    )


class MassUpdater:
    """Not a LeafSystem: a system cannot write another system's parameters, so the sim loop
    calls `update` between `AdvanceTo` steps (see docs/DRAKE_DESIGN.md Q1)."""

    def __init__(self, plant: MultibodyPlant, cfg: VehicleConfig, body_name: str = "hopper"):
        self._plant = plant
        self._cfg = cfg
        self._body = plant.GetBodyByName(body_name)

    def update(self, plant_context, prop_mass: float, keep_com_state: bool = True) -> None:
        """Set the inertia for `prop_mass`. The legacy state is the CoM, continuous when the
        CoM moves inside the body, but Drake's state is the body origin; with `keep_com_state` the
        origin pose and velocity are rewritten so the CoM position and velocity do not jump."""
        plant, body = self._plant, self._body
        if keep_com_state:
            from hopper.drake.state import read_com_state, write_com_state

            com = read_com_state(plant, plant_context, body)
        body.SetSpatialInertiaInBodyFrame(plant_context, spatial_inertia(prop_mass, self._cfg))
        if keep_com_state:
            write_com_state(plant, plant_context, body, *com)
