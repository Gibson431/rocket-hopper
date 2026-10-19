"""The vehicle as one floating Drake body with its frame at the gimbal pivot."""

from pydrake.all import AddMultibodyPlantSceneGraph, DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.drake.mass import spatial_inertia

GRAVITY = 9.80665  # Drake's default is 9.81


def build_vehicle_plant(
    cfg: VehicleConfig, builder: DiagramBuilder | None = None, finalize: bool = True
):
    """Return (plant, scene_graph). The plant is continuous (`time_step=0`) with the full-propellant
    mass properties; `MassUpdater` changes them during a run. With no `builder` the plant stands
    alone and `scene_graph` is None. S17 adds leg geometry with `finalize=False`, then finalises."""
    if builder is None:
        plant, scene_graph = MultibodyPlant(0.0), None
    else:
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 0.0)
    plant.AddRigidBody("hopper", spatial_inertia(cfg.tank.prop_mass_initial, cfg))
    plant.mutable_gravity_field().set_gravity_vector([0.0, 0.0, -GRAVITY])
    if finalize:
        plant.Finalize()
    return plant, scene_graph
