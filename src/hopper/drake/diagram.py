"""Wires the vehicle diagram. Python wiring only; the physics lives in the systems.

ControllerHost -command-> Engine -spatial_forces-+
                                                 +-> ForceCombiner -> plant (applied forces)
WindSource -wind-> AeroLoads -spatial_forces-----+
plant state -> Engine, AeroLoads, ControllerHost, ContactObserver
Engine propellant -> AeroLoads, ControllerHost
plant contact results -> ContactObserver
"""

from dataclasses import dataclass

import numpy as np
from pydrake.all import Diagram, DiagramBuilder
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig
from hopper.control.interface import Controller, StateFeedbackController
from hopper.drake.aero import AeroLoads
from hopper.drake.combiner import ForceCombiner
from hopper.drake.controller_host import ControllerHost
from hopper.drake.events import DrakeEventMonitor
from hopper.drake.legs import ContactObserver, add_legs
from hopper.drake.mass import MassUpdater
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.propulsion import Engine
from hopper.drake.wind import WindSource
from hopper.dynamics.environment import Environment


@dataclass
class DrakeRig:
    diagram: Diagram
    plant: MultibodyPlant
    engine: Engine
    host: ControllerHost
    wind: WindSource
    aero: AeroLoads
    observer: ContactObserver
    monitor: DrakeEventMonitor
    mass_updater: MassUpdater


def build_diagram(
    cfg: VehicleConfig,
    controller: Controller | StateFeedbackController,
    env: Environment,
    rng: np.random.Generator,
    command_delay: float = 0.0,
    perfect_state: bool = True,
) -> DrakeRig:
    """The wind stream is seeded from `rng` exactly like the legacy `Simulation` does."""
    wind_seed = int(rng.integers(2**32))
    builder = DiagramBuilder()
    plant, scene_graph = build_vehicle_plant(
        cfg, builder, finalize=False, gravity_enabled=env.gravity_enabled
    )
    legs = add_legs(plant, scene_graph, cfg)
    plant.Finalize()

    host = builder.AddSystem(ControllerHost(cfg, controller, command_delay, perfect_state))
    engine = builder.AddSystem(Engine(cfg, plant))
    wind = builder.AddSystem(
        WindSource(env.wind.mean_i, env.wind.gust_sigma, env.wind.gust_tau, wind_seed, cfg.sim.dt)
    )
    aero = builder.AddSystem(AeroLoads(cfg, plant, env.aero_enabled))
    combiner = builder.AddSystem(ForceCombiner(2))
    observer = builder.AddSystem(ContactObserver(plant, legs))

    state = plant.get_state_output_port()
    builder.Connect(host.get_command_output_port(), engine.get_command_input_port())
    builder.Connect(state, engine.get_plant_state_input_port())
    builder.Connect(state, host.get_plant_state_input_port())
    builder.Connect(engine.get_propellant_output_port(), host.get_propellant_input_port())
    builder.Connect(state, aero.get_plant_state_input_port())
    builder.Connect(wind.get_wind_output_port(), aero.get_wind_input_port())
    builder.Connect(engine.get_propellant_output_port(), aero.get_propellant_input_port())
    builder.Connect(engine.get_force_output_port(), combiner.get_input_port(0))
    builder.Connect(aero.get_force_output_port(), combiner.get_input_port(1))
    builder.Connect(combiner.get_output_port(), plant.get_applied_spatial_force_input_port())
    builder.Connect(state, observer.get_plant_state_input_port())
    builder.Connect(
        plant.get_contact_results_output_port(), observer.get_contact_results_input_port()
    )
    return DrakeRig(
        diagram=builder.Build(),
        plant=plant,
        engine=engine,
        host=host,
        wind=wind,
        aero=aero,
        observer=observer,
        monitor=DrakeEventMonitor(plant, observer, cfg),
        mass_updater=MassUpdater(plant, cfg),
    )
