"""`DrakeSimulation`: one closed-loop run on the Drake diagram.

The loop steps the simulator one physics step at a time. At the top of step `k` the state is x_k
and the host's periodic update for t_k has not run yet, so the event check (and a terminal row)
sees x_k without a controller call; the next `AdvanceTo` runs the update, then integrates.
"""

from collections.abc import Sequence

import numpy as np
from pydrake.all import Meshcat

from hopper.config import VehicleConfig
from hopper.control.interface import Controller, StateFeedbackController
from hopper.drake.diagram import build_diagram
from hopper.drake.logging import DrakeLogger
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_state_vector
from hopper.dynamics.environment import G0, Environment
from hopper.sim.events import Outcome
from hopper.sim.faults import Fault
from hopper.sim.result import SimResult, seed_of
from hopper.types import GIM, GIMR, PROP, THR


class DrakeSimulation:
    def __init__(
        self,
        cfg: VehicleConfig,
        controller: Controller | StateFeedbackController,
        env: Environment,
        x0: np.ndarray,
        rng: np.random.Generator,
        perfect_state: bool = True,
        command_delay: float = 0.0,
        log_decimation: int = 1,
        faults: Sequence[Fault] = (),
        meshcat: Meshcat | None = None,
    ) -> None:
        if command_delay < 0:
            raise ValueError(f"command_delay must be >= 0, got {command_delay}")
        self.cfg = cfg
        self.env = env
        self.controller = controller
        self.x0 = np.array(x0, dtype=np.float64)
        self.log_decimation = log_decimation
        self.faults = tuple(faults)
        self.seed = seed_of(rng)
        self.context = None
        self.rig = build_diagram(
            cfg, controller, env, rng, command_delay, perfect_state, self.faults, meshcat=meshcat
        )

    def run(self, t_end: float | None = None) -> SimResult:
        cfg, dt, rig = self.cfg, self.cfg.sim.dt, self.rig
        n_max = int(np.ceil(t_end / dt - 1e-9)) if t_end is not None else None
        sim = make_simulator(rig.diagram, dt)
        ctx = sim.get_mutable_context()
        x0 = self.x0
        set_state_vector(rig.plant, rig.plant.GetMyMutableContextFromRoot(ctx), x0, cfg)
        rig.engine.set_actuator_state(
            rig.engine.GetMyMutableContextFromRoot(ctx), x0[PROP], x0[THR], x0[GIM], x0[GIMR]
        )
        rig.host.reset()
        if rig.sensors is not None:
            rig.sensors.reset()
        rig.monitor.reset()
        sim.Initialize()
        self.context = ctx  # final state of the last run, for inspection
        logger = DrakeLogger(rig, cfg, self.log_decimation, self.seed)
        engine_ctx = rig.engine.GetMyContextFromRoot(ctx)
        plant_ctx = rig.plant.GetMyMutableContextFromRoot(ctx)

        outcome = Outcome.RUNNING
        faults, epochs = self.faults, None
        engine_mutable = rig.engine.GetMyMutableContextFromRoot(ctx)
        k = 0
        while True:
            t = k * dt
            if faults:  # rebuild the effective config only when a parameter fault changes it
                key = tuple(f.param_epoch(t) for f in faults)
                if key != epochs:
                    epochs, phys = key, cfg
                    for f in faults:
                        phys = f.modify_params(t, phys)
                    rig.engine.set_config(engine_mutable, phys)
            event = rig.monitor.evaluate(ctx)
            done = event is not None or (n_max is not None and k >= n_max)
            if done:
                logger.record(ctx, t, event)
                if event is not None:
                    outcome = event
                break
            sim.AdvanceTo(t)  # runs the host's update for t_k, so row k logs the command of step k
            logger.record(ctx, t, None)
            # Inertia at the step midpoint (docs/DRAKE_DESIGN.md Q1).
            prop = engine_ctx.get_continuous_state_vector().GetAtIndex(0)
            thrust = rig.engine.get_thrust_output_port().Eval(engine_ctx)[0]
            rig.mass_updater.update(plant_ctx, prop - 0.5 * dt * thrust / (cfg.engine.isp * G0))
            k += 1
            sim.AdvanceTo(k * dt)
        return SimResult(outcome, t, logger.to_dataframe(), self.seed)
