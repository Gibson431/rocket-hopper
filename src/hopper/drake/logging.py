"""Run log on the physics grid: reads the diagram's outputs into the S06 `Logger`."""

import numpy as np

from hopper.analysis.logger import Logger
from hopper.config import VehicleConfig
from hopper.drake.diagram import DrakeRig
from hopper.drake.state import to_legacy_state
from hopper.dynamics.mass_model import mass_properties
from hopper.sim.engine import config_hash
from hopper.sim.events import Outcome
from hopper.types import GIM, GIMR, PROP, THR


class DrakeLogger:
    """One row per physics step (every `decimation`-th, plus the row that carries an outcome).
    Same columns and `df.attrs` as the legacy engine."""

    def __init__(self, rig: DrakeRig, cfg: VehicleConfig, decimation: int, seed: int | None):
        self._rig, self._cfg = rig, cfg
        self.extra_columns = rig.host.log_columns
        self._logger = Logger(
            decimation, {"seed": seed, "config_hash": config_hash(cfg)}, self.extra_columns
        )

    def wants_row(self) -> bool:
        return self._logger.wants_row()

    def record(self, root_context, t: float, event: Outcome | None) -> None:
        rig = self._rig
        keep = event is not None or self._logger.wants_row()
        if not keep:
            self._logger.skip()
            return
        x = to_legacy_state(rig.plant, rig.plant.GetMyContextFromRoot(root_context))
        engine_state = (
            rig.engine.GetMyContextFromRoot(root_context)
            .get_continuous_state_vector()
            .CopyToVector()
        )
        x[PROP], x[THR] = engine_state[0], engine_state[1]
        x[GIM], x[GIMR] = engine_state[2:4], engine_state[4:6]
        cmd = rig.host.get_command_output_port().Eval(rig.host.GetMyContextFromRoot(root_context))
        octx = rig.observer.GetMyContextFromRoot(root_context)
        forces = rig.observer.get_foot_forces_output_port().Eval(octx)
        thrust = rig.engine.get_thrust_output_port().Eval(
            rig.engine.GetMyContextFromRoot(root_context)
        )
        extra = rig.host.log_values() if self.extra_columns else ()
        self._logger.log(
            t,
            x,
            mass_properties(x[PROP], self._cfg).mass,
            float(thrust[0]),
            (cmd[0], (cmd[1], cmd[2]), cmd[3]),
            float(np.sum(forces)),
            int(np.count_nonzero(forces > 0.0)),
            event,
            extra,
        )

    def to_dataframe(self):
        return self._logger.to_dataframe()
