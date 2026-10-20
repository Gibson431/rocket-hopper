"""Hosts the unchanged flight controller on the Drake diagram.

One system owns the controller call, the delay queue and the applied command, all in a single
periodic update at the physics step: that makes delay 0 act on the step the command was computed
(docs/DRAKE_DESIGN.md Q9a). The applied command is discrete state; the engine reads it from there.
"""

import numpy as np
from pydrake.all import BasicVector, LeafSystem

from hopper.config import VehicleConfig
from hopper.control.interface import Controller, StateFeedbackController
from hopper.drake.delay import CommandQueue
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import cross, quat_to_dcm
from hopper.types import StateEstimate


def estimate_from_plant_state(s: np.ndarray, prop: float, cfg: VehicleConfig, t: float):
    """Perfect-state estimate: the plant state (body origin, world-frame rate) mapped to the
    CoM position/velocity and body-frame rate the controller expects."""
    mp = mass_properties(prop, cfg)
    q = s[:4] / np.linalg.norm(s[:4])
    rot = quat_to_dcm(q)
    r = rot @ np.array([0.0, 0.0, mp.com_z])
    return StateEstimate(t, s[4:7] + r, s[10:13] + cross(s[7:10], r), q, rot.T @ s[7:10], mp.mass)


class ControllerHost(LeafSystem):
    """Inputs `plant_state` (13) and `propellant` (1); output `command` (4: throttle, gx, gy, roll),
    the command in force during the next physics step."""

    def __init__(
        self,
        cfg: VehicleConfig,
        controller: Controller | StateFeedbackController,
        command_delay: float,
        perfect_state: bool = True,
        num_states: int = 13,
    ):
        super().__init__()
        if command_delay < 0:
            raise ValueError(f"command_delay must be >= 0, got {command_delay}")
        self._cfg = cfg
        self._controller = controller
        self._dt = cfg.sim.dt
        self._every = max(1, round(1.0 / (cfg.sim.controller_rate * self._dt)))
        self._queue = CommandQueue(round(command_delay / self._dt))
        self._perfect_state = perfect_state

        self._state_port = self.DeclareVectorInputPort("plant_state", num_states)
        self._prop_port = self.DeclareVectorInputPort("propellant", 1)
        self.DeclareDiscreteState(BasicVector(np.zeros(4)))
        self.DeclarePeriodicDiscreteUpdateEvent(self._dt, 0.0, self._update)
        # Discrete state only, so the loop through the plant is not algebraic.
        self._cmd_port = self.DeclareVectorOutputPort(
            "command", 4, self._calc_command, prerequisites_of_calc={self.xd_ticket()}
        )

    @property
    def log_columns(self) -> tuple[str, ...]:
        return tuple(getattr(self._controller, "log_columns", ()))

    def log_values(self):
        return self._controller.log_values()

    def get_plant_state_input_port(self):
        return self._state_port

    def get_propellant_input_port(self):
        return self._prop_port

    def get_command_output_port(self):
        return self._cmd_port

    def reset(self) -> None:
        self._queue.clear()
        self._controller.reset()

    def _update(self, context, discrete_state):
        k = round(context.get_time() / self._dt)
        if k % self._every == 0:
            t = k * self._dt
            est = estimate_from_plant_state(
                self._state_port.Eval(context), self._prop_port.Eval(context)[0], self._cfg, t
            )
            cmd = self._controller.step_state(est, t)
            self._queue.push(k, np.array([cmd.throttle, *cmd.gimbal, cmd.roll_torque]))
        held = (
            context.get_discrete_state_vector().CopyToVector()
        )  # engine off until a command lands
        discrete_state.get_mutable_vector().SetFromVector(self._queue.pop_ready(k, held))

    def _calc_command(self, context, out):
        out.SetFromVector(context.get_discrete_state_vector().CopyToVector())
