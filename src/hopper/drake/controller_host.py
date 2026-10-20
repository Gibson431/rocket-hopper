"""Hosts the unchanged flight controller on the Drake diagram.

One system owns the controller call, the delay queue and the applied command, all in a single
periodic update at the physics step: that makes delay 0 act on the step the command was computed
(docs/DRAKE_DESIGN.md Q9a). The applied command is discrete state; the engine reads it from there.
"""

from collections.abc import Sequence

import numpy as np
from pydrake.all import AbstractValue, BasicVector, LeafSystem

from hopper.config import VehicleConfig
from hopper.control.interface import Controller, StateFeedbackController
from hopper.drake.delay import CommandQueue
from hopper.dynamics.mass_model import mass_properties
from hopper.math3d import cross, quat_to_dcm
from hopper.sim.faults import Fault
from hopper.types import Command, SensorData, StateEstimate

_SENSOR_FIELDS = ("gyro", "accel", "baro_alt", "gps_pos", "gps_vel", "lidar_range")


def estimate_from_plant_state(s: np.ndarray, prop: float, cfg: VehicleConfig, t: float):
    """Perfect-state estimate: the plant state (body origin, world-frame rate) mapped to the
    CoM position/velocity and body-frame rate the controller expects."""
    mp = mass_properties(prop, cfg)
    q = s[:4] / np.linalg.norm(s[:4])
    rot = quat_to_dcm(q)
    r = rot @ np.array([0.0, 0.0, mp.com_z])
    return StateEstimate(t, s[4:7] + r, s[10:13] + cross(s[7:10], r), q, rot.T @ s[7:10], mp.mass)


class ControllerHost(LeafSystem):
    """Inputs `plant_state` (13) and `propellant` (1), plus `sensors` (a `SensorData`) unless
    `perfect_state`; output `command` (4: throttle, gx, gy, roll), the command in force during the
    next physics step.

    `faults` act on the command in force (`apply_command`), on every step from the held raw command,
    so a stuck gimbal ignores both later commands and the delay. Sensor mode keeps the newest
    sample per field since the last controller call."""

    def __init__(
        self,
        cfg: VehicleConfig,
        controller: Controller | StateFeedbackController,
        command_delay: float,
        perfect_state: bool = True,
        num_states: int = 13,
        faults: Sequence[Fault] = (),
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
        self._faults = tuple(faults)
        self._raw = np.zeros(4)  # last command out of the delay queue, before the faults
        self._latest: dict[str, object] = {}

        self._state_port = self.DeclareVectorInputPort("plant_state", num_states)
        self._prop_port = self.DeclareVectorInputPort("propellant", 1)
        self._sensors_port = None
        if not perfect_state:
            self._sensors_port = self.DeclareAbstractInputPort(
                "sensors", AbstractValue.Make(SensorData(0.0, None, None, None, None, None, None))
            )
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

    def get_sensors_input_port(self):
        return self._sensors_port

    def get_command_output_port(self):
        return self._cmd_port

    def reset(self) -> None:
        self._queue.clear()
        self._raw = np.zeros(4)
        self._latest = {}
        self._controller.reset()

    def _call_controller(self, context, k: int) -> np.ndarray:
        t = k * self._dt
        if self._perfect_state:
            est = estimate_from_plant_state(
                self._state_port.Eval(context), self._prop_port.Eval(context)[0], self._cfg, t
            )
            cmd = self._controller.step_state(est, t)
        else:
            data = SensorData(t, **{f: self._latest.get(f) for f in _SENSOR_FIELDS})
            self._latest = {}
            cmd = self._controller.step(data, t)
        return np.array([cmd.throttle, *cmd.gimbal, cmd.roll_torque])

    def _update(self, context, discrete_state):
        k = round(context.get_time() / self._dt)
        if self._sensors_port is not None:
            delivered = self._sensors_port.Eval(context)
            for name in _SENSOR_FIELDS:
                if (v := getattr(delivered, name)) is not None:
                    self._latest[name] = v
        if k % self._every == 0:
            self._queue.push(k, self._call_controller(context, k))
        if self._faults:
            self._raw = applied = self._queue.pop_ready(k, self._raw)
            cmd = Command(applied[0], applied[1:3], applied[3])
            for fault in self._faults:
                cmd = fault.apply_command(k * self._dt, cmd)
            applied = np.array([cmd.throttle, *cmd.gimbal, cmd.roll_torque])
        else:
            # engine off until a command lands
            held = context.get_discrete_state_vector().CopyToVector()
            applied = self._queue.pop_ready(k, held)
        discrete_state.get_mutable_vector().SetFromVector(applied)

    def _calc_command(self, context, out):
        out.SetFromVector(context.get_discrete_state_vector().CopyToVector())
