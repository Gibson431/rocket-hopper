"""Fixed-throttle attitude hold with perfect-state feedback; the base for the S09 cascade."""

import numpy as np

from hopper.control.attitude import AttitudeController, AttitudeGains
from hopper.control.params import ControlParams
from hopper.types import Command, SensorData, StateEstimate


class AttitudeHoldController:
    def __init__(
        self,
        params: ControlParams,
        gains: AttitudeGains,
        throttle: float,
        q_des: np.ndarray | None = None,
        omega_des: np.ndarray | None = None,
    ) -> None:
        self.params = params
        self.attitude = AttitudeController(params, gains)
        self.throttle = throttle
        self.q_des = np.array([1.0, 0.0, 0.0, 0.0]) if q_des is None else np.asarray(q_des)
        self.omega_des = np.zeros(3) if omega_des is None else np.asarray(omega_des)
        self._last_t: float | None = None

    def reset(self) -> None:
        self.attitude.reset()
        self._last_t = None

    def step(self, sensors: SensorData, t: float) -> Command:
        raise NotImplementedError("attitude hold needs a state estimate; use step_state")

    def step_state(self, est: StateEstimate, t: float) -> Command:
        p = self.params
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        # The commanded (not lagged) thrust is what the controller knows.
        thr = 0.0 if self.throttle <= 0 else min(max(self.throttle, p.thrust_min_frac), 1.0)
        com_z, _ = p.mass_properties(est.mass)
        out = self.attitude.update(
            self.q_des, self.omega_des, est.quat, est.omega, thr * p.thrust_max, est.mass, com_z, dt
        )
        return Command(self.throttle, out.gimbal, out.roll_torque)
