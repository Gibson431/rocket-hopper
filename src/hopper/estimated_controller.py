"""Flight-software wiring: sensors -> EKF -> mission controller. Lives outside both packages so the
control and estimation packages stay independent of each other."""

import numpy as np

from hopper.estimation.ekf import EKF, EkfParams
from hopper.types import Command, SensorData, StateEstimate

_OFF = Command(0.0, np.zeros(2), 0.0)
_GROUNDED = ("IDLE", "LANDED")

LOG_COLUMNS = (
    *(f"est_pos_{a}" for a in "xyz"),
    *(f"est_vel_{a}" for a in "xyz"),
    *(f"est_quat_{a}" for a in "wxyz"),
    "est_mass",
    *(f"est_bg_{a}" for a in "xyz"),
    *(f"est_ba_{a}" for a in "xyz"),
    "est_bb",
    *(f"sig_{n}" for n in (
        "pos_x", "pos_y", "pos_z", "vel_x", "vel_y", "vel_z", "att_x", "att_y", "att_z",
        "bg_x", "bg_y", "bg_z", "ba_x", "ba_y", "ba_z", "bb",
    )),
)  # fmt: skip


class EstimatedController:
    """`Controller` that flies a `MissionController` on the EKF estimate.

    The engine stays off until the static alignment finishes. Exposes `log_columns` /
    `log_values()` so the simulation can log the estimate and its 1-sigma next to the truth.
    """

    log_columns = LOG_COLUMNS

    def __init__(self, params: EkfParams, mission) -> None:
        self.ekf = EKF(params)
        self.mission = mission

    def reset(self) -> None:
        self.ekf.reset()
        self.mission.reset()

    def step_state(self, est: StateEstimate, t: float) -> Command:
        raise NotImplementedError("the estimated controller builds its own estimate; use step")

    def step(self, sensors: SensorData, t: float) -> Command:
        grounded = getattr(getattr(self.mission, "state", None), "name", "") in _GROUNDED
        est = self.ekf.process(sensors, grounded=grounded and self.ekf.initialised)
        if est is None:
            return _OFF
        cmd = self.mission.step_state(est, t)
        self.ekf.note_command(cmd.throttle)
        return cmd

    def log_values(self) -> np.ndarray:
        ekf = self.ekf
        if not ekf.initialised:
            return np.full(len(LOG_COLUMNS), np.nan)
        est = ekf.estimate()
        return np.concatenate(
            [est.pos, est.vel, est.quat, [est.mass], ekf.bg, ekf.ba, [ekf.bb], ekf.sigma]
        )
