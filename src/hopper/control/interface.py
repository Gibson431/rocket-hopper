"""Controller protocols (re-exported) and a scripted open-loop controller."""

from collections.abc import Sequence

import numpy as np

from hopper.types import Command, Controller, SensorData, StateEstimate, StateFeedbackController

__all__ = ["Command", "Controller", "OpenLoopController", "StateFeedbackController"]


class OpenLoopController:
    """Constant command, or a time-scheduled list of `(t_start, Command)` entries.

    Before the first scheduled entry it returns `default` (engine off unless given).
    """

    def __init__(
        self,
        command: Command | None = None,
        schedule: Sequence[tuple[float, Command]] | None = None,
        default: Command | None = None,
    ) -> None:
        off = Command(0.0, np.zeros(2), 0.0)
        self._default = default or command or off
        self._schedule = sorted(schedule or [], key=lambda e: e[0])
        self._constant = command if not self._schedule else None

    def reset(self) -> None:
        pass

    def _at(self, t: float) -> Command:
        if self._constant is not None:
            return self._constant
        current = self._default
        for t_start, cmd in self._schedule:
            if t_start > t:
                break
            current = cmd
        return current

    def step(self, sensors: SensorData, t: float) -> Command:
        return self._at(t)

    def step_state(self, est: StateEstimate, t: float) -> Command:
        return self._at(t)
