"""Outcome events on the Drake diagram: `check_events` fed from Drake outputs."""

import numpy as np
from pydrake.all import EventStatus

from hopper.config import VehicleConfig
from hopper.drake.legs import ContactObserver
from hopper.drake.state import to_state_vector
from hopper.sim.events import ContactOutput, EventState, Outcome, check_events


class DrakeEventMonitor:
    """Runs `check_events` on the plant state and the `ContactObserver` outputs.

    `evaluate(diagram_context)` is one call of the event monitor. `as_simulator_monitor` wraps it
    for `Simulator.set_monitor`; the first outcome is kept in `outcome` and ends the run."""

    def __init__(self, plant, observer: ContactObserver, cfg: VehicleConfig):
        self._plant = plant
        self._obs = observer
        self._cfg = cfg
        self._state = EventState()
        self._last_t: float | None = None
        self._last: Outcome | None = None
        self.outcome: Outcome | None = None

    def reset(self) -> None:
        self._state, self._last_t, self._last, self.outcome = EventState(), None, None, None

    def evaluate(self, root_context) -> Outcome | None:
        t = root_context.get_time()
        if t == self._last_t:  # the simulator may call twice per time; history advances once
            return self._last
        pctx = self._plant.GetMyContextFromRoot(root_context)
        octx = self._obs.GetMyContextFromRoot(root_context)
        forces = np.array(self._obs.get_foot_forces_output_port().Eval(octx))
        contact = ContactOutput(
            force_I=np.array(self._obs.get_total_force_output_port().Eval(octx)),
            torque_B=np.zeros(3),  # not needed by the events
            foot_forces=forces,
            any_contact=bool(np.any(forces > 0.0)),
            base_strike=bool(self._obs.get_base_strike_output_port().Eval(octx)[0] > 0.5),
        )
        x = to_state_vector(self._plant, pctx)
        self._last_t, self._last = t, check_events(t, x, contact, self._cfg, self._state)
        if self.outcome is None:
            self.outcome = self._last
        return self._last

    def as_simulator_monitor(self, diagram):
        def monitor(root_context):
            outcome = self.evaluate(root_context)
            if outcome is None:
                return EventStatus.DidNothing()
            return EventStatus.ReachedTermination(diagram, outcome.name)

        return monitor
