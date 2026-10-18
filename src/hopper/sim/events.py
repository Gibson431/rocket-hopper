"""Outcome detection: landing, crash, tip-over, bounds and timeout."""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from hopper.config import VehicleConfig
from hopper.dynamics.contact import ContactOutput
from hopper.math3d import quat_to_dcm
from hopper.types import OMEGA, POS, QUAT, VEL


class Outcome(Enum):
    RUNNING = auto()
    LANDED = auto()
    CRASHED = auto()
    TIPPED_OVER = auto()
    OUT_OF_BOUNDS = auto()
    TIMEOUT = auto()
    FLAMEOUT_FALLING = auto()  # reserved; not detected yet


@dataclass
class EventState:
    """What `check_events` remembers between calls."""

    in_contact: bool = False
    prev_vz: float = 0.0  # vertical speed on the previous call (pre-touchdown approach speed)
    calm_since: float | None = None  # start of the current run of landing conditions
    # True once the feet have left the ground: resting on the pad is not a landing.
    has_flown: bool = False


def tilt_angle(x: np.ndarray) -> float:
    """Angle between z_B and the inertial up axis (rad)."""
    return float(np.arccos(np.clip(quat_to_dcm(x[QUAT])[2, 2], -1.0, 1.0)))


def check_events(
    t: float,
    x: np.ndarray,
    contact: ContactOutput,
    cfg: VehicleConfig,
    prev_state: EventState,
) -> Outcome | None:
    """Outcome at this step, or None to keep running. Updates `prev_state` in place."""
    sim = cfg.sim
    tilt = tilt_angle(x)
    touchdown = contact.any_contact and not prev_state.in_contact
    approach_vz = prev_state.prev_vz
    prev_state.in_contact = contact.any_contact
    prev_state.has_flown |= not contact.any_contact
    prev_state.prev_vz = float(x[VEL][2])

    if contact.base_strike:
        return Outcome.CRASHED
    if touchdown and (-approach_vz > sim.v_crash or tilt > sim.crash_tilt):
        return Outcome.CRASHED
    if contact.any_contact and tilt > sim.tipover_tilt:
        return Outcome.TIPPED_OVER
    if np.hypot(*x[POS][:2]) > sim.max_range or x[POS][2] > sim.max_alt:
        return Outcome.OUT_OF_BOUNDS

    calm = (
        prev_state.has_flown
        and bool(np.all(contact.foot_forces > 0.0))
        and np.linalg.norm(x[VEL]) < sim.land_speed
        and tilt < sim.land_tilt
        and np.linalg.norm(x[OMEGA]) < sim.land_rate
    )
    if not calm:
        prev_state.calm_since = None
    else:
        if prev_state.calm_since is None:
            prev_state.calm_since = t
        if t - prev_state.calm_since >= sim.land_hold:
            return Outcome.LANDED

    if t >= sim.max_time:
        return Outcome.TIMEOUT
    return None


class EventMonitor:
    """Stateful wrapper that owns the hold timer and touchdown history."""

    def __init__(self, cfg: VehicleConfig) -> None:
        self.cfg = cfg
        self.state = EventState()

    def update(self, t: float, x: np.ndarray, contact: ContactOutput) -> Outcome | None:
        return check_events(t, x, contact, self.cfg, self.state)
