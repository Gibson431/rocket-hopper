"""Hop-hover-land mission: an explicit, deterministic state machine plus the controller wiring.

    IDLE -> IGNITION -> ASCEND -> HOVER -> TRANSLATE -> DESCEND -> TOUCHDOWN -> CUTOFF -> LANDED
                          (any flying state) -> ABORT -> CUTOFF

The machine only emits references (altitude/velocity/xy and flags); `PositionController` turns them
into commands. Altitude is the foot-tip height above ground. Vertical references are a velocity
profile with a limited slope (`ref_accel`) so the loops never see a step. Guards that one noisy
sample could fool (arrival, tilt, altitude error) must hold for a time before they fire.
"""

from dataclasses import dataclass
from enum import Enum, auto

import numpy as np

from hopper.control.attitude import AttitudeGains
from hopper.control.params import ControlParams
from hopper.control.position import (
    AltitudeGains,
    HorizontalGains,
    PositionController,
    Setpoint,
)
from hopper.types import Command, SensorData, StateEstimate


class MissionState(Enum):
    IDLE = auto()
    IGNITION = auto()
    ASCEND = auto()
    HOVER = auto()
    TRANSLATE = auto()
    DESCEND = auto()
    TOUCHDOWN = auto()
    CUTOFF = auto()
    LANDED = auto()
    ABORT = auto()


@dataclass(frozen=True)
class MissionParams:
    target_alt: float = 10.0  # m, foot tips above ground
    hover_time: float = 3.0  # s
    target_xy: tuple[float, float] = (3.0, 0.0)  # landing site, offset from the take-off point (m)
    ascent_rate: float = 2.0  # m/s
    descent_rate: float = 2.0  # m/s, above touchdown_alt
    final_descent_rate: float = 0.5  # m/s, below touchdown_alt until contact
    touchdown_alt: float = 3.0  # m, switch to the final descent rate
    cutoff_alt: float = 0.05  # m, engine off at or below this (contact)
    yaw: float = 0.0  # rad
    start_time: float = 0.5  # s, IDLE until then
    ignition_time: float = 0.5  # s of engine running before the climb is commanded
    ref_accel: float = 1.0  # m/s^2, slope limit of the vertical velocity reference
    settle_time: float = 0.5  # s a guard must hold continuously
    pos_tol: float = 0.3  # m, arrival tolerance (altitude and xy)
    speed_tol: float = 0.3  # m/s, arrival speed tolerance
    state_timeout: float = 40.0  # s, longest stay in ASCEND/TRANSLATE/DESCEND/TOUCHDOWN
    abort_tilt: float = np.radians(30.0)  # rad
    abort_tilt_time: float = 0.1  # s the tilt must exceed the limit
    abort_alt_error: float = 6.0  # m from the altitude reference
    abort_cut_alt: float = 1.0  # m, aborting below this just cuts the engine


@dataclass(frozen=True)
class MissionInputs:
    t: float
    alt: float  # foot-tip height above ground (m)
    vz: float
    xy: np.ndarray
    vxy: np.ndarray
    tilt: float  # rad


@dataclass(frozen=True)
class MissionOutput:
    state: MissionState
    alt_ref: float | None  # None: regulate vz only
    vz_ref: float
    xy_ref: np.ndarray
    engine_cut: bool
    level: bool  # hold the thrust axis vertical, ignore xy_ref


_FLYING = (
    MissionState.ASCEND,
    MissionState.HOVER,
    MissionState.TRANSLATE,
    MissionState.DESCEND,
    MissionState.TOUCHDOWN,
)
_TIMED = (
    MissionState.ASCEND,
    MissionState.TRANSLATE,
    MissionState.DESCEND,
    MissionState.TOUCHDOWN,
)
_ALT_CHECKED = (MissionState.ASCEND, MissionState.HOVER, MissionState.TRANSLATE)


class MissionStateMachine:
    def __init__(self, params: MissionParams) -> None:
        self.p = params
        self.reset()

    def reset(self) -> None:
        self.state = MissionState.IDLE
        self.history: list[tuple[float, MissionState]] = []
        self._t_enter = 0.0
        self._t_last: float | None = None
        self._home = np.zeros(2)
        self._alt_ref = 0.0
        self._vz_ref = 0.0
        self._holds: dict[str, float] = {}

    # --- helpers ------------------------------------------------------------------------

    def _held(self, key: str, cond: bool, t: float, duration: float) -> bool:
        """True once `cond` has been continuously true for `duration`."""
        if not cond:
            self._holds.pop(key, None)
            return False
        start = self._holds.setdefault(key, t)
        return t - start >= duration

    def _enter(self, state: MissionState, inp: MissionInputs) -> None:
        self.state = state
        self._t_enter = inp.t
        self._holds.clear()
        self.history.append((inp.t, state))
        if state == MissionState.ASCEND:
            self._alt_ref = inp.alt
            self._vz_ref = 0.0

    def _slew(self, target: float, dt: float) -> None:
        step = self.p.ref_accel * dt
        self._vz_ref += float(np.clip(target - self._vz_ref, -step, step))

    def _abort(self, inp: MissionInputs) -> None:
        self._enter(
            MissionState.CUTOFF if inp.alt < self.p.abort_cut_alt else MissionState.ABORT, inp
        )

    def _check_abort(self, inp: MissionInputs) -> bool:
        p, s = self.p, self.state
        if s not in _FLYING:
            return False
        tilt = self._held("tilt", inp.tilt > p.abort_tilt, inp.t, p.abort_tilt_time)
        alt_err = s in _ALT_CHECKED and self._held(
            "alt", abs(inp.alt - self._alt_ref) > p.abort_alt_error, inp.t, p.settle_time
        )
        timeout = s in _TIMED and inp.t - self._t_enter > p.state_timeout
        return tilt or alt_err or timeout

    # --- update -------------------------------------------------------------------------

    def update(self, inp: MissionInputs) -> MissionOutput:
        p = self.p
        dt = 0.0 if self._t_last is None else inp.t - self._t_last
        if self._t_last is None:
            self._home = np.array(inp.xy, dtype=np.float64)
            self.history.append((inp.t, self.state))
        self._t_last = inp.t
        site = self._home + np.array(p.target_xy)

        if self._check_abort(inp):
            self._abort(inp)

        s = self.state
        if s == MissionState.IDLE and inp.t >= p.start_time:
            self._enter(MissionState.IGNITION, inp)
        elif s == MissionState.IGNITION and inp.t - self._t_enter >= p.ignition_time:
            self._enter(MissionState.ASCEND, inp)
        elif s == MissionState.ASCEND:
            if self._held(
                "arrive",
                abs(inp.alt - p.target_alt) < p.pos_tol and abs(inp.vz) < p.speed_tol,
                inp.t,
                p.settle_time,
            ):
                self._enter(MissionState.HOVER, inp)
        elif s == MissionState.HOVER and inp.t - self._t_enter >= p.hover_time:
            far = np.linalg.norm(np.array(p.target_xy)) > p.pos_tol
            self._enter(MissionState.TRANSLATE if far else MissionState.DESCEND, inp)
        elif s == MissionState.TRANSLATE:
            at_site = np.linalg.norm(inp.xy - site) < p.pos_tol
            slow = np.linalg.norm(inp.vxy) < p.speed_tol
            if self._held("arrive", bool(at_site and slow), inp.t, p.settle_time):
                self._enter(MissionState.DESCEND, inp)
        elif s == MissionState.DESCEND and inp.alt <= p.touchdown_alt:
            self._enter(MissionState.TOUCHDOWN, inp)
        elif s in (MissionState.TOUCHDOWN, MissionState.ABORT) and inp.alt <= p.cutoff_alt:
            self._enter(MissionState.CUTOFF, inp)
        elif s == MissionState.CUTOFF and self._held(
            "rest", abs(inp.vz) < 0.1, inp.t, p.settle_time
        ):
            self._enter(MissionState.LANDED, inp)

        return self._output(inp, dt, site)

    def _output(self, inp: MissionInputs, dt: float, site: np.ndarray) -> MissionOutput:
        p, s = self.p, self.state
        alt_ref: float | None = None
        level = False
        cut = s in (MissionState.IDLE, MissionState.CUTOFF, MissionState.LANDED)
        xy_ref = self._home

        if s == MissionState.IGNITION:
            alt_ref, self._vz_ref = inp.alt, 0.0
        elif s == MissionState.ASCEND:
            gap = p.target_alt - self._alt_ref
            # sqrt profile: stops at the target with deceleration 0.8 * ref_accel
            v_des = min(p.ascent_rate, float(np.sqrt(1.6 * p.ref_accel * max(gap, 0.0))))
            self._slew(v_des, dt)
            self._alt_ref = min(self._alt_ref + self._vz_ref * dt, p.target_alt)
            alt_ref = self._alt_ref
        elif s in (MissionState.HOVER, MissionState.TRANSLATE):
            self._alt_ref, self._vz_ref, alt_ref = p.target_alt, 0.0, p.target_alt
            xy_ref = site if s == MissionState.TRANSLATE else self._home
        elif s in (MissionState.DESCEND, MissionState.TOUCHDOWN, MissionState.ABORT):
            low = s != MissionState.DESCEND and inp.alt <= p.touchdown_alt
            rate = p.final_descent_rate if low else p.descent_rate
            self._slew(-rate, dt)
            xy_ref = site
            level = s == MissionState.ABORT
        elif cut:
            self._vz_ref = 0.0
            xy_ref = site if s != MissionState.IDLE else self._home
            level = s == MissionState.CUTOFF
        return MissionOutput(s, alt_ref, self._vz_ref, np.array(xy_ref), cut, level)


class MissionController:
    """State machine + position loops + attitude controller, perfect-state feedback."""

    def __init__(
        self,
        params: ControlParams,
        mission: MissionParams | None = None,
        attitude_gains: AttitudeGains | None = None,
        altitude_gains: AltitudeGains | None = None,
        horizontal_gains: HorizontalGains | None = None,
    ) -> None:
        self.mission = mission or MissionParams()
        self.machine = MissionStateMachine(self.mission)
        self.position = PositionController(params, attitude_gains, altitude_gains, horizontal_gains)
        self._last_t: float | None = None

    @property
    def state(self) -> MissionState:
        return self.machine.state

    def reset(self) -> None:
        self.machine.reset()
        self.position.reset()
        self._last_t = None

    def step(self, sensors: SensorData, t: float) -> Command:
        raise NotImplementedError("mission controller needs a state estimate; use step_state")

    def step_state(self, est: StateEstimate, t: float) -> Command:
        dt = 0.0 if self._last_t is None else t - self._last_t
        self._last_t = t
        pc = self.position
        out = self.machine.update(
            MissionInputs(
                t,
                pc.altitude_agl(est),
                float(est.vel[2]),
                est.pos[:2],
                est.vel[:2],
                pc.tilt(est),
            )
        )
        sp = Setpoint(
            out.alt_ref,
            out.vz_ref,
            out.xy_ref,
            np.zeros(2),
            self.mission.yaw,
            cut=out.engine_cut,
            level=out.level,
        )
        return pc.command(est, sp, dt)
