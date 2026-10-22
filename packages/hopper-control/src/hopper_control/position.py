"""Outer loops: altitude (throttle) and horizontal position (thrust tilt), on top of attitude.

Bandwidth separation (fast to slow): gimbal 30 rad/s > attitude wn 3 rad/s > altitude velocity loop
kv = 3 /s > horizontal wn = sqrt(kp) = 0.7 rad/s. The horizontal loop is ~4x slower than attitude,
so the attitude loop looks like a unity gain to it; the altitude loop only has the 0.1 s throttle
lag to worry about, which kv = 3 /s tolerates (phase lag ~ 17 degrees).

Altitude is the height of the foot tips above the ground ("AGL"), from the estimated CoM height and
mass. Control is a P-PI cascade: the altitude error sets a (limited) velocity command added to the
reference velocity, and the velocity error drives a PI law on acceleration. Throttle is
`m (g + a_cmd) / (T_max cos tilt)`, so gravity feedforward uses the estimated mass. The integrator
only absorbs unmodelled accelerations (drag, mass error) and is frozen when the throttle is pinned.
"""

from dataclasses import dataclass, field

import numpy as np
from hopper_core.math3d import cross, dcm_to_quat, quat_rotate
from hopper_core.types import Command, StateEstimate

from hopper_control.attitude import AttitudeController, AttitudeGains
from hopper_control.params import ControlParams

G = 9.80665
UP = np.array([0.0, 0.0, 1.0])


@dataclass
class AltitudeGains:
    kz: float = 0.8  # 1/s: altitude error -> velocity command
    kv: float = 3.0  # 1/s: velocity error -> acceleration
    ki: float = 0.5  # 1/s^2: integral of velocity error -> acceleration
    v_up: float = 3.0  # m/s: cap on the velocity command from altitude error
    v_down: float = 3.0
    i_max: float = 4.0  # m: integrator clamp (ki * i_max = 2 m/s^2)


@dataclass(frozen=True)
class AltitudeOutput:
    throttle: float
    a_cmd: float
    saturated: bool


class AltitudeController:
    def __init__(self, params: ControlParams, gains: AltitudeGains | None = None) -> None:
        self.params = params
        self.gains = gains or AltitudeGains()
        self.integral = 0.0

    def reset(self) -> None:
        self.integral = 0.0

    def update(
        self,
        alt: float,
        vz: float,
        alt_ref: float | None,
        vz_ref: float,
        mass: float,
        tilt: float,
        dt: float,
        cut: bool = False,
    ) -> AltitudeOutput:
        """`alt_ref=None` regulates velocity only (descent). `cut` shuts the engine."""
        g, p = self.gains, self.params
        if cut:
            self.integral = 0.0
            return AltitudeOutput(0.0, 0.0, False)
        vz_cmd = vz_ref
        if alt_ref is not None:
            vz_cmd += float(np.clip(g.kz * (alt_ref - alt), -g.v_down, g.v_up))
        ev = vz_cmd - vz
        a_cmd = g.kv * ev + g.ki * self.integral
        raw = mass * (G + a_cmd) / (p.thrust_max * max(np.cos(tilt), 0.5))
        thr = float(np.clip(raw, p.thrust_min_frac, 1.0))
        pinned = (raw > 1.0 and ev > 0.0) or (raw < p.thrust_min_frac and ev < 0.0)
        if not pinned:
            self.integral = float(np.clip(self.integral + ev * dt, -g.i_max, g.i_max))
        return AltitudeOutput(thr, a_cmd, raw != thr)


@dataclass
class HorizontalGains:
    kp: float = 0.5  # 1/s^2: wn = 0.7 rad/s
    kd: float = 1.25  # 1/s: zeta ~ 0.9
    max_tilt: float = np.radians(15.0)
    max_speed: float = 3.0  # m/s: the position term never asks for more than this closing speed
    tilt_rate_max: float = np.radians(30.0)  # rad/s: setpoint slew, well inside the gimbal rate


@dataclass(frozen=True)
class HorizontalOutput:
    q_des: np.ndarray
    a_des: np.ndarray  # (2,) m/s^2 after the tilt clamp
    tilt_des: float


def _attitude_from_axis(n: np.ndarray, yaw: float) -> np.ndarray:
    """Quaternion with z_B = n (unit) and the heading closest to `yaw`."""
    x_c = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    y = cross(n, x_c)
    y /= np.linalg.norm(y)
    x = cross(y, n)
    return dcm_to_quat(np.column_stack([x, y, n]))


class HorizontalController:
    def __init__(self, gains: HorizontalGains | None = None) -> None:
        self.gains = gains or HorizontalGains()
        self._n: np.ndarray | None = None  # last commanded thrust axis

    def reset(self) -> None:
        self._n = None

    def update(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        pos_des: np.ndarray,
        vel_des: np.ndarray,
        yaw_des: float,
        dt: float,
    ) -> HorizontalOutput:
        g = self.gains
        err = np.asarray(pos_des, dtype=np.float64) - pos
        reach = g.max_speed * g.kd / g.kp
        dist = float(np.linalg.norm(err))
        if dist > reach:
            err = err * (reach / dist)
        a = g.kp * err + g.kd * (np.asarray(vel_des, dtype=np.float64) - vel)
        a_max = G * np.tan(g.max_tilt)
        mag = float(np.linalg.norm(a))
        if mag > a_max:
            a = a * (a_max / mag)

        target = np.array([a[0], a[1], G])
        target /= np.linalg.norm(target)
        n = self._rate_limited(target, g.tilt_rate_max * dt)
        self._n = n
        return HorizontalOutput(_attitude_from_axis(n, yaw_des), a, float(np.arccos(n[2])))

    def _rate_limited(self, target: np.ndarray, max_step: float) -> np.ndarray:
        prev = UP if self._n is None else self._n
        return self._slerp_toward(prev, target, max_step)

    @staticmethod
    def _slerp_toward(a: np.ndarray, b: np.ndarray, max_step: float) -> np.ndarray:
        ang = float(np.arccos(np.clip(a @ b, -1.0, 1.0)))
        if ang <= max_step or ang < 1e-12:
            return b
        f = max_step / ang
        n = (np.sin((1 - f) * ang) * a + np.sin(f * ang) * b) / np.sin(ang)
        return n / np.linalg.norm(n)


@dataclass(frozen=True)
class Setpoint:
    alt_ref: float | None = None  # foot height AGL (m); None regulates vz only
    vz_ref: float = 0.0
    xy_ref: np.ndarray = field(default_factory=lambda: np.zeros(2))
    vxy_ref: np.ndarray = field(default_factory=lambda: np.zeros(2))
    yaw: float = 0.0
    cut: bool = False  # engine off
    level: bool = False  # ignore xy_ref: hold the thrust axis vertical


class PositionController:
    """Altitude + horizontal + attitude loops driven by a `Setpoint`."""

    def __init__(
        self,
        params: ControlParams,
        attitude_gains: AttitudeGains | None = None,
        altitude_gains: AltitudeGains | None = None,
        horizontal_gains: HorizontalGains | None = None,
    ) -> None:
        self.params = params
        self.altitude = AltitudeController(params, altitude_gains)
        self.horizontal = HorizontalController(horizontal_gains)
        self.attitude = AttitudeController(
            params, attitude_gains or AttitudeGains.from_bandwidth([3.0, 3.0, 1.5], 0.8)
        )

    def reset(self) -> None:
        self.altitude.reset()
        self.horizontal.reset()
        self.attitude.reset()

    def altitude_agl(self, est: StateEstimate) -> float:
        return self.params.foot_height(est.mass, float(est.pos[2]))

    @staticmethod
    def tilt(est: StateEstimate) -> float:
        return float(np.arccos(np.clip(quat_rotate(est.quat, UP)[2], -1.0, 1.0)))

    def command(self, est: StateEstimate, sp: Setpoint, dt: float) -> Command:
        p = self.params
        alt_out = self.altitude.update(
            self.altitude_agl(est),
            float(est.vel[2]),
            sp.alt_ref,
            sp.vz_ref,
            est.mass,
            self.tilt(est),
            dt,
            cut=sp.cut,
        )
        pos_xy, vel_xy = est.pos[:2], est.vel[:2]
        xy_ref, vxy_ref = (pos_xy, vel_xy) if sp.level else (sp.xy_ref, sp.vxy_ref)
        hor = self.horizontal.update(pos_xy, vel_xy, xy_ref, vxy_ref, sp.yaw, dt)
        com_z, _ = p.mass_properties(est.mass)
        # The commanded (not lagged) thrust is what the controller knows.
        att = self.attitude.update(
            hor.q_des,
            np.zeros(3),
            est.quat,
            est.omega,
            alt_out.throttle * p.thrust_max,
            est.mass,
            com_z,
            dt,
        )
        return Command(alt_out.throttle, att.gimbal, att.roll_torque)
