"""Attitude PD controller: body-frame quaternion error -> torque -> TVC gimbal + RCS roll.

Design (linearised about hover). Pitch/yaw torque comes from the gimbal: tau = z_com T g, so
I theta_ddot = z_com T g, a double integrator with plant gain K = z_com T / I. The control law is
computed as an angular acceleration, tau = I (kp e + kd (omega_des - omega)), so the loop is
theta_ddot = kp e + kd (omega_des - omega) regardless of K: kp = wn^2, kd = 2 zeta wn.
Defaults: wn = 3 rad/s for pitch/yaw (an order of magnitude below the 30 rad/s gimbal bandwidth, so
the actuator lag and ~20 ms of delay cost little phase), wn = 1.5 rad/s for roll (the RCS can only
make 5 N m, about 2 rad/s^2), zeta = 0.8. `scripts/tune_attitude.py` prints gains for any wn, zeta.

Saturation: when the demanded gimbal exceeds the limit both axes are scaled by the same factor, so
the torque direction is kept and the loop backs off its effective gain instead of limit-cycling on a
clipped axis. The command is also slew-limited to the actuator rate limit so it never asks for an
angle the gimbal cannot reach within a step. There is no integral term, hence no windup.
"""

from dataclasses import dataclass, field

import numpy as np

from hopper.control.params import ControlParams
from hopper.math3d import quat_error


@dataclass
class AttitudeGains:
    kp: np.ndarray  # (3,) 1/s^2: angular acceleration per rad of error
    kd: np.ndarray  # (3,) 1/s: angular acceleration per rad/s of rate error

    @classmethod
    def from_bandwidth(cls, wn, zeta) -> "AttitudeGains":
        wn = np.broadcast_to(np.asarray(wn, dtype=np.float64), (3,)).copy()
        zeta = np.broadcast_to(np.asarray(zeta, dtype=np.float64), (3,)).copy()
        return cls(kp=wn**2, kd=2.0 * zeta * wn)


@dataclass(frozen=True)
class AttitudeOutput:
    gimbal: np.ndarray  # (2,) [gx, gy] rad
    roll_torque: float  # -1..1
    torque_des: np.ndarray  # (3,) N m demanded in the body frame (before actuator limits)
    saturated: bool = False
    error: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rotation vector, rad


class AttitudeController:
    def __init__(self, params: ControlParams, gains: AttitudeGains) -> None:
        self.params = params
        self.gains = gains
        self._last_gimbal: np.ndarray | None = None

    def reset(self) -> None:
        self._last_gimbal = None

    def thrust_floor(self) -> float:
        """Below this thrust the lever arm is useless; hold the gimbal centred."""
        return 0.5 * self.params.thrust_min_frac * self.params.thrust_max

    def torque_to_gimbal(self, torque_xy: np.ndarray, thrust: float, com_z: float) -> np.ndarray:
        """Exact inverse of tau = [z F_y, -z F_x] with F = T [cos gx sin gy, -sin gx, ...]."""
        if thrust < self.thrust_floor():
            return np.zeros(2)
        f_x = -torque_xy[1] / com_z
        f_y = torque_xy[0] / com_z
        sin_gx = np.clip(-f_y / thrust, -1.0, 1.0)
        gx = np.arcsin(sin_gx)
        gy = np.arcsin(np.clip(f_x / (thrust * np.cos(gx)), -1.0, 1.0))
        g = np.array([gx, gy])
        peak = float(np.max(np.abs(g)))
        if peak > self.params.gimbal_max:
            g *= self.params.gimbal_max / peak
        return g

    def update(
        self,
        q_des: np.ndarray,
        omega_des: np.ndarray,
        quat: np.ndarray,
        omega: np.ndarray,
        thrust: float,
        mass: float,
        com_z: float,
        dt: float,
    ) -> AttitudeOutput:
        p = self.params
        _, inertia = p.mass_properties(mass)
        e = quat_error(q_des, quat)
        tau = inertia * (self.gains.kp * e + self.gains.kd * (omega_des - omega))

        raw = self.torque_to_gimbal(tau[:2], thrust, com_z)
        # Saturated if the unscaled demand would exceed the limit (scaling keeps the direction).
        want = self._unscaled_peak(tau[:2], thrust, com_z)
        saturated = want > p.gimbal_max

        gimbal = raw
        if self._last_gimbal is not None and dt > 0.0:
            step = p.gimbal_rate_max * dt
            gimbal = self._last_gimbal + np.clip(raw - self._last_gimbal, -step, step)
        self._last_gimbal = gimbal.copy()

        roll = float(np.clip(tau[2] / p.max_roll_torque, -1.0, 1.0))
        saturated = saturated or abs(tau[2]) > p.max_roll_torque
        return AttitudeOutput(gimbal, roll, tau, saturated, e)

    def _unscaled_peak(self, torque_xy: np.ndarray, thrust: float, com_z: float) -> float:
        if thrust < self.thrust_floor():
            return 0.0
        return float(np.max(np.abs(np.arcsin(np.clip(torque_xy / (com_z * thrust), -1, 1)))))
