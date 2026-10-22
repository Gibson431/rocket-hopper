"""Drive the EKF with the real sensor models along an analytic trajectory (no dynamics).

The truth is a smooth manoeuvre that starts from a static hover, so the filter can align first.
`SensorSuite.sample` is called every `DT` seconds with the exact state and specific force, and the
newest sample per field since the last controller call is handed to the filter, as the engine does.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from hopper_core.math3d import quat_conj, quat_error, quat_mul, quat_to_dcm
from hopper_core.types import OMEGA, POS, QUAT, VEL, SensorData, make_state

from hopper.config import VehicleConfig
from hopper.scenario_glue import ekf_params_from_config
from hopper.sensors.suite import SensorSuite

G = 9.80665
DT = 0.0025  # sensor-sampling step: divides every sensor period (400, 50, 50, 10 Hz)
CTRL_EVERY = 4  # 100 Hz controller
T_STATIC = 2.0  # s of hover before the manoeuvre
FIELDS = ("gyro", "accel", "baro_alt", "gps_pos", "gps_vel", "lidar_range")


def _attitude(a_cmd: np.ndarray, yaw: float) -> np.ndarray:
    """Body z along the thrust (a - g), body x as close to the heading `yaw` as possible."""
    z = a_cmd + np.array([0.0, 0.0, G])
    z = z / np.linalg.norm(z)
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    x = x - z * (x @ z)
    x /= np.linalg.norm(x)
    rot = np.column_stack([x, np.cross(z, x), z])
    from hopper_core.math3d import dcm_to_quat

    return dcm_to_quat(rot)


def _pva(t: float, scale: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Position, velocity, acceleration, yaw. Position follows 0.5 A (1 - cos)^2 from rest, so
    velocity and acceleration both start at zero (no attitude step when the manoeuvre begins)."""
    tau = max(t - T_STATIC, 0.0)
    amp = 0.5 * scale * np.array([6.0, 4.0, 2.0])
    w = np.array([0.5, 0.35, 0.6])
    ph = w * tau
    c, s_ = np.cos(ph), np.sin(ph)
    p = np.array([0.0, 0.0, 8.0]) + amp * (1 - c) ** 2
    v = amp * w * 2 * (1 - c) * s_
    a = amp * w**2 * (2 * s_**2 + 2 * (1 - c) * c)
    return p, v, a, 0.3 * np.sin(0.2 * tau) * min(tau, 1.0)


@lru_cache(maxsize=8)
def sampled_truth(scale: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """`truth` on the sensor-sampling grid, computed once and shared by every seed."""
    pairs = [truth(k * DT, scale) for k in range(n)]
    return np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])


def truth(t: float, scale: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """Full 19-element state and the body-frame specific force at time `t`."""
    p, v, a, yaw = _pva(t, scale)
    q = _attitude(a, yaw)
    h = 1e-5
    q_hi = _attitude(_pva(t + h, scale)[2], _pva(t + h, scale)[3])
    q_lo = _attitude(_pva(t - h, scale)[2], _pva(t - h, scale)[3])
    dq = quat_mul(quat_conj(q_lo), q_hi)
    if dq[0] < 0:
        dq = -dq
    omega = 2.0 * dq[1:] / (2 * h)
    # q_lo -> q_hi spans 2h; the body-frame rate at t differs from that by O(h^2).
    x = make_state(p, v, q, omega, 20.0)
    f_b = quat_to_dcm(q).T @ (a + np.array([0.0, 0.0, G]))
    return x, f_b


@dataclass
class Sample:
    t: float
    est: object  # StateEstimate | None


class Feeder:
    """Generates `SensorData` at the controller rate from an analytic trajectory."""

    def __init__(self, cfg: VehicleConfig, seed: int, scale: float = 1.0, duration: float = 34.0):
        self.cfg = cfg
        self._x, self._f = sampled_truth(scale, round(duration / DT) + CTRL_EVERY + 1)
        self.suite = SensorSuite(cfg, np.random.default_rng(seed))
        self.scale = scale
        self._latest: dict[str, object] = {}
        self._k = 0

    def set_gyro_bias(self, bias) -> None:
        self.suite._imu._gyro_bias = np.array(bias, dtype=np.float64)

    def biases(self) -> tuple[np.ndarray, np.ndarray, float]:
        imu, baro = self.suite._imu, self.suite._baro
        return imu._gyro_bias.copy(), imu._accel_bias.copy(), float(baro._bias)

    def __iter__(self):
        return self

    def __next__(self) -> tuple[float, SensorData, np.ndarray]:
        """Advance to the next controller call. Returns `(t, SensorData, true state)`.

        The state is the truth when the newest IMU sample was taken (one grid step before the call,
        since the 2 ms latency rounds up to a step): the instant the filter's estimate refers to.
        """
        while True:
            t = self._k * DT
            x, f_b = self._x[self._k], self._f[self._k]
            data = self.suite.sample(t, x, {"specific_force_B": f_b})
            for name in FIELDS:
                if (v := getattr(data, name)) is not None:
                    self._latest[name] = v
            self._k += 1
            if (self._k - 1) % CTRL_EVERY == 0:
                out = SensorData(t, **{f: self._latest.get(f) for f in FIELDS})
                self._latest = {}
                return t, out, self._x[max(self._k - 2, 0)]


def run(ekf, feeder: Feeder, duration: float, **process_kw):
    """Run `ekf.process` for `duration` s. Yields `(t, x_true, estimate_or_None)`."""
    out = []
    for t, data, x in feeder:
        if t > duration:
            break
        out.append((t, x, ekf.process(data, **process_kw)))
    return out


def attitude_error(q_true: np.ndarray, q_est: np.ndarray) -> np.ndarray:
    return quat_error(q_true, q_est)


def make_ekf(cfg: VehicleConfig, **overrides):
    from dataclasses import replace

    from hopper_estimation.ekf import EKF

    # The newest IMU sample is always exactly one grid step old at a call in this harness.
    params = replace(ekf_params_from_config(cfg), imu_delay=DT)
    return EKF(replace(params, **overrides) if overrides else params)


__all__ = [
    "CTRL_EVERY",
    "DT",
    "POS",
    "QUAT",
    "VEL",
    "OMEGA",
    "Feeder",
    "attitude_error",
    "make_ekf",
    "run",
    "truth",
]
