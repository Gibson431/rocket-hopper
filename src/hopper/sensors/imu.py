"""Gyro and accelerometer: white noise plus a random-walk bias, with saturation."""

import numpy as np

from hopper.config import ImuConfig
from hopper.types import OMEGA


class Imu:
    def __init__(self, cfg: ImuConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng
        self._gyro_bias = cfg.gyro_bias_init * rng.standard_normal(3)
        self._accel_bias = cfg.accel_bias_init * rng.standard_normal(3)
        self._sqrt_rate = np.sqrt(cfg.rate)
        self._sqrt_dt = 1.0 / self._sqrt_rate

    def measure(self, x: np.ndarray, specific_force_B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:  # noqa: N803
        c, rng = self.cfg, self._rng
        # Bias walks one nominal sample period per measurement.
        self._gyro_bias = self._gyro_bias + c.gyro_bias_rw * self._sqrt_dt * rng.standard_normal(3)
        self._accel_bias = self._accel_bias + c.accel_bias_rw * self._sqrt_dt * rng.standard_normal(
            3
        )
        gyro = (
            x[OMEGA]
            + self._gyro_bias
            + c.gyro_noise_density * self._sqrt_rate * rng.standard_normal(3)
        )
        accel = (
            specific_force_B
            + self._accel_bias
            + c.accel_noise_density * self._sqrt_rate * rng.standard_normal(3)
        )
        return np.clip(gyro, -c.gyro_range, c.gyro_range), np.clip(
            accel, -c.accel_range, c.accel_range
        )
