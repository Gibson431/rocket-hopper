"""Downward lidar: slant range along -z_B to the ground plane z = 0."""

import numpy as np
from hopper_core.math3d import quat_to_dcm
from hopper_core.types import POS, QUAT

from hopper.config import LidarConfig


class Lidar:
    def __init__(self, cfg: LidarConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng
        self._cos_tilt = np.cos(cfg.max_tilt)

    def measure(self, x: np.ndarray) -> float | None:
        c = self.cfg
        rot = quat_to_dcm(x[QUAT])
        down_z = -rot[2, 2]  # z component of the inertial ray direction
        height = x[POS][2] + (rot @ np.asarray(c.mount_b))[2]
        # Draw first so a miss does not shift the noise stream.
        noise = self._rng.standard_normal()
        if -down_z < self._cos_tilt or height <= 0.0:
            return None
        rng_true = height / -down_z
        if not c.min_range <= rng_true <= c.max_range:
            return None
        return float(rng_true + (c.sigma_abs + c.sigma_rel * rng_true) * noise)
