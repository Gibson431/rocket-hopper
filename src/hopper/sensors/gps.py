"""GPS position and velocity: white noise, random dropouts and outage windows."""

import numpy as np

from hopper.config import GpsConfig
from hopper.types import POS, VEL


class Gps:
    def __init__(self, cfg: GpsConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng
        self._pos_sigma = np.array([cfg.pos_sigma_h, cfg.pos_sigma_h, cfg.pos_sigma_v])

    def measure(self, t: float, x: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
        """`None` during an outage window or on a random dropout."""
        c, rng = self.cfg, self._rng
        # Always draw so the stream does not depend on which samples are dropped.
        dropped = rng.random() < c.dropout_prob
        pos = x[POS] + self._pos_sigma * rng.standard_normal(3)
        vel = x[VEL] + c.vel_sigma * rng.standard_normal(3)
        if dropped or any(a <= t < b for a, b in c.outages):
            return None
        return pos, vel
