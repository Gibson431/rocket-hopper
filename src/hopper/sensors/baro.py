"""Barometric altitude: ISA pressure round trip, white noise and a random-walk bias."""

import numpy as np
from hopper_core.types import POS

from hopper.config import BaroConfig
from hopper.dynamics.environment import atmosphere, pressure_to_altitude


class Baro:
    def __init__(self, cfg: BaroConfig, rng: np.random.Generator) -> None:
        self.cfg = cfg
        self._rng = rng
        self._bias = 0.0
        self._sqrt_dt = np.sqrt(1.0 / cfg.rate)

    def measure(self, x: np.ndarray) -> float:
        self._bias += self.cfg.bias_rw * self._sqrt_dt * self._rng.standard_normal()
        alt = pressure_to_altitude(atmosphere(x[POS][2]).pressure)
        return float(alt + self._bias + self.cfg.noise_sigma * self._rng.standard_normal())
