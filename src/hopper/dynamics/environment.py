"""Gravity, ISA troposphere, and wind with Gauss-Markov gusts."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

from hopper.dynamics.propulsion import G0

_T0 = 288.15
_P0 = 101325.0
_LAPSE = 0.0065  # K/m
_R = 287.053  # J/(kg K)
_GAMMA = 1.4
_EXP = G0 / (_LAPSE * _R)  # barometric exponent, ~5.2559


def gravity(pos_i: np.ndarray) -> np.ndarray:
    """Flat-Earth gravity (m/s^2). A function of position so a better model can drop in."""
    return np.array([0.0, 0.0, -G0])


class Atmosphere(NamedTuple):
    density: float
    temperature: float
    pressure: float
    speed_of_sound: float


def atmosphere(alt_m: float) -> Atmosphere:
    """ISA troposphere (0-11 km). Altitude below 0 is clamped to sea level."""
    h = min(max(alt_m, 0.0), 11000.0)
    temp = _T0 - _LAPSE * h
    p = _P0 * (temp / _T0) ** _EXP
    return Atmosphere(p / (_R * temp), temp, p, float(np.sqrt(_GAMMA * _R * temp)))


def pressure_to_altitude(p: float) -> float:
    """Inverse of the ISA pressure law (m)."""
    return _T0 / _LAPSE * (1.0 - (p / _P0) ** (1.0 / _EXP))


class Wind:
    """Mean wind plus a first-order Gauss-Markov gust per axis.

    `velocity` only reads state; the sim loop calls `advance(dt)` once per physics step so RK4
    stages all see the same gust.
    """

    def __init__(self, mean_i: np.ndarray, gust_sigma: float, gust_tau: float, rng):
        self.mean_i = np.asarray(mean_i, dtype=float)
        self.gust_sigma = gust_sigma
        self.gust_tau = gust_tau
        self._rng = rng
        self._gust = np.zeros(3)

    @classmethod
    def none(cls) -> "Wind":
        return cls(np.zeros(3), 0.0, 1.0, np.random.default_rng(0))

    def reset(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)
        self._gust = np.zeros(3)

    def advance(self, dt: float) -> None:
        if self.gust_sigma == 0.0:
            return
        phi = np.exp(-dt / self.gust_tau)  # exact discretisation, valid for any dt
        noise = self._rng.standard_normal(3)
        self._gust = phi * self._gust + self.gust_sigma * np.sqrt(1.0 - phi**2) * noise

    def velocity(self, t: float, pos_i: np.ndarray) -> np.ndarray:
        return self.mean_i + self._gust


@dataclass
class Environment:
    wind: Wind = field(default_factory=Wind.none)
    gravity_enabled: bool = True
    aero_enabled: bool = True  # off for rocket-equation style tests

    @classmethod
    def default(cls) -> "Environment":
        return cls()
