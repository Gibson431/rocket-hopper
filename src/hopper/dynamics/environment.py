"""Gravity, ISA troposphere, and the wind and switches that describe the environment."""

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np

G0 = 9.80665  # m/s^2, standard gravity
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


@dataclass(frozen=True)
class Wind:
    """Mean wind plus a first-order Gauss-Markov gust per axis. The gust is drawn by
    `hopper.drake.wind.WindSource`, seeded from the run seed."""

    mean_i: np.ndarray
    gust_sigma: float = 0.0
    gust_tau: float = 1.0

    @classmethod
    def none(cls) -> "Wind":
        return cls(np.zeros(3))


@dataclass
class Environment:
    wind: Wind = field(default_factory=Wind.none)
    gravity_enabled: bool = True
    aero_enabled: bool = True  # off for rocket-equation style tests

    @classmethod
    def default(cls) -> "Environment":
        return cls()
