"""State layout and the controller-facing interfaces (see docs/CONVENTIONS.md)."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)
PROP = 13
THR = 14
GIM = slice(15, 17)
GIMR = slice(17, 19)
N_STATE = 19


def _vec(name: str, value: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    a = np.asarray(value, dtype=np.float64)
    if a.shape != (n,):
        raise ValueError(f"{name} must have shape ({n},), got {a.shape}")
    return a


def make_state(
    pos: Sequence[float] | np.ndarray,
    vel: Sequence[float] | np.ndarray,
    quat: Sequence[float] | np.ndarray | None,
    omega: Sequence[float] | np.ndarray,
    prop: float,
    thr: float = 0.0,
    gim: Sequence[float] | np.ndarray = (0.0, 0.0),
    gimr: Sequence[float] | np.ndarray = (0.0, 0.0),
) -> np.ndarray:
    """Build the flat state vector. `quat=None` means upright (identity)."""
    x = np.zeros(N_STATE)
    x[POS] = _vec("pos", pos, 3)
    x[VEL] = _vec("vel", vel, 3)
    q = np.array([1.0, 0.0, 0.0, 0.0]) if quat is None else _vec("quat", quat, 4)
    x[QUAT] = q / np.linalg.norm(q)
    x[OMEGA] = _vec("omega", omega, 3)
    x[PROP] = prop
    x[THR] = thr
    x[GIM] = _vec("gim", gim, 2)
    x[GIMR] = _vec("gimr", gimr, 2)
    return x


@dataclass(frozen=True)
class Command:
    throttle: float  # 0..1 requested thrust fraction (0 = engine off)
    gimbal: np.ndarray  # (2,) requested [gx, gy], rad
    roll_torque: float  # -1..1 fraction of max RCS roll torque about z_B

    def __post_init__(self) -> None:
        # Frozen dataclass: bypass the frozen __setattr__ to store the coerced array.
        object.__setattr__(self, "gimbal", _vec("gimbal", self.gimbal, 2))
        # A NaN would silently poison the whole integration; ranges are clamped by propulsion.
        if not (
            np.isfinite(self.throttle)
            and np.isfinite(self.roll_torque)
            and np.all(np.isfinite(self.gimbal))
        ):
            raise ValueError("Command fields must be finite")


@dataclass(frozen=True)
class SensorData:
    """None means no new measurement this tick."""

    t: float
    gyro: np.ndarray | None  # (3,) rad/s, body
    accel: np.ndarray | None  # (3,) m/s^2 specific force, body
    baro_alt: float | None  # m above ground reference
    gps_pos: np.ndarray | None  # (3,) m, I
    gps_vel: np.ndarray | None  # (3,) m/s, I
    lidar_range: float | None  # m along -z_B


@dataclass(frozen=True)
class StateEstimate:
    t: float
    pos: np.ndarray
    vel: np.ndarray
    quat: np.ndarray
    omega: np.ndarray
    mass: float  # estimated total mass (kg)


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None: ...
    def step(self, sensors: SensorData, t: float) -> Command: ...


@runtime_checkable
class StateFeedbackController(Controller, Protocol):
    def step_state(self, est: StateEstimate, t: float) -> Command: ...
