"""Fixed-step RK4 integrator for any 1-D state vector."""

from collections.abc import Callable

import numpy as np

from hopper.types import QUAT

Deriv = Callable[[float, np.ndarray], np.ndarray]


def rk4_step(
    f: Deriv, t: float, x: np.ndarray, dt: float, renorm: slice | None = QUAT
) -> np.ndarray:
    """One RK4 step. The `renorm` slice (the quaternion) is rescaled to unit norm afterwards."""
    k1 = f(t, x)
    k2 = f(t + dt / 2, x + dt / 2 * k1)
    k3 = f(t + dt / 2, x + dt / 2 * k2)
    k4 = f(t + dt, x + dt * k3)
    x_new = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    if renorm is not None:
        x_new[renorm] /= np.linalg.norm(x_new[renorm])
    return x_new


def integrate(
    f: Deriv,
    x0: np.ndarray,
    t0: float,
    t1: float,
    dt: float,
    callback: Callable[[float, np.ndarray], None] | None = None,
    renorm: slice | None = QUAT,
    record: bool = False,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Integrate from t0 to t1 in steps of dt; the last step is shortened to land on t1.

    Returns (final state, trajectory). The trajectory has one row per step including the
    initial state when `record` is true, else it is None. `callback(t, x)` runs after each step.
    """
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}")
    x = np.array(x0, dtype=np.float64)
    rows = [x] if record else None
    span = t1 - t0
    n_full = int(np.floor(span / dt + 1e-9))
    n_steps = n_full + (1 if t0 + n_full * dt < t1 - 1e-12 else 0)
    for k in range(n_steps):
        t = t0 + k * dt
        h = min(dt, t1 - t)
        x = rk4_step(f, t, x, h, renorm)
        if rows is not None:
            rows.append(x)
        if callback is not None:
            callback(t1 if k == n_steps - 1 else t + h, x)
    return x, (np.array(rows) if rows is not None else None)
