"""Engine model: throttle lag, gimbal actuator dynamics, thrust force/torque, mass flow."""

from dataclasses import dataclass

import numpy as np

from hopper.config import VehicleConfig
from hopper.dynamics.mass_model import MassProperties
from hopper.math3d import cross
from hopper.types import GIM, GIMR, N_STATE, PROP, THR, Command

G0 = 9.80665


@dataclass(frozen=True)
class PropulsionOutput:
    force_B: np.ndarray  # noqa: N815 (frame suffix is the project convention)
    torque_B: np.ndarray  # noqa: N815
    xdot_act: np.ndarray  # (19,) derivatives of PROP, THR, GIM, GIMR; zero elsewhere


def propulsion(
    x: np.ndarray, cmd: Command, cfg: VehicleConfig, mp: MassProperties
) -> PropulsionOutput:
    eng = cfg.engine
    has_prop = x[PROP] > 0.0

    # 0 = off; any other request is clamped into the throttleable range.
    if cmd.throttle <= 0.0 or not has_prop:
        thr_cmd = 0.0
    else:
        thr_cmd = min(max(cmd.throttle, eng.thrust_min_frac), 1.0)
    thrust = eng.thrust_max * x[THR] if has_prop else 0.0

    # The physical angle never exceeds the limit, even if the integrator overshoots it.
    g = np.clip(x[GIM], -eng.gimbal_max, eng.gimbal_max)
    gx, gy = g
    d_b = np.array([np.cos(gx) * np.sin(gy), -np.sin(gx), np.cos(gx) * np.cos(gy)])
    force_b = thrust * d_b
    torque_b = cross([0.0, 0.0, -mp.com_z], force_b)
    torque_b[2] += float(np.clip(cmd.roll_torque, -1.0, 1.0)) * cfg.rcs.max_roll_torque

    xdot = np.zeros(N_STATE)
    xdot[PROP] = -thrust / (eng.isp * G0)
    xdot[THR] = (thr_cmd - x[THR]) / eng.throttle_tau

    g_cmd = np.clip(cmd.gimbal, -eng.gimbal_max, eng.gimbal_max)
    rate = np.clip(x[GIMR], -eng.gimbal_rate_max, eng.gimbal_rate_max)
    acc = eng.gimbal_wn**2 * (g_cmd - x[GIM]) - 2 * eng.gimbal_zeta * eng.gimbal_wn * x[GIMR]
    # Stop accelerating outward at the rate limit and at the angle limit.
    acc = np.where((np.abs(x[GIMR]) >= eng.gimbal_rate_max) & (acc * x[GIMR] > 0), 0.0, acc)
    at_limit = (np.abs(x[GIM]) >= eng.gimbal_max) & (x[GIM] * rate > 0)
    rate = np.where(at_limit, 0.0, rate)
    acc = np.where(at_limit & (acc * x[GIM] > 0), 0.0, acc)
    xdot[GIM] = rate
    xdot[GIMR] = acc
    return PropulsionOutput(force_b, torque_b, xdot)
