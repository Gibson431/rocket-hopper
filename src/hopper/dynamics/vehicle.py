"""Full vehicle state derivative: rigid body + gravity + propulsion + aero/wind.

`vehicle_derivative(t, x, cmd, cfg, env=None, env_force_I=None, env_torque_B=None)`: `env` is an
`Environment` (wind, `gravity_enabled`, `aero_enabled`; `None` means `Environment.default()`).
The extra force/torque hooks are for ground contact (S05). The sim loop must call
`env.wind.advance(dt)` once per step; wind is read-only inside RK4 stages.
"""

import numpy as np

from hopper.config import VehicleConfig
from hopper.dynamics.aero import aero_loads
from hopper.dynamics.environment import Environment, atmosphere, gravity
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.propulsion import propulsion
from hopper.dynamics.rigid_body import rigid_body_derivative
from hopper.math3d import quat_rotate
from hopper.types import POS, PROP, QUAT, VEL, Command


def vehicle_derivative(
    t: float,
    x: np.ndarray,
    cmd: Command,
    cfg: VehicleConfig,
    env: Environment | None = None,
    env_force_I: np.ndarray | None = None,  # noqa: N803 (frame suffix is the project convention)
    env_torque_B: np.ndarray | None = None,  # noqa: N803
) -> np.ndarray:
    """d(state)/dt of the 19-element state for the commanded engine.

    `env_force_I` (N, inertial) and `env_torque_B` (N m about the CoM, body) are extra hooks, e.g.
    ground contact (S05).

    Neglected (small for a slow burn): d(inertia)/dt and CoM motion terms from propellant
    depletion, and the momentum carried by the exhaust.
    """
    mp = mass_properties(x[PROP], cfg)
    p = propulsion(x, cmd, cfg, mp)

    env = env or Environment.default()
    force_i = quat_rotate(x[QUAT], p.force_B)
    torque_b = p.torque_B
    if env.gravity_enabled:
        force_i = force_i + mp.mass * gravity(x[POS])
    if env.aero_enabled:
        rho = atmosphere(x[POS][2]).density
        wind = env.wind.velocity(t, x[POS])
        f_aero, t_aero = aero_loads(x[VEL], x[QUAT], wind, rho, mp, cfg.aero)
        force_i = force_i + f_aero
        torque_b = torque_b + t_aero
    if env_force_I is not None:
        force_i = force_i + env_force_I
    if env_torque_B is not None:
        torque_b = torque_b + env_torque_B

    xd = rigid_body_derivative(x, mp.mass, mp.inertia, force_i, torque_b)
    # Rigid-body part leaves the actuator slots at 0, so the sum merges the two.
    return xd + p.xdot_act
