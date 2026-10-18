"""Bridges the simulator's vehicle config to the controller's own parameter set."""

from hopper.config import VehicleConfig
from hopper.control.params import ControlParams


def control_params_from_config(cfg: VehicleConfig) -> ControlParams:
    return ControlParams(
        thrust_max=cfg.engine.thrust_max,
        thrust_min_frac=cfg.engine.thrust_min_frac,
        gimbal_max=cfg.engine.gimbal_max,
        gimbal_rate_max=cfg.engine.gimbal_rate_max,
        max_roll_torque=cfg.rcs.max_roll_torque,
        inertia_diag_dry=tuple(cfg.mass.dry_inertia),
        dry_mass=cfg.mass.dry_mass,
        dry_com_z=cfg.mass.dry_com_z,
        prop_mass_initial=cfg.tank.prop_mass_initial,
        tank_bottom_z=cfg.tank.bottom_z,
        tank_height=cfg.tank.height,
        tank_radius=cfg.tank.radius,
    )
