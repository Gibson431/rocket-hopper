"""Vehicle configuration models and YAML loading (see docs/CONVENTIONS.md)."""

from math import pi, radians
from pathlib import Path
from typing import Annotated, Self

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

Pos = Annotated[float, Field(gt=0)]
NonNeg = Annotated[float, Field(ge=0)]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class MassConfig(_Model):
    dry_mass: Pos
    dry_inertia: Annotated[tuple[Pos, Pos, Pos], Field(description="diag, about dry CoM")]
    dry_com_z: Pos  # above the base


class TankConfig(_Model):
    prop_mass_initial: NonNeg
    bottom_z: NonNeg
    height: Pos
    radius: Pos


class EngineConfig(_Model):
    thrust_max: Pos
    thrust_min_frac: Annotated[float, Field(gt=0, le=1)]
    isp: Pos
    throttle_tau: Pos
    gimbal_max: Annotated[float, Field(gt=0, lt=pi / 4)]
    gimbal_rate_max: Pos
    gimbal_wn: Pos
    gimbal_zeta: Pos


class RcsConfig(_Model):
    max_roll_torque: NonNeg


class LegsConfig(_Model):
    count: Annotated[int, Field(ge=1)]
    radius: Pos  # foot circle
    length: Pos
    stiffness: Pos
    damping: NonNeg
    friction: NonNeg


class AeroConfig(_Model):
    cd_axial: NonNeg
    cd_lateral: NonNeg
    area_axial: NonNeg
    area_lateral: NonNeg
    cp_z: float  # above the base


class SensorsConfig(_Model):
    """Placeholder; filled in S07."""


class SimConfig(_Model):
    dt: Pos = 0.001
    controller_rate: Pos = 100.0
    max_time: Pos = 60.0
    # Outcome thresholds (see hopper.sim.events). Angles in rad.
    v_crash: Pos = 3.0  # vertical speed at first touchdown
    crash_tilt: Pos = radians(30.0)  # tilt at first touchdown
    tipover_tilt: Pos = radians(60.0)  # tilt while in contact
    land_speed: Pos = 0.2
    land_tilt: Pos = radians(10.0)
    land_rate: Pos = 0.2  # rad/s
    land_hold: Pos = 1.0  # s all landing conditions must hold
    max_range: Pos = 500.0  # horizontal distance from the origin
    max_alt: Pos = 200.0

    @model_validator(mode="after")
    def _controller_not_faster_than_physics(self) -> Self:
        if self.controller_rate * self.dt > 1.0 + 1e-9:
            raise ValueError("controller_rate must not exceed the physics rate 1/dt")
        return self


class VehicleConfig(_Model):
    mass: MassConfig
    tank: TankConfig
    engine: EngineConfig
    rcs: RcsConfig
    legs: LegsConfig
    aero: AeroConfig
    sensors: SensorsConfig = SensorsConfig()
    sim: SimConfig = SimConfig()


def load_vehicle(path: str | Path) -> VehicleConfig:
    with open(path) as f:
        return VehicleConfig.model_validate(yaml.safe_load(f))
