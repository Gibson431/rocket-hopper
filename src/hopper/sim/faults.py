"""Time-triggered faults, applied by the Drake diagram at three seams.

    apply_command(t, cmd)   what the actuators receive (stuck gimbal, engine out)
    modify_params(t, cfg)   the physics parameters (thrust scale, slower gimbal); the controller
                            keeps the nominal config, so it does not know
    modify_sensors(t, data) what the sensors deliver (GPS outage, bias step)

`param_epoch(t)` changes whenever `modify_params` would return a different config, so the engine
rebuilds the effective config only on those steps. Faults are frozen pydantic models so a scenario
file can list them (`type` is the discriminator).
"""

from dataclasses import replace
from typing import Annotated, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from hopper.config import VehicleConfig
from hopper.types import Command, SensorData


class Fault(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    def apply_command(self, t: float, cmd: Command) -> Command:
        return cmd

    def param_epoch(self, t: float) -> int:
        return 0

    def modify_params(self, t: float, cfg: VehicleConfig) -> VehicleConfig:
        return cfg

    def modify_sensors(self, t: float, data: SensorData) -> SensorData:
        return data


def _engine(cfg: VehicleConfig, **update) -> VehicleConfig:
    return cfg.model_copy(update={"engine": cfg.engine.model_copy(update=update)})


class GimbalStuck(Fault):
    """The gimbal axis is driven to `value_deg` and ignores the controller from `at`."""

    type: Literal["gimbal_stuck"] = "gimbal_stuck"
    axis: Literal["x", "y"]
    at: float
    value_deg: float

    def apply_command(self, t: float, cmd: Command) -> Command:
        if t < self.at:
            return cmd
        gimbal = cmd.gimbal.copy()
        gimbal["xy".index(self.axis)] = np.radians(self.value_deg)
        return replace(cmd, gimbal=gimbal)


class EngineOut(Fault):
    type: Literal["engine_out"] = "engine_out"
    at: float

    def apply_command(self, t: float, cmd: Command) -> Command:
        return replace(cmd, throttle=0.0) if t >= self.at else cmd


class ThrustScale(Fault):
    type: Literal["thrust_scale"] = "thrust_scale"
    at: float
    factor: Annotated[float, Field(ge=0)]

    def param_epoch(self, t: float) -> int:
        return int(t >= self.at)

    def modify_params(self, t: float, cfg: VehicleConfig) -> VehicleConfig:
        if t < self.at:
            return cfg
        return _engine(cfg, thrust_max=cfg.engine.thrust_max * self.factor)


class GimbalSlow(Fault):
    type: Literal["gimbal_slow"] = "gimbal_slow"
    at: float
    rate_factor: Annotated[float, Field(gt=0, le=1)]

    def param_epoch(self, t: float) -> int:
        return int(t >= self.at)

    def modify_params(self, t: float, cfg: VehicleConfig) -> VehicleConfig:
        if t < self.at:
            return cfg
        return _engine(cfg, gimbal_rate_max=cfg.engine.gimbal_rate_max * self.rate_factor)


class GpsOutage(Fault):
    """No GPS samples delivered in `[start, start + duration)`."""

    type: Literal["gps_outage"] = "gps_outage"
    start: float
    duration: Annotated[float, Field(gt=0)]

    def modify_sensors(self, t: float, data: SensorData) -> SensorData:
        if self.start <= t < self.start + self.duration:
            return replace(data, gps_pos=None, gps_vel=None)
        return data


class GyroBiasStep(Fault):
    """Adds a constant rate to the gyro from `at`: every axis, or one if `axis` is given."""

    type: Literal["gyro_bias_step"] = "gyro_bias_step"
    at: float
    value_dps: float
    axis: Literal["x", "y", "z", "all"] = "all"

    def modify_sensors(self, t: float, data: SensorData) -> SensorData:
        if t < self.at or data.gyro is None:
            return data
        bias = np.zeros(3)
        if self.axis == "all":
            bias[:] = np.radians(self.value_dps)
        else:
            bias["xyz".index(self.axis)] = np.radians(self.value_dps)
        return replace(data, gyro=data.gyro + bias)


class GyroBias(Fault):
    """A constant gyro offset vector (rad/s) for the whole run; Monte Carlo uses it."""

    type: Literal["gyro_bias"] = "gyro_bias"
    bias: tuple[float, float, float]

    def modify_sensors(self, t: float, data: SensorData) -> SensorData:
        if data.gyro is None:
            return data
        return replace(data, gyro=data.gyro + np.asarray(self.bias))


AnyFault = Annotated[
    GimbalStuck | EngineOut | ThrustScale | GimbalSlow | GpsOutage | GyroBiasStep,
    Field(discriminator="type"),
]
