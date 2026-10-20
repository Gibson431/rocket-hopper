"""Declarative scenarios: a YAML file fixes the vehicle, mission, environment, faults and (for Monte
Carlo) the dispersions. `build_simulation` turns one into a ready `DrakeSimulation`.

The controller and the estimator are always built from the nominal vehicle config; dispersions and
faults change only the simulated truth, so the flight software is flying a vehicle it mis-models.
"""

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import yaml
from pydrake.all import Meshcat
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, create_model

from hopper.config import VehicleConfig, load_vehicle
from hopper.control.mission import MissionController, MissionParams
from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment, Wind
from hopper.dynamics.mass_model import mass_properties
from hopper.estimated_controller import EstimatedController
from hopper.math3d import quat_from_euler_zyx
from hopper.scenario_glue import (
    HopSummary,
    control_params_from_config,
    ekf_params_from_config,
    summarize_hop,
)
from hopper.sim.faults import AnyFault, Fault, GyroBias
from hopper.sim.result import SimResult
from hopper.types import make_state

Vec3 = tuple[float, float, float]
_SAG = 0.01  # m, legs start slightly compressed so the pad counts as ground contact from t = 0


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _mission_model():
    fields = {f.name: (f.type, f.default) for f in dataclasses.fields(MissionParams)}
    return create_model("MissionSpec", __base__=_Model, **fields)


MissionSpec = _mission_model()


class WindSpec(_Model):
    mean: Vec3 = (0.0, 0.0, 0.0)
    gust_sigma: Annotated[float, Field(ge=0)] = 0.0
    gust_tau: Annotated[float, Field(gt=0)] = 1.0


class InitialSpec(_Model):
    """`pos` is relative to the pad; z = 0 means standing on it."""

    pos: Vec3 = (0.0, 0.0, 0.0)
    vel: Vec3 = (0.0, 0.0, 0.0)
    tilt_deg: float = 0.0
    yaw_deg: float = 0.0


class NormalFrac(_Model):
    dist: Literal["normal"] = "normal"
    sigma_frac: Annotated[float, Field(ge=0)]


class Normal(_Model):
    dist: Literal["normal"] = "normal"
    sigma: Annotated[float, Field(ge=0)]


class NormalVec(_Model):
    dist: Literal["normal"] = "normal"
    sigma: tuple[Annotated[float, Field(ge=0)], ...] = Field(min_length=3, max_length=3)


class NormalDps(_Model):
    dist: Literal["normal"] = "normal"
    sigma_dps: Annotated[float, Field(ge=0)]


class UniformVec(_Model):
    dist: Literal["uniform"] = "uniform"
    low: Vec3
    high: Vec3


class Dispersions(_Model):
    dry_mass: NormalFrac | None = None
    com_z: Normal | None = None
    thrust_scale: NormalFrac | None = None
    wind_mean: UniformVec | None = None
    gyro_bias: NormalDps | None = None
    initial_pos: NormalVec | None = None

    def is_empty(self) -> bool:
        return all(getattr(self, f) is None for f in type(self).model_fields)


class Scenario(_Model):
    name: str
    vehicle: str = "configs/vehicle_default.yaml"
    seed: int = 0
    estimation: Literal["ekf", "perfect"] = "perfect"
    t_max: Annotated[float, Field(gt=0)] = 60.0
    initial: InitialSpec = InitialSpec()
    mission: MissionSpec = Field(default_factory=MissionSpec)  # type: ignore[valid-type]
    wind: WindSpec = WindSpec()
    command_delay: Annotated[float, Field(ge=0)] = 0.0
    faults: tuple[AnyFault, ...] = ()
    dispersions: Dispersions = Dispersions()
    _source: Path | None = PrivateAttr(default=None)  # the file it was loaded from

    @property
    def mission_params(self) -> MissionParams:
        return MissionParams(**self.mission.model_dump())  # type: ignore[attr-defined]

    def vehicle_path(self) -> Path:
        """The vehicle file: as given, else relative to the scenario file or any parent of it."""
        p = Path(self.vehicle)
        if p.is_absolute() or p.exists() or self._source is None:
            return p
        for d in self._source.parents:
            if (d / p).exists():
                return d / p
        return p


def load_scenario(path: str | Path) -> Scenario:
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)
    scenario = Scenario.model_validate(data)
    scenario._source = path.resolve()
    return scenario


def _vehicle(scenario: Scenario) -> VehicleConfig:
    return load_vehicle(scenario.vehicle_path())


_STREAMS = ("dry_mass", "com_z", "thrust_scale", "wind_mean", "gyro_bias", "initial_pos")


def sample_dispersions(scenario: Scenario, seed: int) -> dict[str, float]:
    """Draw the scenario's dispersions; a pure function of `seed`.

    Absolute values: `dry_mass` (kg), `wind_e/n/u` (m/s), `pos_x/y/z` (m), `gyro_bias_x/y/z`
    (deg/s). Offset or factor: `com_z` (m), `thrust_scale`. Each quantity has its own random stream,
    so adding one does not move the others.
    """
    d = scenario.dispersions
    if d.is_empty():
        return {}
    cfg = _vehicle(scenario)
    rngs = dict(
        zip(
            _STREAMS,
            (np.random.default_rng(s) for s in np.random.SeedSequence([seed, 1]).spawn(6)),
            strict=True,
        )
    )
    out: dict[str, float] = {}
    if d.dry_mass:
        m = cfg.mass.dry_mass
        out["dry_mass"] = float(rngs["dry_mass"].normal(m, d.dry_mass.sigma_frac * m))
    if d.com_z:
        out["com_z"] = float(rngs["com_z"].normal(0.0, d.com_z.sigma))
    if d.thrust_scale:
        out["thrust_scale"] = float(rngs["thrust_scale"].normal(1.0, d.thrust_scale.sigma_frac))
    if d.wind_mean:
        w = rngs["wind_mean"].uniform(d.wind_mean.low, d.wind_mean.high)
        out.update(zip(("wind_e", "wind_n", "wind_u"), map(float, w), strict=True))
    if d.gyro_bias:
        b = rngs["gyro_bias"].normal(0.0, d.gyro_bias.sigma_dps, 3)
        out.update(zip(("gyro_bias_x", "gyro_bias_y", "gyro_bias_z"), map(float, b), strict=True))
    if d.initial_pos:
        p = rngs["initial_pos"].normal(0.0, d.initial_pos.sigma)
        out.update(zip(("pos_x", "pos_y", "pos_z"), map(float, p), strict=True))
    return out


def _true_config(nominal: VehicleConfig, disp: dict[str, float]) -> VehicleConfig:
    mass, eng = {}, {}
    if "dry_mass" in disp:
        mass["dry_mass"] = disp["dry_mass"]
    if "com_z" in disp:
        mass["dry_com_z"] = nominal.mass.dry_com_z + disp["com_z"]
    if "thrust_scale" in disp:
        eng["thrust_max"] = nominal.engine.thrust_max * disp["thrust_scale"]
    update = {}
    if mass:
        update["mass"] = nominal.mass.model_copy(update=mass)
    if eng:
        update["engine"] = nominal.engine.model_copy(update=eng)
    return nominal.model_copy(update=update) if update else nominal


@dataclass(frozen=True)
class BuiltScenario:
    sim: DrakeSimulation
    cfg: VehicleConfig  # the truth (dispersed)
    mission: MissionParams
    origin_xy: tuple[float, float]  # take-off point; mission targets are offsets from it


def _build(
    scenario: Scenario,
    seed: int | None,
    dispersions: dict[str, float] | None,
    log_decimation: int,
    meshcat: Meshcat | None = None,
) -> BuiltScenario:
    seed = scenario.seed if seed is None else seed
    disp = dispersions or {}
    nominal = _vehicle(scenario)
    cfg = _true_config(nominal, disp)
    mission = scenario.mission_params

    ini = scenario.initial
    pos = np.array(ini.pos) + np.array([disp.get(f"pos_{a}", 0.0) for a in "xyz"])
    com_z = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z
    z = com_z + cfg.legs.length + pos[2] - (_SAG if pos[2] == 0.0 else 0.0)
    quat = quat_from_euler_zyx(np.radians(ini.yaw_deg), np.radians(ini.tilt_deg), 0.0)
    x0 = make_state([pos[0], pos[1], z], ini.vel, quat, [0, 0, 0], cfg.tank.prop_mass_initial)

    rng = np.random.default_rng(seed)
    wind = scenario.wind
    mean = [disp.get(k, m) for k, m in zip(("wind_e", "wind_n", "wind_u"), wind.mean, strict=True)]
    env = Environment(wind=Wind(np.array(mean), wind.gust_sigma, wind.gust_tau))

    controller = MissionController(control_params_from_config(nominal), mission)
    perfect = scenario.estimation == "perfect"
    if not perfect:
        controller = EstimatedController(ekf_params_from_config(nominal), controller)

    faults: list[Fault] = list(scenario.faults)
    if "gyro_bias_x" in disp:
        bias = np.radians([disp[f"gyro_bias_{a}"] for a in "xyz"])
        faults.append(GyroBias(bias=tuple(map(float, bias))))
    sim = DrakeSimulation(
        cfg,
        controller,
        env,
        x0,
        rng,
        perfect_state=perfect,
        command_delay=scenario.command_delay,
        log_decimation=log_decimation,
        faults=faults,
        meshcat=meshcat,
    )
    return BuiltScenario(sim, cfg, mission, (float(pos[0]), float(pos[1])))


def build_simulation(
    scenario: Scenario,
    seed: int | None = None,
    dispersions: dict[str, float] | None = None,
    log_decimation: int = 10,
) -> DrakeSimulation:
    """A ready `DrakeSimulation`; `dispersions` come from `sample_dispersions` (none by default)."""
    return _build(scenario, seed, dispersions, log_decimation).sim


@dataclass
class ScenarioRun:
    result: SimResult
    summary: HopSummary
    dispersions: dict[str, float]
    seed: int


def run_scenario(
    scenario: Scenario,
    seed: int | None = None,
    dispersed: bool = False,
    log_decimation: int = 10,
    meshcat: Meshcat | None = None,
) -> ScenarioRun:
    seed = scenario.seed if seed is None else seed
    disp = sample_dispersions(scenario, seed) if dispersed else {}
    built = _build(scenario, seed, disp, log_decimation, meshcat)
    res = built.sim.run(scenario.t_max)
    summary = summarize_hop(res, built.cfg, built.mission, built.origin_xy)
    return ScenarioRun(res, summary, disp, seed)
