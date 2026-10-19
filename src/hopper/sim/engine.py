"""Closed-loop simulation: fixed-step RK4 physics, multi-rate controller, command delay, logging."""

import hashlib
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hopper.analysis.logger import Logger
from hopper.config import VehicleConfig
from hopper.control.interface import Controller, StateFeedbackController
from hopper.dynamics.contact import contact_loads
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import MassProperties, mass_properties
from hopper.dynamics.propulsion import propulsion
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.sensors.suite import SensorSuite
from hopper.sim.events import EventMonitor, Outcome
from hopper.sim.faults import Fault
from hopper.sim.integrator import rk4_step
from hopper.types import OMEGA, POS, PROP, QUAT, VEL, Command, SensorData, StateEstimate

ENGINE_OFF = Command(0.0, np.zeros(2), 0.0)
_SENSOR_FIELDS = ("gyro", "accel", "baro_alt", "gps_pos", "gps_vel", "lidar_range")


@dataclass
class SimResult:
    outcome: Outcome  # RUNNING when the run stopped at `t_end` before any outcome
    t_end: float
    log: pd.DataFrame
    seed: int | None


def config_hash(cfg: VehicleConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


def _seed_of(rng: np.random.Generator) -> int | None:
    entropy = getattr(rng.bit_generator.seed_seq, "entropy", None)
    return entropy if isinstance(entropy, int) else None


class Simulation:
    """One closed-loop run. The wind is reseeded from `rng` so the seed fixes everything."""

    def __init__(
        self,
        cfg: VehicleConfig,
        controller: Controller | StateFeedbackController,
        env: Environment,
        x0: np.ndarray,
        rng: np.random.Generator,
        perfect_state: bool = True,
        command_delay: float = 0.0,
        log_decimation: int = 1,
        faults: Sequence[Fault] = (),
    ) -> None:
        if command_delay < 0:
            raise ValueError(f"command_delay must be >= 0, got {command_delay}")
        self.cfg = cfg
        self.controller = controller
        self.env = env
        self.x0 = np.array(x0, dtype=np.float64)
        self.rng = rng
        self.perfect_state = perfect_state
        self.command_delay = command_delay
        self.log_decimation = log_decimation
        self.faults = tuple(faults)
        self.seed = _seed_of(rng)
        env.wind.reset(int(rng.integers(2**32)))
        self.sensors = None if perfect_state else SensorSuite(cfg, rng)
        self._latest: dict[
            str, object
        ] = {}  # newest delivered sample per field since the last call

    def _sense(self, t: float, x: np.ndarray, cmd: Command, cfg: VehicleConfig) -> None:
        """Advance the sensors one physics step and keep the newest sample per field."""
        info: dict = {}
        if self.sensors.needs_info(t):
            vehicle_derivative(t, x, cmd, cfg, self.env, info=info)
        data = self.sensors.sample(t, x, info)
        for fault in self.faults:
            data = fault.modify_sensors(t, data)
        for name in _SENSOR_FIELDS:
            if (v := getattr(data, name)) is not None:
                self._latest[name] = v

    def _build_controller_input(self, t: float, x: np.ndarray, mp: MassProperties):
        """Seam for the estimator (S10): true state, or the sensor samples since the last call."""
        if not self.perfect_state:
            data = SensorData(t, **{f: self._latest.get(f) for f in _SENSOR_FIELDS})
            self._latest = {}
            return data
        return StateEstimate(
            t, x[POS].copy(), x[VEL].copy(), x[QUAT].copy(), x[OMEGA].copy(), mp.mass
        )

    def _call_controller(self, t: float, x: np.ndarray, mp: MassProperties) -> Command:
        inp = self._build_controller_input(t, x, mp)
        if self.perfect_state:
            return self.controller.step_state(inp, t)
        return self.controller.step(inp, t)

    def run(self, t_end: float | None = None) -> SimResult:
        cfg, dt = self.cfg, self.cfg.sim.dt
        ctrl_every = max(1, round(1.0 / (cfg.sim.controller_rate * dt)))
        delay_steps = round(self.command_delay / dt)
        n_max = int(np.ceil(t_end / dt - 1e-9)) if t_end is not None else None

        # A controller may expose `log_columns` and `log_values()` (e.g. its state estimate).
        extra_cols = tuple(getattr(self.controller, "log_columns", ()))
        logger = Logger(
            self.log_decimation, {"seed": self.seed, "config_hash": config_hash(cfg)}, extra_cols
        )
        monitor = EventMonitor(cfg)
        self.controller.reset()
        x = self.x0.copy()
        faults = self.faults
        phys, epochs = cfg, None  # physics config: nominal until a parameter fault triggers
        raw = applied = ENGINE_OFF
        pending: deque[tuple[int, Command]] = deque()  # (first step it applies to, command)
        outcome = Outcome.RUNNING
        k = 0
        while True:
            t = k * dt
            if faults:
                key = tuple(f.param_epoch(t) for f in faults)
                if key != epochs:
                    phys, epochs = cfg, key
                    for f in faults:
                        phys = f.modify_params(t, phys)
            mp = mass_properties(x[PROP], cfg)
            contact = contact_loads(x, mp, cfg.legs)
            event = monitor.update(t, x, contact)
            done = event is not None or (n_max is not None and k >= n_max)
            if self.sensors is not None and not done:
                self._sense(t, x, applied, phys)
            if not done and k % ctrl_every == 0:
                pending.append((k + delay_steps, self._call_controller(t, x, mp)))
            while pending and pending[0][0] <= k:
                raw = applied = pending.popleft()[1]
            if faults:
                applied = raw
                for f in faults:
                    applied = f.apply_command(t, applied)

            keep = event is not None or done or logger.wants_row()
            thrust = 0.0
            if keep:
                thrust = float(np.linalg.norm(propulsion(x, applied, phys, mp).force_B))
            logger.log(
                t,
                x,
                mp.mass,
                thrust,
                applied,
                float(contact.foot_forces.sum()),
                int(np.count_nonzero(contact.foot_forces > 0.0)),
                event if done else None,
                self.controller.log_values()
                if extra_cols and keep
                else (np.nan,) * len(extra_cols),
            )
            if done:
                if event is not None:
                    outcome = event
                break

            cmd = applied
            x = rk4_step(
                lambda tt, xx, c=cmd, p=phys: vehicle_derivative(tt, xx, c, p, self.env), t, x, dt
            )
            self.env.wind.advance(dt)
            k += 1

        return SimResult(outcome, t, logger.to_dataframe(), self.seed)
