"""What a simulation run returns."""

import hashlib
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hopper.config import VehicleConfig
from hopper.sim.events import Outcome


@dataclass
class SimResult:
    outcome: Outcome  # RUNNING when the run stopped at `t_end` before any outcome
    t_end: float
    log: pd.DataFrame
    seed: int | None


def config_hash(cfg: VehicleConfig) -> str:
    return hashlib.sha256(cfg.model_dump_json().encode()).hexdigest()[:16]


def seed_of(rng: np.random.Generator) -> int | None:
    entropy = getattr(rng.bit_generator.seed_seq, "entropy", None)
    return entropy if isinstance(entropy, int) else None
