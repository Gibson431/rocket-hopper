"""Monte Carlo over a dispersed scenario: independent seeded runs, pass/fail statistics.

Run `i` gets the int made from `SeedSequence(base_seed).spawn(n)[i]`. That seed alone fixes
the dispersions, the noise and the gusts, so any run can be repeated singly
(`run_scenario(scenario, seed=s, dispersed=True)`) and the summary does not depend on `workers`.
"""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from hopper.analysis.logger import save_frame
from hopper.scenario import ScenarioRun, load_scenario, run_scenario
from hopper.sim.events import Outcome

METRICS = (
    "landing_error",
    "touchdown_vspeed",
    "touchdown_hspeed",
    "max_tilt_deg",
    "max_gimbal_deg",
    "fuel_used",
    "flight_time",
    "est_rms_pos",
    "est_rms_vel",
    "est_rms_att_deg",
)
PERCENTILES = (5, 50, 95)
_MAX_SEEDS_SHOWN = 20


def run_seeds(base_seed: int, n: int) -> list[int]:
    """Per-run seeds; the first `k` of `run_seeds(b, n)` equal `run_seeds(b, k)`."""
    return [
        int(child.generate_state(1, np.uint64)[0] >> np.uint64(1))
        for child in np.random.SeedSequence(base_seed).spawn(n)
    ]


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95% by default)."""
    p = successes / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, float(centre - half)), min(1.0, float(centre + half))


def percentile_table(
    summary: pd.DataFrame, columns=METRICS, percentiles: tuple[float, ...] = PERCENTILES
) -> pd.DataFrame:
    cols = [c for c in columns if c in summary.columns]
    q = summary[cols].quantile([p / 100 for p in percentiles]).T
    q.columns = [f"p{p:g}" for p in percentiles]
    return q


def _estimation_rms(log: pd.DataFrame, k_td: int) -> tuple[float, float, float]:
    if "est_pos_x" not in log.columns:
        return (np.nan,) * 3
    seg = log.iloc[: k_td + 1]
    seg = seg[seg["est_pos_x"].notna()]
    if len(seg) == 0:
        return (np.nan,) * 3

    def rms(a: str, b: str) -> float:
        d = (
            seg[[f"{a}_{c}" for c in "xyz"]].to_numpy()
            - seg[[f"{b}_{c}" for c in "xyz"]].to_numpy()
        )
        return float(np.sqrt(np.mean(np.sum(d**2, axis=1))))

    dot = np.abs(
        np.sum(
            seg[[f"quat_{c}" for c in "wxyz"]].to_numpy()
            * seg[[f"est_quat_{c}" for c in "wxyz"]].to_numpy(),
            axis=1,
        )
    )
    ang = 2 * np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
    return rms("est_pos", "pos"), rms("est_vel", "vel"), float(np.sqrt(np.mean(ang**2)))


def _row(i: int, run: ScenarioRun) -> dict:
    s = run.summary
    rms = _estimation_rms(run.result.log, s.touchdown_index)
    row = {
        "run": i,
        "seed": run.seed,
        "outcome": s.outcome.name,
        "success": s.outcome is Outcome.LANDED,
        "landing_error": s.landing_error,
        "touchdown_vspeed": s.touchdown_speed,
        "touchdown_hspeed": s.touchdown_hspeed,
        "max_tilt_deg": float(np.degrees(s.max_tilt)),
        "max_gimbal_deg": float(np.degrees(s.max_gimbal)),
        "fuel_used": s.fuel_used,
        "flight_time": s.flight_time,
        "t_end": s.t_end,
        "est_rms_pos": rms[0],
        "est_rms_vel": rms[1],
        "est_rms_att_deg": rms[2],
    }
    row.update({f"disp_{k}": v for k, v in run.dispersions.items()})
    return row


def _run_one(job: tuple[str, int, int, str | None]) -> dict:
    path, i, seed, log_dir = job
    run = run_scenario(load_scenario(path), seed=seed, dispersed=True)
    if log_dir is not None and run.summary.outcome is not Outcome.LANDED:
        save_frame(run.result.log, Path(log_dir) / f"run_{i:04d}_seed_{seed}.parquet")
    return _row(i, run)


@dataclass
class MonteCarloResult:
    scenario: str
    summary: pd.DataFrame
    failed_seeds: list[int] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.summary)

    @property
    def success_rate(self) -> float:
        return float(self.summary["success"].mean())

    @property
    def ci(self) -> tuple[float, float]:
        return wilson_interval(int(self.summary["success"].sum()), self.n)

    def report(self) -> str:
        k = int(self.summary["success"].sum())
        lo, hi = self.ci
        lines = [
            f"scenario {self.scenario}: {k}/{self.n} landed "
            f"({100 * self.success_rate:.1f}%, 95% Wilson interval {100 * lo:.1f}-{100 * hi:.1f}%)",
            "",
            percentile_table(self.summary).to_string(float_format=lambda v: f"{v:.3f}"),
            "",
        ]
        outcomes = self.summary["outcome"].value_counts()
        lines.append("outcomes: " + ", ".join(f"{k} {v}" for k, v in outcomes.items()))
        if self.failed_seeds:
            lines.append("failed seeds (rerun with run_scenario.py --seed S --dispersed):")
            shown = self.failed_seeds[:_MAX_SEEDS_SHOWN]
            lines.append("  " + " ".join(map(str, shown)))
            if len(self.failed_seeds) > len(shown):
                lines.append(
                    f"  ... and {len(self.failed_seeds) - len(shown)} more (see the summary)"
                )
        else:
            lines.append("failed seeds: none")
        return "\n".join(lines)


def run_monte_carlo(
    scenario_path: str | Path,
    n: int,
    base_seed: int = 0,
    workers: int = 1,
    out_dir: str | Path | None = None,
) -> MonteCarloResult:
    """Run `n` dispersed simulations. With `out_dir`, writes `summary.parquet` and the logs of the
    runs that did not land under `logs/`."""
    scenario_path = str(scenario_path)
    name = load_scenario(scenario_path).name  # fail early on a bad file
    log_dir = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        log_dir = out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
    jobs = [
        (scenario_path, i, seed, None if log_dir is None else str(log_dir))
        for i, seed in enumerate(run_seeds(base_seed, n))
    ]
    if workers <= 1:
        rows = [_run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(_run_one, jobs))
    summary = pd.DataFrame(rows)
    if out_dir is not None:
        summary.to_parquet(out_dir / "summary.parquet", index=False)
    failed = [int(s) for s in summary.loc[~summary["success"], "seed"]]
    return MonteCarloResult(name, summary, failed)
