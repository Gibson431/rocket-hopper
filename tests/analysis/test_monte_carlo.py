import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from hopper.analysis.monte_carlo import (
    percentile_table,
    run_monte_carlo,
    run_seeds,
    wilson_interval,
)
from hopper.scenario import load_scenario, run_scenario, sample_dispersions

ROOT = Path(__file__).parent.parent.parent
QUICK = {"target_alt": 3.0, "hover_time": 0.5, "target_xy": [1.0, 0.0]}
SMALL = {
    "dry_mass": {"dist": "normal", "sigma_frac": 0.01},
    "com_z": {"dist": "normal", "sigma": 0.005},
    "thrust_scale": {"dist": "normal", "sigma_frac": 0.01},
    "wind_mean": {"dist": "uniform", "low": [-1, -1, 0], "high": [1, 1, 0]},
    "gyro_bias": {"dist": "normal", "sigma_dps": 0.05},
    "initial_pos": {"dist": "normal", "sigma": [0.1, 0.1, 0]},
}


def _scenario(tmp_path, name="s", **overrides):
    d = {
        "name": name,
        "vehicle": str(ROOT / "configs" / "vehicles" / "vehicle_default.yaml"),
        "mission": QUICK,
        "dispersions": SMALL,
        **overrides,
    }
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_run_seeds_are_distinct_reproducible_and_a_prefix_of_longer_runs():
    a = run_seeds(7, 50)
    assert len(set(a)) == 50
    assert a == run_seeds(7, 50)
    assert a[:10] == run_seeds(7, 10)
    assert a != run_seeds(8, 50)
    assert all(isinstance(s, int) and 0 <= s < 2**63 for s in a)


def test_wilson_interval_matches_known_values():
    lo, hi = wilson_interval(20, 20)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(20 / (20 + 1.96**2), abs=1e-3)
    lo, hi = wilson_interval(0, 20)
    assert lo == pytest.approx(0.0, abs=1e-12) and hi == pytest.approx(
        1.96**2 / (20 + 1.96**2), abs=1e-3
    )
    lo, hi = wilson_interval(50, 100)  # textbook value: 0.404 .. 0.596
    assert (lo, hi) == pytest.approx((0.4038, 0.5962), abs=1e-3)


def test_percentile_table_reports_the_requested_percentiles():
    df = pd.DataFrame({"landing_error": np.arange(101.0), "fuel_used": np.full(101, 2.0)})
    t = percentile_table(df, ["landing_error", "fuel_used"], percentiles=(5, 50, 95))
    assert t.loc["landing_error", "p5"] == pytest.approx(5.0)
    assert t.loc["landing_error", "p50"] == pytest.approx(50.0)
    assert t.loc["landing_error", "p95"] == pytest.approx(95.0)
    assert t.loc["fuel_used", "p50"] == 2.0


@pytest.fixture(scope="module")
def small_mc(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("mc")
    path = _scenario(tmp)
    return path, run_monte_carlo(path, n=20, base_seed=11, workers=8, out_dir=tmp / "out"), tmp


def test_nominal_hop_with_small_dispersions_always_lands(small_mc):
    _, mc, _ = small_mc
    assert len(mc.summary) == 20
    assert mc.success_rate == 1.0
    assert mc.failed_seeds == []
    assert (mc.summary["landing_error"] < 0.5).all()
    assert mc.ci[0] > 0.8 and mc.ci[1] == pytest.approx(1.0)


def test_summary_has_metrics_seeds_and_the_sampled_dispersions(small_mc):
    path, mc, _ = small_mc
    df = mc.summary
    for col in (
        "run",
        "seed",
        "outcome",
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
    ):
        assert col in df.columns, col
    assert list(df["seed"]) == run_seeds(11, 20)
    d = sample_dispersions(load_scenario(path), int(df["seed"][3]))
    assert df["disp_dry_mass"][3] == pytest.approx(d["dry_mass"])
    assert df["disp_wind_e"][3] == pytest.approx(d["wind_e"])
    assert df["disp_dry_mass"].std() > 0.0
    assert np.isnan(df["est_rms_pos"]).all()  # perfect-state scenario: no estimator


def test_summary_parquet_round_trips(small_mc):
    _, mc, tmp = small_mc
    back = pd.read_parquet(tmp / "out" / "summary.parquet")
    pd.testing.assert_frame_equal(back, mc.summary)


def test_report_states_the_success_rate_interval_and_percentiles(small_mc):
    _, mc, _ = small_mc
    text = mc.report()
    assert "20/20" in text and "Wilson" in text and "p50" in text and "landing_error" in text


def test_same_base_seed_gives_identical_summaries_for_one_and_four_workers(tmp_path):
    path = _scenario(tmp_path)
    one = run_monte_carlo(path, n=4, base_seed=3, workers=1)
    four = run_monte_carlo(path, n=4, base_seed=3, workers=4)
    pd.testing.assert_frame_equal(one.summary, four.summary, check_exact=True)


@pytest.fixture(scope="module")
def failing_mc(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("fail")
    path = _scenario(tmp, faults=[{"type": "engine_out", "at": 3.0}])
    return path, run_monte_carlo(path, n=3, base_seed=5, workers=3, out_dir=tmp / "out"), tmp


def test_failed_runs_are_reported_with_their_seeds_and_logs(failing_mc):
    _, mc, tmp = failing_mc
    assert mc.success_rate == 0.0
    assert mc.failed_seeds == list(mc.summary["seed"])
    logs = sorted((tmp / "out" / "logs").glob("*.parquet"))
    assert len(logs) == 3
    assert all(str(s) in p.name for s, p in zip(mc.failed_seeds, logs, strict=True))
    assert "failed seeds" in mc.report().lower()


def test_a_failed_seed_rerun_singly_reproduces_the_same_run(failing_mc):
    path, mc, _ = failing_mc
    row = mc.summary.iloc[1]
    again = run_scenario(load_scenario(path), seed=int(row["seed"]), dispersed=True)
    assert again.summary.outcome.name == row["outcome"]
    assert again.summary.landing_error == row["landing_error"]
    assert again.summary.fuel_used == row["fuel_used"]
    assert again.dispersions["dry_mass"] == row["disp_dry_mass"]


def test_ekf_scenario_reports_estimation_rms(tmp_path):
    mc = run_monte_carlo(_scenario(tmp_path, estimation="ekf", dispersions={}), n=1, workers=1)
    row = mc.summary.iloc[0]
    assert 0.0 < row["est_rms_pos"] < 1.0
    assert 0.0 < row["est_rms_vel"] < 0.5
    assert 0.0 < row["est_rms_att_deg"] < 2.0


def test_cli_runs_end_to_end(tmp_path):
    path = _scenario(tmp_path)
    out = tmp_path / "cli"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "run_monte_carlo.py"), str(path)]
        + ["-n", "2", "--workers", "2", "--out", str(out)],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stderr
    assert "2/2" in r.stdout and (out / "summary.parquet").exists()
