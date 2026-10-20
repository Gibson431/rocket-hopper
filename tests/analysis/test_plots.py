"""Log plots: every function returns a Figure (never shows), tolerates missing optional columns."""

import importlib.util
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

from matplotlib.figure import Figure  # noqa: E402

from hopper.analysis import plots  # noqa: E402
from hopper.analysis.logger import COLUMNS, save_frame  # noqa: E402
from hopper.scenario_glue import run_hop  # noqa: E402

N = 200


def synthetic_log(n: int = N, with_est: bool = False) -> pd.DataFrame:
    t = np.linspace(0.0, 4.0, n)
    df = pd.DataFrame(0.0, index=range(n), columns=COLUMNS[:-1])
    df["t"] = t
    df["pos_x"], df["pos_z"] = 0.5 * t, 1.0 + np.sin(t)
    df["vel_x"], df["vel_z"] = 0.5, np.cos(t)
    df["quat_w"], df["quat_y"] = np.cos(0.05 * t), np.sin(0.05 * t)
    df["thr"], df["thrust"], df["cmd_throttle"] = 0.5, 1000.0, 0.55
    df["gim_x"], df["cmd_gim_x"] = 0.02 * np.sin(t), 0.03 * np.sin(t)
    df["mass"] = 100.0 - t
    df["event"] = ""
    df.loc[n - 1, "event"] = "LANDED"
    if with_est:
        rng = np.random.default_rng(0)
        for c in "xyz":
            df[f"est_pos_{c}"] = df[f"pos_{c}"] + 0.01 * rng.standard_normal(n)
            df[f"est_vel_{c}"] = df[f"vel_{c}"] + 0.01 * rng.standard_normal(n)
            df[f"sig_pos_{c}"] = 0.01
            df[f"sig_vel_{c}"] = 0.01
    df.attrs.update({"seed": 0, "config_hash": "abc"})
    return df


@pytest.fixture(scope="module")
def real_log() -> pd.DataFrame:
    from hopper.config import load_vehicle

    cfg = load_vehicle(Path(__file__).parents[2] / "configs" / "vehicle_default.yaml")
    res, _ = run_hop(cfg, t_end=3.0, perfect_state=False, backend="drake")
    return res.log


def test_plot_overview_has_six_panels_plus_the_thrust_twin_on_synthetic_and_real_logs(real_log):
    for log in (synthetic_log(), real_log):
        fig = plots.plot_overview(log)
        assert isinstance(fig, Figure)
        assert len(fig.axes) == 7  # six panels and the thrust twin axis


def test_plot_overview_shades_each_mission_phase_and_marks_faults_and_outcome():
    log = synthetic_log()
    log["mission_state"] = ["ASCEND"] * 100 + ["HOVER"] * 60 + ["DESCEND"] * 40
    fig = plots.plot_overview(log, fault_times=[1.0, 2.5])
    top = fig.axes[0]
    assert len(top.patches) == 3  # one span per phase on each axis
    lines_x = [ln.get_xdata()[0] for ln in top.lines if len(ln.get_xdata()) == 2]
    assert 1.0 in lines_x and 2.5 in lines_x
    assert any("LANDED" in tx.get_text() for ax in fig.axes for tx in ax.texts)


def test_plots_work_without_optional_columns():
    log = synthetic_log().drop(columns=["mass", "cmd_roll", "gimr_x"])
    for fn in (plots.plot_overview, plots.plot_trajectory, plots.plot_control):
        assert isinstance(fn(log), Figure)


def test_plot_trajectory_and_control_axes_counts():
    log = synthetic_log()
    assert len(plots.plot_trajectory(log).axes) == 2
    assert len(plots.plot_control(log).axes) == 4


def test_plot_estimation_shows_error_with_three_sigma_bounds(real_log):
    log = synthetic_log(with_est=True)
    fig = plots.plot_estimation(log)
    assert len(fig.axes) == 6
    ax = fig.axes[0]
    ys = [np.asarray(ln.get_ydata()) for ln in ax.lines]
    assert any(np.allclose(y, 0.03) for y in ys)  # +3 sigma, sigma = 0.01
    assert any(np.allclose(y, -0.03) for y in ys)  # -3 sigma
    err = log["est_pos_x"].to_numpy() - log["pos_x"].to_numpy()  # estimate minus truth
    assert any(np.allclose(y, err) for y in ys)
    assert any("est_pos_x" in c for c in real_log.columns)
    assert isinstance(plots.plot_estimation(real_log), Figure)


def test_plot_estimation_without_estimator_columns_gives_a_placeholder_figure():
    fig = plots.plot_estimation(synthetic_log())
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1


def test_plot_mc_summary_has_four_axes():
    rng = np.random.default_rng(1)
    n = 30
    summary = pd.DataFrame(
        {
            "outcome": rng.choice(["LANDED", "CRASHED"], n),
            "landing_error": rng.random(n),
            "touchdown_vspeed": rng.random(n),
            "max_tilt_deg": rng.random(n) * 10,
        }
    )
    fig = plots.plot_mc_summary(summary)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4


def test_save_all_writes_the_expected_files(tmp_path):
    paths = plots.save_all(synthetic_log(), tmp_path / "figs")
    assert sorted(p.name for p in paths) == ["control.png", "overview.png", "trajectory.png"]
    paths = plots.save_all(synthetic_log(with_est=True), tmp_path / "figs2")
    assert sorted(p.name for p in paths) == [
        "control.png",
        "estimation.png",
        "overview.png",
        "trajectory.png",
    ]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_plotting_does_not_open_a_pyplot_figure():
    import matplotlib.pyplot as plt

    before = len(plt.get_fignums())
    plots.plot_overview(synthetic_log())
    assert len(plt.get_fignums()) == before


def test_cli_writes_figures_and_html(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "plot_run", Path(__file__).parents[2] / "scripts" / "plot_run.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    src = tmp_path / "run.parquet"
    save_frame(synthetic_log(), src)
    out, html = tmp_path / "figs", tmp_path / "out.html"
    mod.main([str(src), "--out", str(out), "--html", str(html)])
    assert (out / "overview.png").exists()
    assert html.stat().st_size > 1000


def test_plot_overview_gimbal_panel_is_in_degrees():
    log = synthetic_log()
    ys = [np.asarray(ln.get_ydata()) for ln in plots.plot_overview(log).axes[5].lines]
    assert any(np.allclose(y, np.degrees(log["gim_x"])) for y in ys)
