"""Plots of a run log. Every function returns a `Figure` (built without pyplot, so nothing is shown
or left open); columns beyond the S06 schema are optional."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from hopper.math3d import quat_to_euler_zyx

_XYZ = "xyz"
_PHASE_COLORS = ("#cfe3f5", "#f5e6c8", "#d9efd2", "#f2d4d4", "#e3d6f0", "#d8e8e8")


def _t(log: pd.DataFrame) -> np.ndarray:
    return log["t"].to_numpy(float)


def _col(log: pd.DataFrame, name: str) -> np.ndarray | None:
    return log[name].to_numpy(float) if name in log.columns else None


def _euler_deg(log: pd.DataFrame) -> np.ndarray:
    q = log[[f"quat_{a}" for a in "wxyz"]].to_numpy(float)
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    return np.degrees([quat_to_euler_zyx(row) for row in q])  # yaw, pitch, roll


def _phase_spans(log: pd.DataFrame) -> list[tuple[float, float, str]]:
    if "mission_state" not in log.columns:
        return []
    t, states = _t(log), log["mission_state"].astype(str).to_numpy()
    spans, start = [], 0
    for k in range(1, len(t) + 1):
        if k == len(t) or states[k] != states[start]:
            spans.append((t[start], t[k] if k < len(t) else t[-1], states[start]))
            start = k
    return spans


def plot_overview(log: pd.DataFrame, fault_times: Sequence[float] = ()) -> Figure:
    """Position, velocity, attitude (Euler angles, display only), body rates, throttle and thrust,
    gimbal actual vs commanded. Phases are shaded when `mission_state` exists."""
    t = _t(log)
    fig = Figure(figsize=(10, 12), layout="constrained")
    axes = fig.subplots(6, 1, sharex=True)
    for a in _XYZ:
        axes[0].plot(t, log[f"pos_{a}"], label=a)
        axes[1].plot(t, log[f"vel_{a}"], label=a)
        axes[3].plot(t, np.degrees(log[f"omega_{a}"]), label=a)
    for lab, series in zip(("yaw", "pitch", "roll"), _euler_deg(log).T, strict=True):
        axes[2].plot(t, series, label=lab)
    axes[4].plot(t, log["thr"], label="throttle (actual)")
    axes[4].plot(t, log["cmd_throttle"], "--", label="throttle (cmd)")
    twin = axes[4].twinx()
    twin.plot(t, log["thrust"], color="tab:red", alpha=0.6, label="thrust")
    twin.set_ylabel("thrust [N]")
    for a, color in zip("xy", ("tab:blue", "tab:orange"), strict=True):
        axes[5].plot(t, np.degrees(log[f"gim_{a}"]), color=color, label=f"gimbal {a}")
        axes[5].plot(t, np.degrees(log[f"cmd_gim_{a}"]), "--", color=color, label=f"cmd {a}")
    for ax, label in zip(
        axes,
        (
            "position [m]",
            "velocity [m/s]",
            "attitude [deg]",
            "body rate [deg/s]",
            "throttle",
            "gimbal [deg]",
        ),
        strict=True,
    ):
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7, ncols=3)
    axes[-1].set_xlabel("t [s]")
    for k, (t0, t1, name) in enumerate(_phase_spans(log)):
        for ax in axes:
            ax.axvspan(t0, t1, color=_PHASE_COLORS[k % len(_PHASE_COLORS)], alpha=0.5, lw=0)
        axes[0].text(
            t0, 1.0, name, transform=axes[0].get_xaxis_transform(), fontsize=7, va="bottom"
        )
    for tf in fault_times:
        for ax in axes:
            ax.axvline(tf, color="tab:red", ls=":", lw=1.2)
    event = str(log["event"].iloc[-1]) if "event" in log.columns else ""
    if event:
        for ax in axes:
            ax.plot([t[-1]], [ax.get_ylim()[0]], "kv", ms=6, clip_on=False)
        axes[0].annotate(
            event, (t[-1], 1.0), xycoords=axes[0].get_xaxis_transform(), ha="right", fontsize=9,
            fontweight="bold",
        )  # fmt: skip
    return fig


def plot_trajectory(log: pd.DataFrame) -> Figure:
    """Plan view (east-north) and side view (range from start vs height)."""
    fig = Figure(figsize=(10, 4.5), layout="constrained")
    plan, side = fig.subplots(1, 2)
    x, y, z = (log[f"pos_{a}"].to_numpy(float) for a in _XYZ)
    plan.plot(x, y)
    plan.plot(x[0], y[0], "go", label="start")
    plan.plot(x[-1], y[-1], "rs", label="end")
    plan.set(xlabel="east [m]", ylabel="north [m]", title="plan view", aspect="equal")
    side.plot(np.hypot(x - x[0], y - y[0]), z)
    side.plot(0.0, z[0], "go")
    side.plot(np.hypot(x[-1] - x[0], y[-1] - y[0]), z[-1], "rs")
    side.set(xlabel="range from start [m]", ylabel="height [m]", title="side view")
    for ax in (plan, side):
        ax.grid(alpha=0.3)
    plan.legend(fontsize=8)
    return fig


def plot_control(log: pd.DataFrame) -> Figure:
    """Command against the actuator: throttle, gimbal x, gimbal y, roll torque command."""
    t = _t(log)
    fig = Figure(figsize=(10, 8), layout="constrained")
    axes = fig.subplots(4, 1, sharex=True)
    axes[0].plot(t, log["thr"], label="actual")
    axes[0].plot(t, log["cmd_throttle"], "--", label="command")
    axes[0].set_ylabel("throttle")
    for ax, a in zip(axes[1:3], "xy", strict=True):
        ax.plot(t, np.degrees(log[f"gim_{a}"]), label="actual")
        ax.plot(t, np.degrees(log[f"cmd_gim_{a}"]), "--", label="command")
        ax.set_ylabel(f"gimbal {a} [deg]")
    roll = _col(log, "cmd_roll")
    axes[3].plot(t, np.zeros_like(t) if roll is None else roll, label="command")
    axes[3].set_ylabel("roll cmd")
    axes[3].set_xlabel("t [s]")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_estimation(log: pd.DataFrame) -> Figure:
    """Estimate minus truth for position and velocity with +/-3 sigma, from `est_*` / `sig_*`."""
    need = [f"{p}_{k}_{a}" for p in ("est",) for k in ("pos", "vel") for a in _XYZ]
    if not all(c in log.columns for c in need):
        fig = Figure(figsize=(6, 3))
        ax = fig.subplots()
        ax.text(0.5, 0.5, "no est_* columns in this log", ha="center", va="center")
        ax.set_axis_off()
        return fig
    t = _t(log)
    fig = Figure(figsize=(12, 7), layout="constrained")
    axes = fig.subplots(2, 3, sharex=True)
    for r, (kind, unit) in enumerate((("pos", "m"), ("vel", "m/s"))):
        for c, a in enumerate(_XYZ):
            ax = axes[r, c]
            err = log[f"est_{kind}_{a}"].to_numpy(float) - log[f"{kind}_{a}"].to_numpy(float)
            ax.plot(t, err, lw=0.8, label="error")
            sig = _col(log, f"sig_{kind}_{a}")
            if sig is not None:
                ax.plot(t, 3 * sig, "r--", lw=0.8, label="+3 sigma")
                ax.plot(t, -3 * sig, "r--", lw=0.8, label="-3 sigma")
            ax.set(title=f"{kind} {a}", ylabel=f"error [{unit}]")
            ax.grid(alpha=0.3)
            if r == 1:
                ax.set_xlabel("t [s]")
    axes[0, 0].legend(fontsize=7)
    return fig


def plot_mc_summary(summary: pd.DataFrame) -> Figure:
    """Outcome counts and histograms of landing error, touchdown speed and maximum tilt."""
    fig = Figure(figsize=(10, 7), layout="constrained")
    axes = fig.subplots(2, 2).ravel()
    counts = summary["outcome"].value_counts()
    axes[0].bar(counts.index.astype(str), counts.to_numpy())
    axes[0].set(title="outcomes", ylabel="runs")
    for ax, col, label in zip(
        axes[1:],
        ("landing_error", "touchdown_vspeed", "max_tilt_deg"),
        ("landing error [m]", "touchdown vertical speed [m/s]", "max tilt [deg]"),
        strict=True,
    ):
        ax.hist(summary[col].dropna().to_numpy(float), bins=20)
        ax.set(xlabel=label, ylabel="runs")
    for ax in axes:
        ax.grid(alpha=0.3)
    return fig


def save_all(log: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    """Write overview, trajectory, control and (when the log has them) estimation figures."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    figs = {
        "overview": plot_overview(log),
        "trajectory": plot_trajectory(log),
        "control": plot_control(log),
    }
    if "est_pos_x" in log.columns:
        figs["estimation"] = plot_estimation(log)
    paths = []
    for name, fig in figs.items():
        path = out / f"{name}.png"
        fig.savefig(path, dpi=110)
        paths.append(path)
    return paths
