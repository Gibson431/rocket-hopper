"""The full hop flown on the EKF estimate (perfect_state=False), truth and estimate in the log."""

import numpy as np
import pytest

from hopper.config import load_vehicle
from hopper.math3d import quat_error
from hopper.scenario_glue import ekf_params_from_config, run_hop
from hopper.sim.events import Outcome

NOISELESS = {
    "imu": dict(
        gyro_noise_density=0.0,
        gyro_bias_init=0.0,
        gyro_bias_rw=0.0,
        accel_noise_density=0.0,
        accel_bias_init=0.0,
        accel_bias_rw=0.0,
    ),
    "baro": dict(noise_sigma=0.0, bias_rw=0.0),
    "gps": dict(pos_sigma_h=0.0, pos_sigma_v=0.0, vel_sigma=0.0),
    "lidar": dict(sigma_abs=0.0, sigma_rel=0.0),
}


def noiseless(cfg):
    groups = {k: getattr(cfg.sensors, k).model_copy(update=v) for k, v in NOISELESS.items()}
    return cfg.model_copy(update={"sensors": cfg.sensors.model_copy(update=groups)})


def errors(res, until_touchdown=False):
    """Per-row position (m), velocity (m/s), attitude (rad) and mass (kg) estimation errors."""
    df = res.log.dropna(subset=["est_pos_x"])
    if until_touchdown:  # the contact impact is a few-ms event the 100 Hz IMU cannot resolve
        flying = np.flatnonzero(df["pos_z"].to_numpy() > 3.0)
        down = np.flatnonzero((df["n_contacts"].to_numpy() > 0) & (np.arange(len(df)) > flying[0]))
        df = df.iloc[: down[0]]
    pos = (
        df[[f"pos_{a}" for a in "xyz"]].to_numpy() - df[[f"est_pos_{a}" for a in "xyz"]].to_numpy()
    )
    vel = (
        df[[f"vel_{a}" for a in "xyz"]].to_numpy() - df[[f"est_vel_{a}" for a in "xyz"]].to_numpy()
    )
    q = df[[f"quat_{a}" for a in "wxyz"]].to_numpy()
    qe = df[[f"est_quat_{a}" for a in "wxyz"]].to_numpy()
    att = np.array([np.linalg.norm(quat_error(a, b)) for a, b in zip(q, qe, strict=True)])
    return df, pos, vel, att, (df["mass"] - df["est_mass"]).to_numpy()


def rms(a):
    return (
        float(np.sqrt((np.asarray(a) ** 2).sum(axis=-1).mean()))
        if np.ndim(a) > 1
        else float(np.sqrt((np.asarray(a) ** 2).mean()))
    )


@pytest.fixture(scope="module")
def cfg():
    return load_vehicle("configs/vehicles/vehicle_default.yaml")


@pytest.fixture(scope="module")
def noisy(cfg):
    return run_hop(cfg, seed=0, perfect_state=False)


@pytest.fixture(scope="module")
def clean(cfg):
    return run_hop(
        noiseless(cfg), seed=0, perfect_state=False, ekf_params=ekf_params_from_config(cfg)
    )


def test_noise_free_sensors_estimate_tracks_truth_across_the_hop(clean):
    res, s = clean
    assert res.outcome is Outcome.LANDED
    _, pos, vel, att, dm = errors(res, until_touchdown=True)
    print("noise-free rms", rms(pos), rms(vel), np.degrees(rms(att)), np.abs(dm).max())
    assert rms(pos) < 0.02
    assert rms(vel) < 0.01
    assert np.degrees(rms(att)) < 0.02
    assert np.abs(dm).max() < 0.01


def test_estimate_settles_at_rest_on_the_pad_after_touchdown(noisy):
    res, _ = noisy
    _, pos, vel, _, _ = errors(res)
    assert np.linalg.norm(pos[-1]) < 0.3
    assert np.linalg.norm(vel[-1]) < 0.1  # the mission's own "at rest" threshold


def test_nominal_noise_rms_errors_meet_the_budget_over_a_full_hop(noisy):
    res, _ = noisy
    _, pos, vel, att, dm = errors(res)
    print("nominal rms", rms(pos), rms(vel), np.degrees(rms(att)), np.abs(dm).max())
    assert rms(pos) < 0.5
    assert rms(vel) < 0.1
    assert np.degrees(rms(att)) < 0.5
    assert np.abs(dm).max() < 0.5  # dead-reckoned propellant mass


def test_hop_flown_on_the_ekf_estimate_lands_softly_on_the_site(noisy):
    res, s = noisy
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 1.0
    assert s.touchdown_speed < 1.5
    assert np.degrees(s.touchdown_tilt) < 5.0


def test_log_carries_the_estimate_and_its_standard_deviations(noisy):
    res, _ = noisy
    cols = set(res.log.columns)
    for c in ("est_pos_x", "est_vel_z", "est_quat_w", "est_mass", "est_bg_x", "est_ba_z", "est_bb"):
        assert c in cols
    assert sum(c.startswith("sig_") for c in cols) == 16
    df = res.log
    assert df["est_pos_x"].iloc[0] != df["est_pos_x"].iloc[0]  # NaN before alignment completes
    assert df["sig_pos_x"].dropna().iloc[-1] < df["sig_pos_x"].dropna().iloc[0]
