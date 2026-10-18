"""EKF unit behaviour: alignment, propagation, covariance, each measurement update."""

from dataclasses import replace

import numpy as np
import pytest
from estimation_harness import Feeder, G, make_ekf, run

from hopper.math3d import quat_from_axis_angle, quat_mul, quat_to_dcm
from hopper.types import StateEstimate

DEG = np.pi / 180


def tilt_between(q_a, q_b) -> float:
    dq = quat_mul(np.array([q_a[0], *-q_a[1:]]), q_b)
    return 2 * np.arctan2(np.linalg.norm(dq[1:]), abs(dq[0]))


def hover_ekf(cfg, **overrides):
    """EKF initialised upright at 10 m, at rest, with default covariances."""
    ekf = make_ekf(cfg, **overrides)
    ekf.initialise(0.0, pos=[0.0, 0.0, 10.0], vel=[0.0, 0.0, 0.0], quat=[1.0, 0.0, 0.0, 0.0])
    return ekf


# --- alignment ---------------------------------------------------------------------------


def test_returns_none_until_aligned_then_levels_the_attitude_and_zeroes_velocity(cfg):
    ekf, feeder = make_ekf(cfg), Feeder(cfg, 1)
    out = run(ekf, feeder, 1.5)
    first = next(i for i, (_, _, e) in enumerate(out) if e is not None)
    assert out[first][0] >= ekf.params.align_time - 0.011
    assert all(e is None for _, _, e in out[:first])
    _, x, est = out[-1]
    assert isinstance(est, StateEstimate)
    assert tilt_between(est.quat, x[6:10]) < 0.5 * DEG
    assert np.linalg.norm(est.vel) < 0.05
    assert est.pos[2] == pytest.approx(8.0, abs=0.1)  # lidar fixes the height
    assert np.hypot(*est.pos[:2]) < 1.0  # GPS fixes the position


def test_alignment_recovers_a_tilted_start(cfg):
    ekf = make_ekf(cfg)
    q = quat_from_axis_angle([1.0, 1.0, 0.0], 20 * DEG)
    f_b = quat_to_dcm(q).T @ np.array([0.0, 0.0, G])
    t = 0.0
    while not ekf.initialised:
        from hopper.types import SensorData

        ekf.process(SensorData(t, np.zeros(3), f_b, None, None, None, None))
        t += 0.01
    # yaw is assumed zero, so compare the vertical as seen from the body (the third row of R)
    z_est = quat_to_dcm(ekf.q)[2, :]
    z_true = quat_to_dcm(q)[2, :]
    assert np.arccos(np.clip(z_est @ z_true, -1, 1)) < 0.01 * DEG


# --- propagation -------------------------------------------------------------------------


def _reference(p0, v0, q0, omega, f_b, t_end, dt=1e-3):
    """Fine RK4 reference for p, v, q with constant body rate and body specific force."""
    g = np.array([0.0, 0.0, -G])

    def deriv(s):
        v, q = s[3:6], s[6:]
        qd = 0.5 * quat_mul(q, np.concatenate(([0.0], omega)))
        return np.concatenate([v, quat_to_dcm(q) @ f_b + g, qd])

    s = np.concatenate([p0, v0, q0])
    for _ in range(round(t_end / dt)):
        k1 = deriv(s)
        k2 = deriv(s + dt / 2 * k1)
        k3 = deriv(s + dt / 2 * k2)
        k4 = deriv(s + dt * k3)
        s = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        s[6:] /= np.linalg.norm(s[6:])
    return s


def test_imu_integration_matches_a_fine_reference_for_a_rotating_vehicle(cfg):
    omega, f_b = np.array([0.5, -0.2, 0.3]), np.array([0.5, 0.2, 11.0])
    ekf = make_ekf(cfg)
    q0 = quat_from_axis_angle([0.0, 1.0, 0.0], 0.1)
    ekf.initialise(0.0, pos=[0.0, 0.0, 10.0], vel=[1.0, 0.0, 0.0], quat=q0)
    for k in range(1, 401):
        ekf.update_imu(0.01 * k, omega, f_b)
    ref = _reference(np.array([0.0, 0.0, 10.0]), np.array([1.0, 0.0, 0.0]), q0, omega, f_b, 4.0)
    assert np.linalg.norm(ekf.p - ref[:3]) < 1e-3
    assert np.linalg.norm(ekf.v - ref[3:6]) < 1e-3
    assert tilt_between(ekf.q, ref[6:]) < 1e-4


def test_bias_estimates_are_subtracted_from_the_imu_before_integration(cfg):
    ekf = make_ekf(cfg)
    ekf.initialise(
        0.0,
        pos=[0.0, 0.0, 10.0],
        vel=[0.0] * 3,
        quat=[1.0, 0, 0, 0],
        bg=[0.01, 0, 0],
        ba=[0, 0, 0.1],
    )
    for k in range(1, 101):  # gyro reads the bias, accel reads g plus its bias
        ekf.update_imu(0.01 * k, [0.01, 0.0, 0.0], [0.0, 0.0, G + 0.1])
    assert tilt_between(ekf.q, [1.0, 0.0, 0.0, 0.0]) < 1e-9
    assert abs(ekf.v[2]) < 1e-9


def test_free_fall_accelerometer_reads_zero_and_velocity_follows_gravity(cfg):
    ekf = hover_ekf(cfg)
    for k in range(1, 201):
        ekf.update_imu(0.01 * k, np.zeros(3), np.zeros(3))
    assert ekf.v[2] == pytest.approx(-G * 2.0, abs=1e-9)
    assert ekf.p[2] == pytest.approx(10.0 - 0.5 * G * 4.0, abs=1e-9)


def test_dead_reckoning_covariance_grows_with_the_imu_noise_densities(cfg):
    n, dt = 100, 0.01
    ekf = hover_ekf(
        cfg,
        accel_bias_init=0.0,
        accel_bias_rw=0.0,
        gyro_bias_init=0.0,
        gyro_bias_rw=0.0,
        init_vel_sigma=0.0,
        init_att_sigma=(0.0, 0.0, 0.0),
    )
    imu = cfg.sensors.imu
    for k in range(1, n + 1):
        ekf.update_imu(dt * k, np.zeros(3), [0.0, 0.0, G])
    var_a = imu.accel_noise_density**2 * imu.rate  # per-sample variance
    var_g = imu.gyro_noise_density**2 * imu.rate
    assert ekf.P[5, 5] == pytest.approx(var_a * dt**2 * n, rel=0.02)  # vertical velocity
    assert ekf.P[8, 8] == pytest.approx(var_g * dt**2 * n, rel=0.02)  # yaw about the vertical


def test_covariance_stays_symmetric_and_positive_definite_through_a_long_run(cfg):
    ekf, feeder = make_ekf(cfg), Feeder(cfg, 3)
    run(ekf, feeder, 15.0)
    assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
    assert np.linalg.eigvalsh(ekf.P).min() > 0.0


# --- measurement updates -----------------------------------------------------------------


def test_baro_update_moves_the_height_toward_the_measurement_and_shrinks_its_variance(cfg):
    ekf = hover_ekf(cfg)
    var0 = ekf.P[2, 2]
    assert ekf.update_baro(0.0, 12.0)
    assert 10.0 < ekf.p[2] < 12.0
    assert ekf.P[2, 2] < var0


def test_baro_bias_state_absorbs_a_persistent_offset_when_lidar_knows_the_height(cfg):
    ekf = hover_ekf(cfg)
    for k in range(400):  # baro reads 1 m high, lidar reads the true 10 m
        ekf.update_imu(0.01 * (k + 1), np.zeros(3), [0.0, 0.0, G])
        ekf.update_baro(0.01 * (k + 1), 11.0)
        ekf.update_lidar(0.01 * (k + 1), 10.0)
    assert ekf.bb == pytest.approx(1.0, abs=0.15)
    assert ekf.p[2] == pytest.approx(10.0, abs=0.05)


def test_lidar_range_is_projected_through_the_estimated_tilt(cfg):
    ekf = make_ekf(cfg)
    q = quat_from_axis_angle([1.0, 0.0, 0.0], 30 * DEG)
    ekf.initialise(0.0, pos=[0.0, 0.0, 9.0], vel=[0.0, 0.0, 0.0], quat=q)  # 1 m low
    slant = 10.0 / np.cos(30 * DEG)
    for _ in range(30):
        assert ekf.update_lidar(0.0, slant)
    assert ekf.p[2] == pytest.approx(10.0, abs=0.1)  # a tilt-blind filter would say 11.55


def test_lidar_is_ignored_when_the_estimated_tilt_exceeds_45_degrees(cfg):
    ekf = make_ekf(cfg)
    q = quat_from_axis_angle([1.0, 0.0, 0.0], 50 * DEG)
    ekf.initialise(0.0, pos=[0.0, 0.0, 9.0], vel=[0.0, 0.0, 0.0], quat=q)
    p_before = ekf.p.copy()
    assert not ekf.update_lidar(0.0, 10.0 / np.cos(50 * DEG))
    assert np.array_equal(ekf.p, p_before)


def test_lidar_outlier_is_rejected_by_the_innovation_gate(cfg):
    ekf = hover_ekf(cfg)
    for _ in range(40):
        ekf.update_lidar(0.0, 10.0)
    z_before = ekf.p[2]
    assert not ekf.update_lidar(0.0, 15.0)
    assert ekf.p[2] == z_before
    assert ekf.update_lidar(0.0, 10.02)


def test_gps_updates_pull_position_and_velocity(cfg):
    ekf = hover_ekf(cfg)
    assert ekf.update_gps(0.0, [0.5, -0.5, 10.5], [0.1, 0.0, 0.0])
    assert ekf.p[0] > 0.0 > ekf.p[1]
    assert ekf.v[0] > 0.0


def test_zero_velocity_update_pins_velocity_at_rest(cfg):
    ekf = make_ekf(cfg, init_vel_sigma=0.3)  # a start-up velocity error the pad update must remove
    ekf.initialise(0.0, pos=[0.0, 0.0, 10.0], vel=[0.3, 0.0, 0.0], quat=[1.0, 0, 0, 0])
    for _ in range(20):
        ekf.update_zero_velocity(0.0)
    assert np.linalg.norm(ekf.v) < 0.05


def test_stale_measurement_older_than_the_history_is_dropped_and_counted(cfg):
    ekf = hover_ekf(cfg)
    for k in range(1, 301):
        ekf.update_imu(0.01 * k, np.zeros(3), [0.0, 0.0, G])
    z = ekf.p[2]
    assert not ekf.update_baro(0.5, 99.0)  # 2.5 s old; the history is shorter
    assert ekf.p[2] == z
    assert ekf.n_stale == 1


# --- along a manoeuvre -------------------------------------------------------------------


def _rms(out, skip=5.0):
    e = np.array(
        [np.concatenate([x[:3] - est.pos, x[3:6] - est.vel]) for t, x, est in out if t > skip]
    )
    return np.sqrt((e**2).mean(axis=0))


def test_gps_outage_inflates_the_covariance_and_the_estimate_stays_bounded(cfg):
    sensors = cfg.sensors.model_copy(
        update={"gps": cfg.sensors.gps.model_copy(update={"outages": ((8.0, 13.0),)})}
    )
    cfg = cfg.model_copy(update={"sensors": sensors})
    ekf, feeder = make_ekf(cfg), Feeder(cfg, 4)
    sigma_at, err_at = {}, {}
    for t, data, x in feeder:
        if t > 16.0:
            break
        est = ekf.process(data)
        for mark in (8.0, 13.0, 16.0):
            if abs(t - mark) < 0.006:
                sigma_at[mark] = np.sqrt(ekf.P[0, 0])
                err_at[mark] = np.linalg.norm(x[:3] - est.pos)
    assert sigma_at[13.0] > 2.0 * sigma_at[8.0]
    assert err_at[13.0] < 3.0
    assert err_at[16.0] < 1.0  # GPS is accepted again after the outage


def test_gps_spike_of_50_m_is_rejected_and_the_next_good_fix_accepted(cfg):
    ekf, feeder = make_ekf(cfg), Feeder(cfg, 5)
    out = run(ekf, feeder, 10.0)
    t, x, est = out[-1]
    p = est.pos.copy()
    assert not ekf.update_gps(t, x[:3] + [50.0, 0.0, 0.0], x[3:6])
    assert np.allclose(ekf.p, p, atol=1e-9)
    assert ekf.n_rejected["gps"] >= 1
    assert ekf.update_gps(t, x[:3] + [0.3, 0.0, 0.0], x[3:6])


def test_persistent_gps_disagreement_is_accepted_after_the_reject_timeout(cfg):
    ekf, feeder = make_ekf(cfg, reject_timeout=1.0), Feeder(cfg, 5)
    out = run(ekf, feeder, 10.0)
    t0, x0, est0 = out[-1]
    shifted = np.array([20.0, 0.0, 0.0])
    held = None
    for t, data, x in feeder:
        if t > t0 + 4.0:
            break
        if data.gps_pos is not None:
            data = replace(data, gps_pos=data.gps_pos + shifted)
        est = ekf.process(data)
        if abs(t - (t0 + 0.7)) < 0.006:
            held = est.pos[0] - x[0]  # still ignoring the shifted fixes
    assert abs(held) < 1.0
    assert est.pos[0] - x[0] > 10.0  # then trusts them: a filter must not lock itself out
    assert ekf.n_rejected["gps"] >= 5


def test_buffered_gps_latency_compensation_beats_naive_application(cfg):
    results = {}
    for name, latency in (("buffered", cfg.sensors.gps.latency), ("naive", 0.0)):
        ekf = make_ekf(cfg, gps_latency=latency)
        out = run(ekf, Feeder(cfg, 6, scale=2.0), 30.0)
        results[name] = _rms(out)
    assert results["buffered"][:2].max() < 0.7 * results["naive"][:2].max()


def test_constant_gyro_bias_is_estimated_within_10_percent_after_30_s_of_motion(cfg):
    imu = cfg.sensors.imu.model_copy(update={"gyro_bias_init": 0.0, "gyro_bias_rw": 0.0})
    cfg = cfg.model_copy(update={"sensors": cfg.sensors.model_copy(update={"imu": imu})})
    bias = np.array([0.5, -0.5, 0.5]) * DEG
    ekf = make_ekf(cfg, gyro_bias_init=1.0 * DEG)
    feeder = Feeder(cfg, 7, scale=2.0)
    feeder.set_gyro_bias(bias)
    run(ekf, feeder, 32.0)  # 2 s hover for alignment, then 30 s of motion
    assert np.abs(ekf.bg - bias).max() < 0.1 * 0.5 * DEG


def test_estimate_tracks_the_manoeuvre_with_nominal_noise(cfg):
    ekf, feeder = make_ekf(cfg), Feeder(cfg, 8)
    out = run(ekf, feeder, 30.0)
    pos_rms, vel_rms = _rms(out)[:3], _rms(out)[3:]
    assert pos_rms.max() < 0.5
    assert vel_rms.max() < 0.1
