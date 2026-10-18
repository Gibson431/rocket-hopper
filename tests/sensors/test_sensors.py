import numpy as np
import pytest

from hopper.config import SensorsConfig
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.math3d import quat_from_axis_angle
from hopper.sensors.suite import SensorSuite
from hopper.types import Command, make_state

G = 9.80665
OFF = Command(0.0, np.zeros(2), 0.0)
DT = 0.001


def quiet(**over):
    """All noise and bias off; override per-sensor fields with `imu={...}` etc."""
    base = {
        "imu": {
            "gyro_noise_density": 0.0,
            "gyro_bias_init": 0.0,
            "gyro_bias_rw": 0.0,
            "accel_noise_density": 0.0,
            "accel_bias_init": 0.0,
            "accel_bias_rw": 0.0,
            "latency": 0.0,
        },
        "baro": {"noise_sigma": 0.0, "bias_rw": 0.0, "latency": 0.0},
        "gps": {"pos_sigma_h": 0.0, "pos_sigma_v": 0.0, "vel_sigma": 0.0, "latency": 0.0},
        "lidar": {"sigma_abs": 0.0, "sigma_rel": 0.0, "latency": 0.0},
    }
    for k, v in over.items():
        base[k] = {**base[k], **v}
    return SensorsConfig.model_validate(base)


def make_suite(cfg, sensors, seed=0):
    return SensorSuite(cfg.model_copy(update={"sensors": sensors}), np.random.default_rng(seed))


def run(suite, x, seconds, info=None, state_fn=None):
    """Step the suite at dt = 1 ms; return list of (t, SensorData)."""
    out = []
    for k in range(int(round(seconds / DT)) + 1):
        t = k * DT
        xs = state_fn(t) if state_fn else x
        out.append((t, suite.sample(t, xs, info or {"specific_force_B": np.array([0, 0, G])})))
    return out


def stack(samples, field):
    return np.array([getattr(s, field) for _, s in samples if getattr(s, field) is not None])


def hover(z=10.0):
    return make_state([0, 0, z], [0, 0, 0], None, [0, 0, 0], 20.0)


# --- specific force from the vehicle model -------------------------------------------------


def test_specific_force_is_zero_in_free_fall(cfg):
    x = make_state([0, 0, 50], [0, 0, 0], None, [0, 0, 0], 20.0)
    info = {}
    vehicle_derivative(0.0, x, OFF, cfg, Environment(aero_enabled=False), info=info)
    assert info["specific_force_B"] == pytest.approx(np.zeros(3), abs=1e-9)


def test_specific_force_at_rest_on_legs_is_plus_g_upward(cfg):
    mp = mass_properties(20.0, cfg)
    k, n = cfg.legs.stiffness, cfg.legs.count
    depth = mp.mass * G / (n * k)
    z = mp.com_z + cfg.legs.length - depth
    x = make_state([0, 0, z], [0, 0, 0], None, [0, 0, 0], 20.0)
    info = {}
    vehicle_derivative(0.0, x, OFF, cfg, Environment(aero_enabled=False), info=info)
    assert info["specific_force_B"] == pytest.approx([0, 0, G], abs=1e-6)


def test_specific_force_from_thrust_is_thrust_over_mass_in_body_frame(cfg):
    x = make_state(
        [0, 0, 50],
        [0, 0, 0],
        quat_from_axis_angle(np.array([1.0, 0, 0]), 0.5),
        [0, 0, 0],
        20.0,
        thr=1.0,
    )
    info = {}
    vehicle_derivative(0.0, x, OFF, cfg, Environment(aero_enabled=False), info=info)
    mass = mass_properties(20.0, cfg).mass
    assert info["specific_force_B"] == pytest.approx([0, 0, cfg.engine.thrust_max / mass], rel=1e-9)


# --- IMU -----------------------------------------------------------------------------------


def test_static_imu_reads_upward_specific_force_and_bias_only_gyro(cfg):
    s = quiet(imu={"gyro_bias_init": 1e-3})
    suite = make_suite(cfg, s)
    samples = run(suite, hover(), 1.0)
    acc, gyr = stack(samples, "accel"), stack(samples, "gyro")
    assert acc.mean(axis=0) == pytest.approx([0, 0, G], abs=1e-9)
    # one bias value per axis, all small and constant (no walk, no noise)
    assert np.ptp(gyr, axis=0) == pytest.approx(np.zeros(3), abs=1e-12)
    assert np.abs(gyr[0]).max() > 0 and np.abs(gyr[0]).max() < 1e-2


def test_gyro_measures_body_rate(cfg):
    x = make_state([0, 0, 10], [0, 0, 0], None, [0.1, -0.2, 0.3], 20.0)
    samples = run(make_suite(cfg, quiet()), x, 0.1)
    assert stack(samples, "gyro")[-1] == pytest.approx([0.1, -0.2, 0.3], abs=1e-12)


def test_imu_noise_statistics_match_density(cfg):
    s = quiet(imu={"gyro_noise_density": 1e-3, "accel_noise_density": 2e-2, "rate": 400.0})
    samples = run(make_suite(cfg, s, seed=3), hover(), 25.0)
    gyr, acc = stack(samples, "gyro"), stack(samples, "accel")
    assert len(gyr) >= 10000
    assert gyr.std(axis=0) == pytest.approx(np.full(3, 1e-3 * np.sqrt(400.0)), rel=0.05)
    assert acc.std(axis=0) == pytest.approx(np.full(3, 2e-2 * np.sqrt(400.0)), rel=0.05)
    assert gyr.mean(axis=0) == pytest.approx(
        np.zeros(3), abs=4 * 1e-3 * 20 / np.sqrt(len(gyr))
    )  # 4 standard errors


def test_gyro_bias_random_walk_variance_grows_linearly(cfg):
    s = quiet(imu={"gyro_bias_rw": 1e-3, "rate": 100.0})
    finals = {}
    for span in (2.0, 4.0):
        vals = []
        for seed in range(300):
            samples = run(make_suite(cfg, s, seed), hover(), span)
            vals.append(stack(samples, "gyro")[-1])
        finals[span] = np.var(np.array(vals), axis=0).mean()
    assert finals[2.0] == pytest.approx(1e-6 * 2.0, rel=0.25)
    assert finals[4.0] / finals[2.0] == pytest.approx(2.0, rel=0.25)


def test_imu_saturates_at_configured_range(cfg):
    x = make_state([0, 0, 10], [0, 0, 0], None, [50.0, -50.0, 0.0], 20.0)
    s = quiet(imu={"gyro_range": 10.0, "accel_range": 5.0})
    samples = run(make_suite(cfg, s), x, 0.1, info={"specific_force_B": np.array([100.0, 0, -100])})
    assert stack(samples, "gyro")[-1] == pytest.approx([10.0, -10.0, 0.0])
    assert stack(samples, "accel")[-1] == pytest.approx([5.0, 0.0, -5.0])


# --- baro ----------------------------------------------------------------------------------


def test_baro_reads_altitude(cfg):
    samples = run(make_suite(cfg, quiet()), hover(123.0), 0.2)
    assert stack(samples, "baro_alt")[-1] == pytest.approx(123.0, abs=1e-6)


def test_baro_noise_and_bias_walk(cfg):
    s = quiet(baro={"noise_sigma": 0.5})
    alts = stack(run(make_suite(cfg, s, 5), hover(), 250.0), "baro_alt")
    assert len(alts) >= 10000
    assert alts.std() == pytest.approx(0.5, rel=0.05)
    assert alts.mean() == pytest.approx(10.0, abs=0.05)
    s = quiet(baro={"bias_rw": 0.05, "rate": 50.0})
    ends = [
        stack(run(make_suite(cfg, s, k), hover(), 4.0), "baro_alt")[-1] - 10 for k in range(300)
    ]
    assert np.var(ends) == pytest.approx(0.05**2 * 4.0, rel=0.25)


# --- GPS -----------------------------------------------------------------------------------


def test_gps_noise_statistics(cfg):
    s = quiet(gps={"pos_sigma_h": 0.5, "pos_sigma_v": 1.0, "vel_sigma": 0.05, "rate": 100.0})
    x = make_state([5, -3, 10], [1, 2, 3], None, [0, 0, 0], 20.0)
    samples = run(make_suite(cfg, s, 7), x, 110.0)
    pos, vel = stack(samples, "gps_pos"), stack(samples, "gps_vel")
    assert len(pos) >= 10000
    assert pos.std(axis=0) == pytest.approx([0.5, 0.5, 1.0], rel=0.05)
    assert vel.std(axis=0) == pytest.approx(np.full(3, 0.05), rel=0.05)
    assert pos.mean(axis=0) == pytest.approx([5, -3, 10], abs=0.05)
    assert vel.mean(axis=0) == pytest.approx([1, 2, 3], abs=0.005)


def test_gps_outage_window_gives_none_for_its_duration(cfg):
    s = quiet(gps={"rate": 10.0, "outages": [(2.0, 4.0)]})
    samples = run(make_suite(cfg, s), hover(), 6.0)
    times = [t for t, d in samples if d.gps_pos is not None]
    assert not any(2.0 <= t < 4.0 for t in times)
    assert any(t < 2.0 for t in times) and any(t >= 4.0 for t in times)
    assert [t for t, d in samples if d.gps_vel is not None] == times


def test_gps_dropout_probability(cfg):
    s = quiet(gps={"rate": 100.0, "dropout_prob": 0.3})
    n = len(stack(run(make_suite(cfg, s, 1), hover(), 100.0), "gps_pos"))
    assert n == pytest.approx(0.7 * 10000, rel=0.05)


# --- lidar ---------------------------------------------------------------------------------


def test_lidar_reads_height_upright(cfg):
    samples = run(make_suite(cfg, quiet()), hover(10.0), 0.1)
    assert stack(samples, "lidar_range")[-1] == pytest.approx(10.0, abs=1e-9)


def test_lidar_slant_range_when_tilted(cfg):
    q = quat_from_axis_angle(np.array([1.0, 0, 0]), np.radians(30))
    x = make_state([0, 0, 10], [0, 0, 0], q, [0, 0, 0], 20.0)
    samples = run(make_suite(cfg, quiet()), x, 0.1)
    assert stack(samples, "lidar_range")[-1] == pytest.approx(
        10.0 / np.cos(np.radians(30)), rel=1e-9
    )


def test_lidar_none_out_of_range(cfg):
    assert len(stack(run(make_suite(cfg, quiet()), hover(50.0), 0.2), "lidar_range")) == 0
    assert len(stack(run(make_suite(cfg, quiet()), hover(0.1), 0.2), "lidar_range")) == 0


def test_lidar_none_when_on_its_side_or_beyond_60_degrees(cfg):
    for deg, expect in ((90, False), (70, False), (50, True)):
        q = quat_from_axis_angle(np.array([0.0, 1, 0]), np.radians(deg))
        x = make_state([0, 0, 10], [0, 0, 0], q, [0, 0, 0], 20.0)
        got = len(stack(run(make_suite(cfg, quiet()), x, 0.1), "lidar_range")) > 0
        assert got is expect, deg


def test_lidar_noise_has_absolute_and_relative_parts(cfg):
    s = quiet(lidar={"sigma_abs": 0.02, "sigma_rel": 0.001, "rate": 100.0})
    r = stack(run(make_suite(cfg, s, 2), hover(20.0), 110.0), "lidar_range")
    assert r.std() == pytest.approx(0.02 + 0.001 * 20.0, rel=0.05)


# --- scheduling, latency, reproducibility --------------------------------------------------


def test_sample_counts_follow_rates(cfg):
    suite = make_suite(cfg, SensorsConfig())
    samples = run(suite, hover(), 10.0)
    for field, rate in (
        ("gyro", 400),
        ("accel", 400),
        ("baro_alt", 50),
        ("gps_pos", 10),
        ("lidar_range", 50),
    ):
        n = sum(1 for _, s in samples if getattr(s, field) is not None)
        # latency delays the last few samples out of the window
        assert abs(n - rate * 10) <= 1 + int(rate * 0.16), field


def test_sample_counts_exact_without_latency(cfg):
    samples = run(make_suite(cfg, quiet()), hover(), 10.0)
    for field, rate in (("gyro", 400), ("baro_alt", 50), ("gps_pos", 10), ("lidar_range", 50)):
        n = sum(1 for _, s in samples if getattr(s, field) is not None)
        assert abs(n - rate * 10) <= 1, field


def test_latency_delays_a_step_in_altitude(cfg):
    s = quiet(baro={"rate": 100.0, "latency": 0.05})
    suite = make_suite(cfg, s)
    samples = run(suite, None, 1.0, state_fn=lambda t: hover(10.0 if t < 0.5 else 20.0))
    first = [t for t, d in samples if d.baro_alt is not None and d.baro_alt > 15]
    assert first[0] >= 0.5 + 0.05 - 1e-9
    assert first[0] <= 0.5 + 0.05 + 0.011
    assert all(abs(d.t - t) < 1e-12 for t, d in samples)  # SensorData.t is delivery time


def test_same_seed_gives_identical_streams(cfg):
    a = run(make_suite(cfg, SensorsConfig(), 11), hover(), 2.0)
    b = run(make_suite(cfg, SensorsConfig(), 11), hover(), 2.0)
    c = run(make_suite(cfg, SensorsConfig(), 12), hover(), 2.0)
    assert np.array_equal(stack(a, "gyro"), stack(b, "gyro"))
    assert np.array_equal(stack(a, "gps_pos"), stack(b, "gps_pos"))
    assert not np.array_equal(stack(a, "gyro"), stack(c, "gyro"))


def test_changing_one_sensor_does_not_change_others(cfg):
    base = SensorsConfig()
    other = SensorsConfig.model_validate(
        {**base.model_dump(), "gps": {**base.gps.model_dump(), "dropout_prob": 0.5, "rate": 20.0}}
    )
    a = run(make_suite(cfg, base, 4), hover(), 2.0)
    b = run(make_suite(cfg, other, 4), hover(), 2.0)
    for f in ("gyro", "accel", "baro_alt", "lidar_range"):
        assert np.array_equal(stack(a, f), stack(b, f)), f
    assert not np.array_equal(stack(a, "gps_pos"), stack(b, "gps_pos"))


def test_needs_info_only_when_imu_is_due(cfg):
    suite = make_suite(cfg, quiet(imu={"rate": 100.0}))
    assert suite.needs_info(0.0)
    suite.sample(0.0, hover(), {"specific_force_B": np.zeros(3)})
    assert not suite.needs_info(0.001)
    assert suite.needs_info(0.010)


def test_default_config_values():
    c = SensorsConfig()
    assert c.imu.rate == 400.0 and c.baro.rate == 50.0 and c.gps.rate == 10.0
    assert c.gps.latency == 0.15 and c.lidar.max_range == 40.0 and c.lidar.min_range == 0.2
    assert c.imu.gyro_noise_density == pytest.approx(np.radians(0.005))
    assert c.imu.accel_range == pytest.approx(16 * G)
