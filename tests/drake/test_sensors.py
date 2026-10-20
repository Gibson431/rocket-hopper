"""Sensors on the Drake diagram: same models and noise streams as the legacy engine."""

import numpy as np
import pytest

from hopper.control.interface import OpenLoopController
from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.dynamics.vehicle import vehicle_derivative
from hopper.math3d import quat_error
from hopper.scenario_glue import run_hop
from hopper.sim.engine import Simulation
from hopper.sim.events import Outcome
from hopper.types import GIM, GIMR, PROP, THR, Command, make_state

OFF = Command(0.0, np.zeros(2), 0.0)
FIELDS = ("gyro", "accel", "baro_alt", "gps_pos", "gps_vel", "lidar_range")
G = 9.80665

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


def every_step(cfg):
    """Controller called on every physics step, so each call sees one step's delivery."""
    return cfg.model_copy(
        update={"sim": cfg.sim.model_copy(update={"controller_rate": 1.0 / cfg.sim.dt})}
    )


class Recorder:
    def __init__(self, command=OFF):
        self.calls = []
        self._command = command

    def reset(self):
        self.calls = []

    def step(self, sensors, t):
        self.calls.append((t, sensors))
        return self._command

    def step_state(self, est, t):
        raise AssertionError("sensor mode must use step")


def space():
    return Environment(gravity_enabled=False, aero_enabled=False)


def high_state(z=50.0):
    return make_state([0, 0, z], [0, 0, 0], None, [0, 0, 0], 20.0)


def drake_sim(cfg, ctl, env=None, x0=None, seed=0, **kw):
    return DrakeSimulation(
        cfg,
        ctl,
        env or space(),
        high_state() if x0 is None else x0,
        np.random.default_rng(seed),
        perfect_state=False,
        **kw,
    )


def test_closed_loop_sensors_deliver_at_their_rates_with_none_between(cfg):
    ctl = Recorder()
    drake_sim(cfg, ctl).run(2.0)
    assert len(ctl.calls) == 200
    got = dict.fromkeys(FIELDS, 0)
    for _, s in ctl.calls:
        for f in got:
            got[f] += getattr(s, f) is not None
    # 100 Hz controller: IMU every call, baro/lidar every other call, GPS every tenth
    assert got["gyro"] == got["accel"] == 199  # the first call precedes the 2 ms IMU latency
    assert abs(got["baro_alt"] - 100) <= 2 and abs(got["gps_pos"] - 20) <= 2
    assert got["gps_pos"] == got["gps_vel"]
    t_last, s_last = ctl.calls[-1]
    assert s_last.t == pytest.approx(t_last)
    assert ctl.calls[0][1].gps_pos is None  # 150 ms latency: nothing delivered yet


def test_closed_loop_free_fall_accelerometer_reads_only_its_bias(cfg):
    ctl = Recorder()
    drake_sim(cfg, ctl).run(1.0)
    acc = np.array([s.accel for _, s in ctl.calls[10:] if s.accel is not None])
    assert np.abs(acc.mean(axis=0)).max() < 0.2  # no gravity in specific force


def test_closed_loop_accelerometer_reads_g_up_for_a_vehicle_at_rest_on_the_pad(cfg):
    clean = noiseless(cfg)
    com = mass_properties(clean.tank.prop_mass_initial, clean).com_z
    x0 = make_state(
        [0, 0, com + clean.legs.length - 0.01],
        [0, 0, 0],
        None,
        [0, 0, 0],
        clean.tank.prop_mass_initial,
    )
    ctl = Recorder()
    drake_sim(clean, ctl, env=Environment.default(), x0=x0).run(1.0)
    acc = np.array([s.accel for _, s in ctl.calls[-20:]])
    assert np.isfinite(acc).all()
    assert np.allclose(acc.mean(axis=0), [0.0, 0.0, G], atol=0.05)


def test_closed_loop_specific_force_matches_the_legacy_model_during_a_burn(cfg):
    sim = DrakeSimulation(
        cfg,
        OpenLoopController(command=Command(0.6, np.array([0.02, -0.03]), 0.0)),
        Environment.default(),
        high_state(),
        np.random.default_rng(0),
        perfect_state=False,
    )
    sim.run(1.0)
    ctx = sim.context
    rig = sim.rig
    f_drake = rig.sensors.specific_force(rig.sensors.GetMyContextFromRoot(ctx))
    from hopper.drake.state import to_legacy_state

    x = to_legacy_state(rig.plant, rig.plant.GetMyContextFromRoot(ctx))
    es = rig.engine.GetMyContextFromRoot(ctx).get_continuous_state_vector().CopyToVector()
    x[PROP], x[THR], x[GIM], x[GIMR] = es[0], es[1], es[2:4], es[4:6]
    cmd = rig.host.get_command_output_port().Eval(rig.host.GetMyContextFromRoot(ctx))
    applied = Command(cmd[0], cmd[1:3], cmd[3])
    info = {}
    vehicle_derivative(1.0, x, applied, cfg, Environment.default(), info=info)
    assert np.linalg.norm(info["specific_force_B"]) > 5.0  # thrusting, not a trivial zero
    assert np.allclose(f_drake, info["specific_force_B"], atol=1e-3)


def test_closed_loop_same_seed_is_bit_identical_and_other_seed_differs(cfg):
    def run(seed):
        ctl = Recorder()
        drake_sim(cfg, ctl, seed=seed).run(0.5)
        return np.array([s.gyro for _, s in ctl.calls if s.gyro is not None])

    assert np.array_equal(run(3), run(3))
    assert not np.array_equal(run(3), run(4))


def test_closed_loop_noise_stream_per_step_matches_the_legacy_engine(cfg):
    """At rest in space every field is a pure noise stream, so with a controller call on every
    step the Drake delivery equals the legacy one step for step: same seed, same spawn order, same
    latency (Drake runs the sensors' unrestricted update before the host's discrete update)."""
    cfg1 = every_step(cfg)
    leg, dr = Recorder(), Recorder()
    Simulation(cfg1, leg, space(), high_state(), np.random.default_rng(5), perfect_state=False).run(
        0.4
    )
    drake_sim(cfg1, dr, seed=5).run(0.4)
    n = 0
    for (_, a), (_, b) in zip(leg.calls, dr.calls, strict=True):
        for f in FIELDS:
            x, y = getattr(a, f), getattr(b, f)
            assert (x is None) == (y is None)
            if x is not None:
                n += 1
                assert np.allclose(x, y, atol=1e-9, rtol=0)
    assert n > 300  # imu 2 fields x ~320/2 + baro + gps


def test_closed_loop_ekf_hop_lands_with_estimation_errors_in_the_s10_bounds(cfg):
    res, s = run_hop(cfg, seed=0, perfect_state=False, backend="drake")
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 1.0
    assert s.touchdown_speed < 1.5
    assert np.degrees(s.touchdown_tilt) < 5.0
    df = res.log.dropna(subset=["est_pos_x"])
    pos = (
        df[[f"pos_{a}" for a in "xyz"]].to_numpy() - df[[f"est_pos_{a}" for a in "xyz"]].to_numpy()
    )
    vel = (
        df[[f"vel_{a}" for a in "xyz"]].to_numpy() - df[[f"est_vel_{a}" for a in "xyz"]].to_numpy()
    )
    rms = lambda a: float(np.sqrt((a**2).sum(axis=1).mean()))  # noqa: E731
    assert rms(pos) < 0.5
    assert rms(vel) < 0.1
    q = df[[f"quat_{a}" for a in "wxyz"]].to_numpy()
    qe = df[[f"est_quat_{a}" for a in "wxyz"]].to_numpy()
    att = np.array([quat_error(a, b) for a, b in zip(q, qe, strict=True)])
    assert np.degrees(rms(att)) < 0.5
    assert np.abs(df["mass"] - df["est_mass"]).max() < 0.5
    assert sum(c.startswith("sig_") for c in res.log.columns) == 16


def test_closed_loop_a_second_sensor_run_repeats_the_noise_stream(cfg):
    ctl = Recorder()
    sim = drake_sim(cfg, ctl, seed=2)
    sim.run(0.3)
    first = np.array([s.gyro for _, s in ctl.calls if s.gyro is not None])
    sim.run(0.3)
    second = np.array([s.gyro for _, s in ctl.calls if s.gyro is not None])
    assert len(first) > 10 and np.array_equal(first, second)
