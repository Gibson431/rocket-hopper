"""Faults are time-triggered modifiers at the engine's command, parameter and sensor seams."""

import numpy as np
import pytest
from pydantic import ValidationError

from hopper.control.interface import OpenLoopController
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.sim.engine import Simulation
from hopper.sim.faults import (
    EngineOut,
    GimbalSlow,
    GimbalStuck,
    GpsOutage,
    GyroBias,
    GyroBiasStep,
    ThrustScale,
)
from hopper.types import Command, SensorData, make_state


def _sim(cfg, faults=(), throttle=0.5, gimbal=(0.0, 0.0), perfect_state=True):
    com = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z
    x0 = make_state([0, 0, com + cfg.legs.length + 50.0], [0, 0, 0], None, [0, 0, 0], 20.0)
    ctrl = OpenLoopController(command=Command(throttle, np.array(gimbal), 0.0))
    return Simulation(
        cfg,
        ctrl,
        Environment.default(),
        x0,
        np.random.default_rng(0),
        perfect_state=perfect_state,
        faults=faults,
    )


def _at(log, t):
    return log.iloc[(log["t"] - t).abs().argmin()]


def test_unknown_fault_key_is_rejected():
    with pytest.raises(ValidationError):
        ThrustScale(at=1.0, factor=0.8, bogus=1)


def test_gimbal_stuck_holds_the_actual_gimbal_from_the_trigger_time(cfg):
    value = np.radians(2.0)
    res = _sim(cfg, [GimbalStuck(axis="y", at=0.5, value_deg=2.0)]).run(2.0)
    before, after = _at(res.log, 0.4), _at(res.log, 1.9)
    assert before.gim_y == pytest.approx(0.0, abs=1e-9)
    assert after.gim_y == pytest.approx(value, abs=1e-4)
    assert after.gim_x == pytest.approx(0.0, abs=1e-9)  # the other axis is untouched


def test_gimbal_stuck_ignores_later_commands(cfg):
    res = _sim(cfg, [GimbalStuck(axis="x", at=0.0, value_deg=-1.0)], gimbal=(0.05, 0.0)).run(2.0)
    assert _at(res.log, 1.9).gim_x == pytest.approx(np.radians(-1.0), abs=1e-4)


def test_thrust_scale_scales_thrust_by_the_factor_after_the_trigger(cfg):
    base = _sim(cfg).run(2.0).log
    res = _sim(cfg, [ThrustScale(at=0.5, factor=0.8)]).run(2.0).log
    assert _at(res, 0.4).thrust == pytest.approx(_at(base, 0.4).thrust, rel=1e-12)
    assert _at(res, 1.9).thrust == pytest.approx(0.8 * _at(base, 1.9).thrust, rel=1e-6)


def test_engine_out_zeroes_the_thrust(cfg):
    res = _sim(cfg, [EngineOut(at=0.5)]).run(2.0).log
    assert _at(res, 0.4).thrust > 700.0
    assert _at(res, 1.9).thrust == pytest.approx(0.0, abs=1.0)


def test_gimbal_slow_limits_the_gimbal_rate(cfg):
    fast = _sim(cfg, gimbal=(0.0, 0.1)).run(0.6).log
    slow = _sim(cfg, [GimbalSlow(at=0.0, rate_factor=0.3)], gimbal=(0.0, 0.1)).run(0.6).log
    assert fast.gimr_y.abs().max() > 0.9 * cfg.engine.gimbal_rate_max
    assert slow.gimr_y.abs().max() == pytest.approx(0.3 * cfg.engine.gimbal_rate_max, rel=0.02)


def _data(**kw):
    base = dict(
        gyro=np.zeros(3),
        accel=np.zeros(3),
        baro_alt=1.0,
        gps_pos=np.ones(3),
        gps_vel=np.ones(3),
        lidar_range=2.0,
    )
    return SensorData(t=0.0, **{**base, **kw})


def test_gps_outage_blanks_gps_fields_only_inside_the_window():
    f = GpsOutage(start=4.0, duration=5.0)
    inside = f.modify_sensors(6.0, _data())
    assert inside.gps_pos is None and inside.gps_vel is None
    assert inside.baro_alt == 1.0 and inside.gyro is not None
    for t in (3.99, 9.0, 12.0):
        out = f.modify_sensors(t, _data())
        assert out.gps_pos is not None and out.gps_vel is not None


def test_gps_outage_reaches_the_controller_as_none_fields(cfg):
    seen = []

    class Spy(OpenLoopController):
        def step(self, sensors, t):
            seen.append((t, sensors.gps_pos))
            return super().step(sensors, t)

    sim = _sim(cfg, [GpsOutage(start=1.0, duration=1.0)], perfect_state=False)
    sim.controller = Spy(command=Command(0.5, np.zeros(2), 0.0))
    sim.run(3.0)
    assert any(p is not None for t, p in seen if t < 1.0)
    assert all(p is None for t, p in seen if 1.0 <= t < 2.0)
    assert any(p is not None for t, p in seen if t > 2.2)


def test_gyro_bias_step_adds_the_bias_after_the_trigger_only():
    f = GyroBiasStep(at=6.0, value_dps=0.5)
    assert np.allclose(f.modify_sensors(5.9, _data()).gyro, 0.0)
    assert np.allclose(f.modify_sensors(6.0, _data()).gyro, np.radians(0.5))
    assert f.modify_sensors(7.0, _data(gyro=None)).gyro is None


def test_gyro_bias_step_can_target_one_axis():
    g = GyroBiasStep(at=0.0, value_dps=1.0, axis="y").modify_sensors(1.0, _data()).gyro
    assert np.allclose(g, [0.0, np.radians(1.0), 0.0])


def test_faults_compose(cfg):
    res = _sim(
        cfg, [ThrustScale(at=0.2, factor=0.5), GimbalStuck(axis="y", at=0.2, value_deg=1.0)]
    ).run(1.5)
    base = _sim(cfg).run(1.5).log
    assert _at(res.log, 1.4).thrust == pytest.approx(0.5 * _at(base, 1.4).thrust, rel=1e-6)
    assert _at(res.log, 1.4).gim_y == pytest.approx(np.radians(1.0), abs=1e-4)


def test_no_faults_is_bit_identical_to_the_default_engine(cfg):
    a = _sim(cfg).run(1.0).log
    com = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z
    x0 = make_state([0, 0, com + cfg.legs.length + 50.0], [0, 0, 0], None, [0, 0, 0], 20.0)
    b = Simulation(
        cfg,
        OpenLoopController(command=Command(0.5, np.zeros(2), 0.0)),
        Environment.default(),
        x0,
        np.random.default_rng(0),
    ).run(1.0)
    assert a.equals(b.log)


def test_gyro_bias_adds_the_constant_vector_to_every_gyro_sample():
    f = GyroBias(bias=(0.001, -0.002, 0.003))
    assert np.allclose(f.modify_sensors(0.0, _data()).gyro, [0.001, -0.002, 0.003])
    assert f.modify_sensors(0.0, _data(gyro=None)).gyro is None
