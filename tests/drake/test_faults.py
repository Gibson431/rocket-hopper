"""The S11 faults through the Drake diagram: same seams and semantics as `hopper.sim.faults`."""

import numpy as np
import pytest
from hopper_control.interface import OpenLoopController
from hopper_core.types import Command, make_state

from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.sim.faults import (
    EngineOut,
    GimbalSlow,
    GimbalStuck,
    GpsOutage,
    GyroBias,
    GyroBiasStep,
    ThrustScale,
)


def _sim(cfg, faults=None, throttle=0.5, gimbal=(0.0, 0.0), controller=None, **kw):
    com = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z
    x0 = make_state([0, 0, com + cfg.legs.length + 50.0], [0, 0, 0], None, [0, 0, 0], 20.0)
    ctrl = controller or OpenLoopController(command=Command(throttle, np.array(gimbal), 0.0))
    if faults is not None:
        kw["faults"] = faults
    return DrakeSimulation(cfg, ctrl, Environment.default(), x0, np.random.default_rng(0), **kw)


def _at(log, t):
    return log.iloc[(log["t"] - t).abs().argmin()]


def test_gimbal_stuck_holds_the_actual_gimbal_from_the_trigger_time(cfg):
    value = np.radians(2.0)
    res = _sim(cfg, [GimbalStuck(axis="y", at=0.5, value_deg=2.0)]).run(2.0)
    assert _at(res.log, 0.4).gim_y == pytest.approx(0.0, abs=1e-9)
    assert _at(res.log, 1.9).gim_y == pytest.approx(value, abs=1e-4)
    assert _at(res.log, 1.9).gim_x == pytest.approx(0.0, abs=1e-9)  # the other axis is untouched


def test_gimbal_stuck_ignores_later_commands_and_the_command_delay(cfg):
    res = _sim(
        cfg,
        [GimbalStuck(axis="x", at=0.0, value_deg=-1.0)],
        gimbal=(0.05, 0.0),
        command_delay=0.2,
    ).run(2.0)
    log = res.log
    assert log.cmd_gim_x.iloc[0] == pytest.approx(np.radians(-1.0))  # even before the delay ends
    assert _at(log, 1.9).gim_x == pytest.approx(np.radians(-1.0), abs=1e-4)
    assert np.allclose(log.cmd_gim_x, np.radians(-1.0))


def test_thrust_scale_scales_thrust_by_the_factor_after_the_trigger(cfg):
    base = _sim(cfg).run(2.0).log
    res = _sim(cfg, [ThrustScale(at=0.5, factor=0.8)]).run(2.0).log
    assert _at(res, 0.4).thrust == pytest.approx(_at(base, 0.4).thrust, rel=1e-12)
    assert _at(res, 1.9).thrust == pytest.approx(0.8 * _at(base, 1.9).thrust, rel=1e-6)


def test_thrust_scale_switches_on_the_step_of_the_trigger_time(cfg):
    base = _sim(cfg).run(1.0).log
    res = _sim(cfg, [ThrustScale(at=0.5, factor=0.5)]).run(1.0).log
    i = int(round(0.5 / cfg.sim.dt))
    assert res.thrust.iloc[i - 1] == base.thrust.iloc[i - 1]
    assert res.thrust.iloc[i] == pytest.approx(0.5 * base.thrust.iloc[i], rel=1e-9)


def test_engine_out_zeroes_the_thrust(cfg):
    res = _sim(cfg, [EngineOut(at=0.5)]).run(2.0).log
    assert _at(res, 0.4).thrust > 700.0
    assert _at(res, 1.9).thrust == pytest.approx(0.0, abs=1.0)


def test_gimbal_slow_limits_the_gimbal_rate(cfg):
    fast = _sim(cfg, gimbal=(0.0, 0.1)).run(0.6).log
    slow = _sim(cfg, [GimbalSlow(at=0.0, rate_factor=0.3)], gimbal=(0.0, 0.1)).run(0.6).log
    assert fast.gimr_y.abs().max() > 0.9 * cfg.engine.gimbal_rate_max
    assert slow.gimr_y.abs().max() == pytest.approx(0.3 * cfg.engine.gimbal_rate_max, rel=0.02)


def test_faults_compose(cfg):
    res = _sim(
        cfg, [ThrustScale(at=0.2, factor=0.5), GimbalStuck(axis="y", at=0.2, value_deg=1.0)]
    ).run(1.5)
    base = _sim(cfg).run(1.5).log
    assert _at(res.log, 1.4).thrust == pytest.approx(0.5 * _at(base, 1.4).thrust, rel=1e-6)
    assert _at(res.log, 1.4).gim_y == pytest.approx(np.radians(1.0), abs=1e-4)


def test_a_second_run_starts_from_the_nominal_parameters(cfg):
    sim = _sim(cfg, [ThrustScale(at=0.0, factor=0.5)])
    first = sim.run(0.5).log
    assert first.thrust.iloc[-1] > 0
    second = sim.run(0.5).log
    assert second.thrust.iloc[-1] == pytest.approx(first.thrust.iloc[-1], rel=1e-9)


class SensorSpy(OpenLoopController):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seen: list = []

    def step(self, sensors, t):
        self.seen.append((t, sensors))
        return super().step(sensors, t)


def _spy_run(cfg, faults, t_end):
    spy = SensorSpy(command=Command(0.5, np.zeros(2), 0.0))
    _sim(cfg, faults, controller=spy, perfect_state=False).run(t_end)
    return spy.seen


def test_gps_outage_reaches_the_controller_as_none_fields(cfg):
    seen = _spy_run(cfg, [GpsOutage(start=1.0, duration=1.0)], 3.0)
    assert any(s.gps_pos is not None for t, s in seen if t < 1.0)
    assert all(s.gps_pos is None and s.gps_vel is None for t, s in seen if 1.0 <= t < 2.0)
    assert any(s.gps_pos is not None for t, s in seen if t > 2.2)


def test_gyro_bias_step_offsets_the_gyro_from_the_trigger(cfg):
    base = _spy_run(cfg, [], 1.0)
    biased = _spy_run(cfg, [GyroBiasStep(at=0.5, value_dps=2.0, axis="x")], 1.0)
    for (t, a), (_, b) in zip(base, biased, strict=True):
        if a.gyro is None or b.gyro is None:
            continue
        d = b.gyro - a.gyro
        # the rate gyro is the same stream; only the injected offset differs, once it applies
        if t < 0.49:
            assert np.allclose(d, 0.0, atol=1e-12)
        elif t > 0.51:
            assert np.allclose(d, [np.radians(2.0), 0, 0], atol=1e-6)


def test_gyro_bias_fault_adds_a_constant_vector(cfg):
    base = _spy_run(cfg, [], 0.5)
    biased = _spy_run(cfg, [GyroBias(bias=(0.001, -0.002, 0.003))], 0.5)
    for (_, a), (_, b) in zip(base[1:], biased[1:], strict=True):
        assert np.allclose(b.gyro - a.gyro, [0.001, -0.002, 0.003], atol=1e-9)


def _inert_faults():
    return [EngineOut(at=1e9), ThrustScale(at=0.0, factor=1.0), GpsOutage(start=1e9, duration=1.0)]


def test_no_faults_builds_no_fault_systems_and_matches_the_run_without_the_argument(cfg):
    plain = _sim(cfg)
    empty = _sim(cfg, [])
    assert len(empty.rig.diagram.GetSystems()) == len(plain.rig.diagram.GetSystems())
    assert plain.run(1.0).log.equals(empty.run(1.0).log)


def test_inert_faults_leave_the_run_bit_identical(cfg):
    a = _sim(cfg).run(1.0).log
    b = _sim(cfg, _inert_faults()).run(1.0).log
    assert a.equals(b)
