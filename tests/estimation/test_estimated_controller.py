import numpy as np
import pytest
from estimation_harness import Feeder

from hopper.estimated_controller import EstimatedController
from hopper.scenario_glue import ekf_params_from_config
from hopper.types import Command, SensorData, StateEstimate


class StubMission:
    state = None

    def __init__(self):
        self.seen: list[StateEstimate] = []
        self.resets = 0

    def reset(self):
        self.resets += 1

    def step_state(self, est, t):
        self.seen.append(est)
        return Command(0.6, np.array([0.01, -0.02]), 0.1)


def test_engine_stays_off_until_the_filter_is_aligned_then_the_mission_flies_the_estimate(cfg):
    stub = StubMission()
    ctl = EstimatedController(ekf_params_from_config(cfg), stub)
    outs = []
    for t, data, _ in Feeder(cfg, 0):
        if t > 1.5:
            break
        outs.append((t, ctl.step(data, t)))
    off = [c for t, c in outs if t < 0.9]
    assert all(c.throttle == 0.0 for c in off)
    assert len(stub.seen) > 0
    assert outs[-1][1].throttle == 0.6
    assert isinstance(stub.seen[-1], StateEstimate)


def test_controller_feeds_its_own_throttle_back_so_the_mass_estimate_falls(cfg):
    stub = StubMission()
    ctl = EstimatedController(ekf_params_from_config(cfg), stub)
    for t, data, _ in Feeder(cfg, 0):
        if t > 6.0:
            break
        ctl.step(data, t)
    assert stub.seen[-1].mass < stub.seen[0].mass


def test_reset_resets_the_filter_and_the_mission(cfg):
    stub = StubMission()
    ctl = EstimatedController(ekf_params_from_config(cfg), stub)
    for t, data, _ in Feeder(cfg, 0):
        if t > 1.5:
            break
        ctl.step(data, t)
    ctl.reset()
    assert stub.resets == 1
    empty = SensorData(0.0, None, None, None, None, None, None)
    assert ctl.step(empty, 0.0).throttle == 0.0
    assert not ctl.ekf.initialised


def test_step_state_is_not_supported_because_the_controller_needs_raw_sensors(cfg):
    ctl = EstimatedController(ekf_params_from_config(cfg), StubMission())
    with pytest.raises(NotImplementedError):
        ctl.step_state(None, 0.0)


def test_log_hook_exposes_nan_before_alignment_and_finite_values_after(cfg):
    ctl = EstimatedController(ekf_params_from_config(cfg), StubMission())
    assert len(ctl.log_values()) == len(ctl.log_columns)
    assert np.isnan(ctl.log_values()).all()
    for t, data, _ in Feeder(cfg, 0):
        if t > 1.5:
            break
        ctl.step(data, t)
    assert np.isfinite(ctl.log_values()).all()
