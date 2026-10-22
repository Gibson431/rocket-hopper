import numpy as np
from hopper_control.interface import (
    Controller,
    OpenLoopController,
    StateFeedbackController,
)
from hopper_core.types import Command, SensorData, StateEstimate


def est(t):
    z = np.zeros(3)
    return StateEstimate(t, z, z, np.array([1.0, 0, 0, 0]), z, 100.0)


def test_open_loop_controller_satisfies_both_protocols():
    c = OpenLoopController()
    assert isinstance(c, Controller)
    assert isinstance(c, StateFeedbackController)


def test_default_is_engine_off():
    cmd = OpenLoopController().step_state(est(0.0), 0.0)
    assert cmd.throttle == 0.0
    assert np.all(cmd.gimbal == 0.0)
    assert cmd.roll_torque == 0.0


def test_constant_command_is_returned_every_time():
    fixed = Command(0.6, np.array([0.01, -0.02]), 0.3)
    c = OpenLoopController(fixed)
    assert c.step_state(est(0.0), 0.0) is fixed
    assert c.step_state(est(9.0), 9.0) is fixed


def test_schedule_returns_latest_entry_not_after_t():
    off = Command(0.0, np.zeros(2), 0.0)
    a = Command(0.5, np.zeros(2), 0.0)
    b = Command(0.8, np.zeros(2), 0.0)
    c = OpenLoopController(schedule=[(1.0, a), (2.0, b)], default=off)
    assert c.step_state(est(0.5), 0.5) is off
    assert c.step_state(est(1.0), 1.0) is a
    assert c.step_state(est(1.99), 1.99) is a
    assert c.step_state(est(2.0), 2.0) is b


def test_sensor_path_uses_the_same_schedule():
    a = Command(0.5, np.zeros(2), 0.0)
    c = OpenLoopController(schedule=[(1.0, a)])
    s = SensorData(1.5, None, None, None, None, None, None)
    assert c.step(s, 1.5) is a
