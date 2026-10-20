"""Fault models: validation and the pure sensor-side modifiers. Their effect on a run is tested
in `tests/drake/test_faults.py`."""

import numpy as np
import pytest
from pydantic import ValidationError

from hopper.sim.faults import GpsOutage, GyroBias, GyroBiasStep, ThrustScale
from hopper.types import SensorData


def test_unknown_fault_key_is_rejected():
    with pytest.raises(ValidationError):
        ThrustScale(at=1.0, factor=0.8, bogus=1)


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


def test_gyro_bias_step_adds_the_bias_after_the_trigger_only():
    f = GyroBiasStep(at=6.0, value_dps=0.5)
    assert np.allclose(f.modify_sensors(5.9, _data()).gyro, 0.0)
    assert np.allclose(f.modify_sensors(6.0, _data()).gyro, np.radians(0.5))
    assert f.modify_sensors(7.0, _data(gyro=None)).gyro is None


def test_gyro_bias_step_can_target_one_axis():
    g = GyroBiasStep(at=0.0, value_dps=1.0, axis="y").modify_sensors(1.0, _data()).gyro
    assert np.allclose(g, [0.0, np.radians(1.0), 0.0])


def test_gyro_bias_adds_the_constant_vector_to_every_gyro_sample():
    f = GyroBias(bias=(0.001, -0.002, 0.003))
    assert np.allclose(f.modify_sensors(0.0, _data()).gyro, [0.001, -0.002, 0.003])
    assert f.modify_sensors(0.0, _data(gyro=None)).gyro is None
