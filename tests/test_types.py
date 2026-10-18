import dataclasses

import numpy as np
import pytest

from hopper import types as ht


def test_state_index_constants_match_conventions():
    assert ht.N_STATE == 19
    assert ht.POS == slice(0, 3)
    assert ht.VEL == slice(3, 6)
    assert ht.QUAT == slice(6, 10)
    assert ht.OMEGA == slice(10, 13)
    assert ht.PROP == 13
    assert ht.THR == 14
    assert ht.GIM == slice(15, 17)
    assert ht.GIMR == slice(17, 19)


def test_make_state_places_each_field_in_its_slice():
    x = ht.make_state(
        pos=[1, 2, 3],
        vel=[4, 5, 6],
        quat=[0, 0, 0, 1],
        omega=[7, 8, 9],
        prop=20.0,
        thr=0.5,
        gim=(0.1, 0.2),
        gimr=(0.3, 0.4),
    )
    assert x.shape == (19,)
    assert x.dtype == np.float64
    assert x[ht.POS] == pytest.approx([1, 2, 3])
    assert x[ht.VEL] == pytest.approx([4, 5, 6])
    assert x[ht.QUAT] == pytest.approx([0, 0, 0, 1])
    assert x[ht.OMEGA] == pytest.approx([7, 8, 9])
    assert x[ht.PROP] == 20.0
    assert x[ht.THR] == 0.5
    assert x[ht.GIM] == pytest.approx([0.1, 0.2])
    assert x[ht.GIMR] == pytest.approx([0.3, 0.4])


def test_make_state_defaults_to_upright_attitude_and_idle_engine():
    x = ht.make_state([0, 0, 1], [0, 0, 0], None, [0, 0, 0], 5.0)
    assert x[ht.QUAT] == pytest.approx([1, 0, 0, 0])
    assert x[ht.THR] == 0.0
    assert x[ht.GIM] == pytest.approx([0, 0])
    assert x[ht.GIMR] == pytest.approx([0, 0])


def test_make_state_normalises_the_quaternion():
    x = ht.make_state([0, 0, 0], [0, 0, 0], [2, 0, 0, 0], [0, 0, 0], 1.0)
    assert x[ht.QUAT] == pytest.approx([1, 0, 0, 0])


def test_make_state_rejects_wrong_shapes():
    with pytest.raises(ValueError):
        ht.make_state([0, 0], [0, 0, 0], None, [0, 0, 0], 1.0)


def test_command_accepts_valid_gimbal_and_converts_to_array():
    c = ht.Command(throttle=0.5, gimbal=[0.01, -0.02], roll_torque=0.0)
    assert isinstance(c.gimbal, np.ndarray)
    assert c.gimbal.shape == (2,)


@pytest.mark.parametrize("gimbal", [[0.0], [0.0, 0.0, 0.0], [[0.0, 0.0]]])
def test_command_rejects_wrong_gimbal_shape(gimbal):
    with pytest.raises(ValueError):
        ht.Command(throttle=0.5, gimbal=np.array(gimbal), roll_torque=0.0)


def test_interface_dataclasses_are_frozen():
    c = ht.Command(throttle=0.5, gimbal=np.zeros(2), roll_torque=0.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        c.throttle = 1.0
    s = ht.SensorData(
        t=0.0, gyro=None, accel=None, baro_alt=None, gps_pos=None, gps_vel=None, lidar_range=None
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.t = 1.0
    e = ht.StateEstimate(
        t=0.0, pos=np.zeros(3), vel=np.zeros(3), quat=np.array([1.0, 0, 0, 0]),
        omega=np.zeros(3), mass=100.0,
    )  # fmt: skip
    with pytest.raises(dataclasses.FrozenInstanceError):
        e.mass = 1.0


def test_controller_protocols_are_structurally_satisfied():
    class Dummy:
        def reset(self) -> None: ...
        def step(self, sensors, t):
            return ht.Command(0.0, np.zeros(2), 0.0)

        def step_state(self, est, t):
            return ht.Command(0.0, np.zeros(2), 0.0)

    assert isinstance(Dummy(), ht.Controller)
    assert isinstance(Dummy(), ht.StateFeedbackController)
    assert not isinstance(object(), ht.Controller)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"throttle": float("nan"), "gimbal": [0.0, 0.0], "roll_torque": 0.0},
        {"throttle": 0.5, "gimbal": [float("inf"), 0.0], "roll_torque": 0.0},
        {"throttle": 0.5, "gimbal": [0.0, 0.0], "roll_torque": float("nan")},
    ],
)
def test_command_rejects_non_finite_values(kwargs):
    with pytest.raises(ValueError):
        ht.Command(**kwargs)
