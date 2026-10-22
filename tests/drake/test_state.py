import numpy as np
import pytest
from hopper_core.math3d import quat_derivative, quat_to_dcm
from hopper_core.types import OMEGA, POS, QUAT, VEL, make_state

from hopper.drake.plant import build_vehicle_plant
from hopper.drake.state import set_state_vector, to_state_vector
from hopper.dynamics.mass_model import mass_properties


@pytest.fixture
def plant(cfg):
    plant, _ = build_vehicle_plant(cfg)
    return plant


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("prop", [20.0, 7.0, 0.0])
def test_state_vector_round_trips(cfg, plant, seed, prop):
    rng = np.random.default_rng(seed)
    q = rng.normal(size=4)
    x = make_state(
        pos=rng.normal(size=3) * 10,
        vel=rng.normal(size=3) * 5,
        quat=q,
        omega=rng.normal(size=3),
        prop=prop,
    )
    ctx = plant.CreateDefaultContext()
    set_state_vector(plant, ctx, x, cfg)
    y = to_state_vector(plant, ctx)
    for sl in (POS, VEL, OMEGA):
        np.testing.assert_allclose(y[sl], x[sl], atol=1e-12)
    # Drake stores w >= 0; q and -q are the same rotation
    np.testing.assert_allclose(y[QUAT], x[QUAT] * np.sign(x[QUAT][0]), atol=1e-12)


def test_drake_body_origin_is_the_gimbal_pivot_below_the_com(cfg, plant):
    x = make_state(
        pos=[1, 2, 3], vel=[0, 0, 0], quat=[0.9, 0.1, -0.2, 0.3], omega=[0, 0, 0], prop=15.0
    )
    ctx = plant.CreateDefaultContext()
    set_state_vector(plant, ctx, x, cfg)
    body = plant.GetBodyByName("hopper")
    pose = plant.EvalBodyPoseInWorld(ctx, body)
    r = quat_to_dcm(x[QUAT]) @ [0.0, 0.0, mass_properties(15.0, cfg).com_z]
    np.testing.assert_allclose(pose.translation(), x[POS] - r, atol=1e-12)
    np.testing.assert_allclose(pose.rotation().matrix(), quat_to_dcm(x[QUAT]), atol=1e-12)


def test_body_rate_is_expressed_in_the_body_frame(cfg, plant):
    # spin about the body z axis of a tilted vehicle: the inertial rate must be R @ omega_B,
    # and Drake's quaternion derivative must equal the body-frame kinematic one.
    x = make_state(
        pos=[0, 0, 0], vel=[0, 0, 0], quat=[0.9, 0.1, -0.2, 0.3], omega=[0.1, 0.2, 0.3], prop=5.0
    )
    ctx = plant.CreateDefaultContext()
    set_state_vector(plant, ctx, x, cfg)
    s = plant.GetPositionsAndVelocities(ctx)
    np.testing.assert_allclose(s[7:10], quat_to_dcm(x[QUAT]) @ x[OMEGA], atol=1e-12)
    qdot = plant.MapVelocityToQDot(ctx, s[7:])
    np.testing.assert_allclose(qdot[:4], quat_derivative(x[QUAT], x[OMEGA]), atol=1e-12)


def test_adapters_leave_actuator_slots_to_other_systems(cfg, plant):
    ctx = plant.CreateDefaultContext()
    y = to_state_vector(plant, ctx)
    assert y.shape == (19,)
    assert not np.any(y[13:])
