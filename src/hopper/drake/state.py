"""Adapters between the `hopper_core.types` state vector (CoM, body-frame rate) and the Drake body.

Drake's body frame has its origin at the gimbal pivot; its state holds the origin position, the
origin velocity and the angular velocity in the world frame (docs/DRAKE_DESIGN.md Q2). The CoM
offset comes from the spatial inertia stored in the context, so no propellant argument is needed.
"""

import numpy as np
from hopper_core.math3d import cross, quat_to_dcm
from hopper_core.types import N_STATE, OMEGA, POS, PROP, QUAT, VEL
from pydrake.all import Quaternion, RigidTransform, SpatialVelocity
from pydrake.multibody.plant import MultibodyPlant

from hopper.config import VehicleConfig

BODY_NAME = "hopper"


def read_com_state(plant: MultibodyPlant, context, body):
    """(position, velocity) of the CoM in the world, from the current spatial inertia."""
    s = plant.GetPositionsAndVelocities(context)
    q, p_o, w_i, v_o = s[:4], s[4:7], s[7:10], s[10:13]
    r = quat_to_dcm(q) @ body.CalcSpatialInertiaInBodyFrame(context).get_com()
    return p_o + r, v_o + cross(w_i, r)


def write_com_state(plant: MultibodyPlant, context, body, p_com, v_com) -> None:
    """Rewrite the origin pose and velocity so the CoM has the given position and velocity.
    Uses the CoM offset of the spatial inertia currently in the context."""
    s = plant.GetPositionsAndVelocities(context)
    q, w_i = s[:4], s[7:10]
    r = quat_to_dcm(q) @ body.CalcSpatialInertiaInBodyFrame(context).get_com()
    plant.SetFreeBodyPose(context, body, RigidTransform(Quaternion(q), p_com - r))
    plant.SetFreeBodySpatialVelocity(context, body, SpatialVelocity(w_i, v_com - cross(w_i, r)))


def to_state_vector(plant: MultibodyPlant, context) -> np.ndarray:
    """State vector with pos, vel, quat, omega filled; PROP, THR, GIM, GIMR (other systems): 0."""
    body = plant.GetBodyByName(BODY_NAME)
    s = plant.GetPositionsAndVelocities(context)
    p_com, v_com = read_com_state(plant, context, body)
    x = np.zeros(N_STATE)
    x[POS], x[VEL] = p_com, v_com
    x[QUAT] = s[:4] / np.linalg.norm(s[:4])
    x[OMEGA] = quat_to_dcm(x[QUAT]).T @ s[7:10]
    return x


def set_state_vector(plant: MultibodyPlant, context, x: np.ndarray, cfg: VehicleConfig) -> None:
    """Set the body from a state vector; the inertia is set from x[PROP] (mass model of `cfg`)."""
    from hopper.drake.mass import spatial_inertia

    body = plant.GetBodyByName(BODY_NAME)
    body.SetSpatialInertiaInBodyFrame(context, spatial_inertia(x[PROP], cfg))
    q = np.asarray(x[QUAT], dtype=float)
    q = q / np.linalg.norm(q)
    r = quat_to_dcm(q) @ body.CalcSpatialInertiaInBodyFrame(context).get_com()
    w_i = quat_to_dcm(q) @ x[OMEGA]
    plant.SetFreeBodyPose(context, body, RigidTransform(Quaternion(q), x[POS] - r))
    plant.SetFreeBodySpatialVelocity(context, body, SpatialVelocity(w_i, x[VEL] - cross(w_i, r)))
