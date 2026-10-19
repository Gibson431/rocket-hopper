"""Q2: floating body frame at the gimbal pivot; verify QuaternionFloatingJoint conventions."""
import numpy as np
from pydrake.all import (DiagramBuilder, AddMultibodyPlantSceneGraph, SpatialInertia, RigidBody,
                         RotationMatrix, Quaternion, RollPitchYaw, MultibodyPlant)

plant = MultibodyPlant(0.0)
M, COM, I = 100.0, np.array([0, 0, 0.85]), np.diag([30.0, 30.0, 3.0])
si = SpatialInertia.MakeFromCentralInertia(M, COM, __import__("pydrake.all", fromlist=["RotationalInertia"]).RotationalInertia(30.0, 30.0, 3.0))
body = plant.AddRigidBody("hopper", si)
plant.Finalize()
ctx = plant.CreateDefaultContext()
print("positions", plant.num_positions(), "velocities", plant.num_velocities())
print("joint", [plant.get_joint(j).name() for j in plant.GetJointIndices()])
j = plant.get_joint(plant.GetJointIndices()[0]) if plant.num_joints() else None
print(type(j))
# Set pose + velocity and read back
q = np.array([0.9, 0.1, -0.2, 0.3]); q /= np.linalg.norm(q)
X = __import__("pydrake.all", fromlist=["RigidTransform"]).RigidTransform(Quaternion(q), [1, 2, 3])
plant.SetFreeBodyPose(ctx, body, X)
w_B = np.array([0.1, 0.2, 0.3]); v_I = np.array([1.0, -1.0, 0.5])
from pydrake.all import SpatialVelocity
# SetFreeBodySpatialVelocity: V_WB expressed in W, at body origin
R = RotationMatrix(Quaternion(q))
plant.SetFreeBodySpatialVelocity(ctx, body, SpatialVelocity(R.matrix() @ w_B, v_I))
print("state x:", plant.GetPositionsAndVelocities(ctx))
print("expect q", q, "p", [1,2,3], "w_I", R.matrix()@w_B, "v", v_I)
print("R w_B =", R.matrix() @ w_B, " (compare v[0:3]) ; if v[0:3]==w_B then joint is in child frame")
# Is v translational the body-origin velocity in W?  Then v_I at origin. Body origin is the pivot, not CoM.
# Verify body-frame (child) vs parent frame by direct generalized velocity read
x = plant.GetPositionsAndVelocities(ctx)
print("q", x[:7]); print("v", x[7:])
# qdot via MapVelocityToQDot
qdot = plant.MapVelocityToQDot(ctx, x[7:])
# legacy quat_derivative: 0.5 * q ⊗ [0, w_B]
import sys; sys.path.insert(0, "src")
from hopper.math3d import quat_derivative
print("drake qdot", qdot[:4]); print("legacy   ", quat_derivative(q, w_B))
