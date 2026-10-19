"""Q4(b): prismatic leg joints with spring-damper (LinearSpringDamper) and a foot sphere in contact."""
import time
import numpy as np
import s12_04_contact as C
from s12_common import *
from pydrake.all import (AddMultibodyPlantSceneGraph, PrismaticJoint, LinearSpringDamper, UnitInertia, CoulombFriction,
                         HalfSpace, Sphere, ProximityProperties, AddContactMaterial, SpatialInertia)

def run(m_leg=5.0, k_leg=33333.0, k_c=3e5, dt=1e-3, T=3.0, X0=None, integ="runge_kutta3"):
    X0 = C.X0 if X0 is None else X0
    L = C.L; L0 = 0.1; r = 0.01
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, 0.0)
    body = plant.AddRigidBody("hopper", spatial_inertia(C.PM))
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -G])
    def props(k, d=0.0):
        pp = ProximityProperties(); AddContactMaterial(d, k, CoulombFriction(L.friction, L.friction), pp); return pp
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", props(2 * k_c))
    legs = []
    for i, f in enumerate(C.FEET_B):
        leg = plant.AddRigidBody(f"leg{i}", SpatialInertia.SolidSphereWithMass(m_leg, r))
        # joint frame on the hopper at the extended foot position; leg origin coincides at q = 0
        plant.AddJoint(PrismaticJoint(f"j{i}", plant.GetFrameByName("hopper"), plant.GetFrameByName(f"leg{i}"), [0, 0, 1]))
        plant.RegisterCollisionGeometry(leg, RigidTransform([0, 0, r]), Sphere(r), f"foot{i}", props(2 * k_c))
        legs.append(leg)
    plant.Finalize()
    for i, f in enumerate(C.FEET_B):  # place joint parent frame: X_PF set via frames, so use offsets in the spring instead
        pass
    return plant, body, legs

# Joint frames: prismatic joint between frames F (on parent) and M (on child). Using the body frames directly would put
# the leg origin at the pivot; instead add explicit offset frames.
def build(m_leg=5.0, k_leg=33333.0, c_leg=1500.0, k_c=3e5, dt=1e-3, T=3.0, X0=None, integ="runge_kutta3"):
    from pydrake.all import FixedOffsetFrame
    X0 = C.X0 if X0 is None else X0
    L = C.L; L0 = 0.1; r = 0.01
    b = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(b, 0.0)
    body = plant.AddRigidBody("hopper", spatial_inertia(C.PM))
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -G])
    def props(k):
        pp = ProximityProperties(); AddContactMaterial(0.0, k, CoulombFriction(L.friction, L.friction), pp); return pp
    plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", props(2 * k_c))
    legs = []
    for i, f in enumerate(C.FEET_B):
        leg = plant.AddRigidBody(f"leg{i}", SpatialInertia.SolidSphereWithMass(m_leg, r))
        fr = plant.AddFrame(FixedOffsetFrame(f"mount{i}", body.body_frame(), RigidTransform(f)))
        plant.AddJoint(PrismaticJoint(f"j{i}", fr, leg.body_frame(), [0, 0, 1]))
        plant.RegisterCollisionGeometry(leg, RigidTransform([0, 0, r]), Sphere(r), f"foot{i}", props(2 * k_c))
        plant.AddForceElement(LinearSpringDamper(body, f + [0, 0, L0], leg, np.zeros(3), L0, k_leg, c_leg))
        legs.append(leg)
    plant.Finalize()
    dia = b.Build(); sim = Simulator(dia)
    ResetIntegratorFromFlags(sim, integ, dt).set_fixed_step_mode(integ == "runge_kutta3")
    ctx = sim.get_mutable_context(); pc = plant.GetMyMutableContextFromRoot(ctx)
    q, p_o, w, v = legacy_to_drake(X0)
    plant.SetFreeBodyPose(pc, body, RigidTransform(Quaternion(q), p_o)); plant.SetFreeBodySpatialVelocity(pc, body, SpatialVelocity(w, v))
    for lg in legs:  # leg velocity = body velocity (joint velocities zero): set joint states zero is default
        pass
    sim.Initialize()
    fz, fs = [], []
    t0 = time.time()
    for k in range(int(T / dt)):
        sim.AdvanceTo((k + 1) * dt)
        zs = [plant.EvalBodyPoseInWorld(pc, lg).translation()[2] for lg in legs]
        fz.append(min(zs))
        cr = plant.get_contact_results_output_port().Eval(pc)
        fs.append(sum(abs(cr.point_pair_contact_info(i).contact_force()[2]) for i in range(cr.num_point_pair_contacts())))
    return fz, fs, time.time() - t0

if __name__ == "__main__":
    x, lz, lf = C.legacy(); C.summarize("legacy", lz, lf)
    for m, kc, dt in ((5.0, 3e5, 1e-3), (5.0, 3e5, 5e-4), (2.0, 3e5, 5e-4), (5.0, 1e6, 2.5e-4)):
        try:
            fz, fs, el = build(m_leg=m, k_c=kc, dt=dt, T=3.0)
            C.summarize(f"(b) m_leg={m} k_c={kc:g} dt={dt:g}", fz, fs); print("    speed", int(3.0 / dt / el), "steps/s")
        except Exception as e:
            print("(b) failed", m, kc, dt, type(e).__name__, str(e)[:120])
