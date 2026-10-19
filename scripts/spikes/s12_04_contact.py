"""Q4: 0.5 m drop. Legacy leg law vs (a) Drake point contact vs (c) legacy law as applied force."""
import sys, time
from s12_common import *
from pydrake.all import (AddMultibodyPlantSceneGraph, CoulombFriction, HalfSpace, Sphere, ProximityProperties,
                         AddContactMaterial, RigidTransform, ContactModel)
from hopper.dynamics.contact import contact_loads

PM = 20.0
MP = mass_properties(PM, CFG)
L = CFG.legs
ang = 2 * np.pi * np.arange(L.count) / L.count
FEET_B = np.stack([L.radius * np.cos(ang), L.radius * np.sin(ang), np.full(L.count, -L.length)], 1)  # from pivot
Z0 = 0.5 + MP.com_z + L.length
X0 = make_state([0, 0, Z0], [0, 0, 0], None, [0, 0, 0], prop=PM)
T = 3.0


def metrics(t, foot_z, fsum):
    foot_z, fsum = np.array(foot_z), np.array(fsum)
    first = np.argmax(foot_z <= 0)
    after = foot_z[first:]
    # rebound: highest foot height after the first time the peak force occurred
    kpk = int(np.argmax(fsum))
    return dict(peak_force=fsum.max(), max_pen=-foot_z.min(), rebound=after[np.argmax(fsum[first:] > 0.0) + 5:].max() if len(after) > 10 else 0.0,
                rebound_after_peak=foot_z[kpk:].max() if kpk else 0.0, settle_com=None)


def legacy(dt=1e-3):
    x = X0.copy(); fz, fs = [], []
    n = int(T / dt)
    cmd = Command(throttle=0.0, gimbal=np.zeros(2), roll_torque=0.0)
    for k in range(n):
        info = {}
        x = rk4_step(lambda tt, xx: vehicle_derivative(tt, xx, cmd, CFG, Environment(aero_enabled=False), info=info), k * dt, x, dt)
        R = quat_to_dcm(x[QUAT]); feet = x[POS] + (R @ (FEET_B - [0, 0, MP.com_z]).T).T
        fz.append(feet[:, 2].min()); fs.append(contact_loads(x, MP, L).foot_forces.sum())
    return x, fz, fs


class LegacyContact(LeafSystem):
    def __init__(self, body):
        super().__init__()
        self.bi = body.index()
        self.DeclareVectorInputPort("s", 13)
        self.DeclareAbstractOutputPort("f", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self.calc)

    def calc(self, c, out):
        s = self.get_input_port().Eval(c)
        p, v, w = drake_to_legacy(s[:4], s[4:7], s[7:10], s[10:13], PM)
        x = np.zeros(N_STATE); x[POS], x[VEL], x[QUAT], x[OMEGA], x[PROP] = p, v, s[:4], w, PM
        co = contact_loads(x, MP, L)
        R = quat_to_dcm(s[:4]); f = ExternallyAppliedSpatialForce()
        f.body_index = self.bi; f.p_BoBq_B = np.array([0, 0, MP.com_z])
        f.F_Bq_W = SpatialForce(R @ co.torque_B, co.force_I); out.set_value([f])


def drake_run(kind, dt=1e-3, k_series=None, diss=0.0, integ="runge_kutta3", plant_dt=0.0, mu=None, ret_plant=False):
    b = DiagramBuilder()
    if kind == "point":
        plant, sg = AddMultibodyPlantSceneGraph(b, plant_dt)
    else:
        plant = b.AddSystem(MultibodyPlant(plant_dt))
    body = plant.AddRigidBody("hopper", spatial_inertia(PM))
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -G])
    if kind == "point":
        kf = 2 * (k_series or L.stiffness)  # two equal geometries in series -> k_series
        def props():
            pp = ProximityProperties(); AddContactMaterial(diss, kf, CoulombFriction(L.friction, L.friction), pp); return pp
        plant.RegisterCollisionGeometry(plant.world_body(), RigidTransform(), HalfSpace(), "ground", props())
        for i, f in enumerate(FEET_B):
            plant.RegisterCollisionGeometry(body, RigidTransform(f + [0, 0, 0.01]), Sphere(0.01), f"foot{i}", props())
    plant.Finalize()
    if kind == "legacy":
        lc = b.AddSystem(LegacyContact(body))
        b.Connect(plant.get_state_output_port(), lc.get_input_port())
        b.Connect(lc.get_output_port(), plant.get_applied_spatial_force_input_port())
    dia = b.Build(); sim = Simulator(dia)
    if plant_dt == 0.0:
        ResetIntegratorFromFlags(sim, integ, dt).set_fixed_step_mode(integ == "runge_kutta3")
    ctx = sim.get_mutable_context(); pc = plant.GetMyMutableContextFromRoot(ctx)
    q, p_o, w, v = legacy_to_drake(X0)
    plant.SetFreeBodyPose(pc, body, RigidTransform(Quaternion(q), p_o)); plant.SetFreeBodySpatialVelocity(pc, body, SpatialVelocity(w, v))
    sim.Initialize()
    fz, fs, zc = [], [], []
    t0 = time.time()
    for k in range(int(T / dt)):
        sim.AdvanceTo((k + 1) * dt)
        s = plant.GetPositionsAndVelocities(pc); R = quat_to_dcm(s[:4])
        feet = s[4:7] + (R @ FEET_B.T).T
        fz.append(feet[:, 2].min()); zc.append(s[6])
        f = 0.0
        if kind == "point":
            cr = plant.get_contact_results_output_port().Eval(pc)
            for i in range(cr.num_point_pair_contacts()):
                f += abs(cr.point_pair_contact_info(i).contact_force()[2])
        else:
            st = np.zeros(N_STATE); pp, vv, ww = drake_to_legacy(s[:4], s[4:7], s[7:10], s[10:13], PM)
            st[POS], st[VEL], st[QUAT], st[OMEGA], st[PROP] = pp, vv, s[:4], ww, PM
            f = contact_loads(st, MP, L).foot_forces.sum()
        fs.append(f)
    return fz, fs, zc, time.time() - t0, s


def summarize(name, fz, fs, zc=None):
    fz, fs = np.array(fz), np.array(fs)
    i1 = int(np.argmax(fs > 0)); kpk = int(np.argmax(fs))
    # lowest point after first impact, then highest foot height afterwards (rebound), then final rest
    lo = int(np.argmin(fz)); post = fz[lo + 20:]
    print(f"{name:34s} peakF {fs.max():7.0f} N  max pen {-fz.min()*1000:6.1f} mm  rebound {post.max()*1000 if len(post) else 0:7.1f} mm "
          f"rest foot z {fz[-1]*1000:7.2f} mm  rest F {fs[-1]:7.1f} N  sum F/weight {fs[-1]/(MP.mass*G):.3f}")

if __name__ == "__main__":
    x, fz, fs = legacy(); summarize("legacy RK4 1ms", fz, fs)
    r = drake_run("legacy"); summarize("(c) legacy law, RK3 1ms", r[0], r[1]); print("   speed", int(T/1e-3/r[3]), "steps/s")
    for d in (0.0, 0.5, 2.0, 6.0, 10.0):
        r = drake_run("point", diss=d); summarize(f"(a) point k_series=30k diss={d}", r[0], r[1])
