"""Q1: compare mass-update mechanisms by speed/behaviour: (a) outer AdvanceTo loop (see s12_02),
(b) periodic publish callback mutating the root context, (c) custom LeafSystem rigid body."""
import time
from s12_common import *
from pydrake.all import LeafSystem, Simulator, DiagramBuilder, ResetIntegratorFromFlags

T, DT = 6.0, 1e-3
sc = lambda t: Command(throttle=0.7, gimbal=np.zeros(2), roll_torque=0.0)
x0 = make_state([0, 0, 30], [0, 0, 0], None, [0, 0, 0], prop=20.0)
truth = legacy_run(T, 5e-4, x0=x0, sched=sc)

# (a)
d = DrakeHopper(DT, "rk3", sched=sc, x0=x0); d.mid = True
t0 = time.time(); out = d.run(T, mass_mode="pos"); ta = time.time() - t0
print("(a) outer loop      ", compare(out, truth), f"{T/DT/ta:.0f} steps/s")

# (b) publish callback with the root context captured
class Updater(LeafSystem):
    def __init__(self, hop):
        super().__init__()
        self.hop = hop
        self.DeclarePeriodicPublishEvent(DT, 0.0, self.pub)
        self.DeclareVectorInputPort("prop", 1)

    def pub(self, c):
        h = self.hop
        prop = self.get_input_port().Eval(c)[0]  # prop at start of the step
        h.body.SetSpatialInertiaInBodyFrame(h.plant.GetMyMutableContextFromRoot(h.root), spatial_inertia(prop))

b = DiagramBuilder()
plant = b.AddSystem(MultibodyPlant(0.0))
body = plant.AddRigidBody("hopper", spatial_inertia(20.0))
plant.mutable_gravity_field().set_gravity_vector([0, 0, -G])
plant.Finalize()
eng = b.AddSystem(Engine(plant, body, sc))
b.Connect(eng.GetOutputPort("force"), plant.get_applied_spatial_force_input_port())
b.Connect(plant.get_state_output_port(), eng.GetInputPort("plant_state"))
class H: pass
h = H(); h.plant, h.body = plant, body
upd = b.AddSystem(Updater(h))
b.Connect(eng.GetOutputPort("prop"), upd.get_input_port())
dia = b.Build(); sim = Simulator(dia); ResetIntegratorFromFlags(sim, "runge_kutta3", DT).set_fixed_step_mode(True)
h.root = sim.get_mutable_context()
pc = plant.GetMyMutableContextFromRoot(h.root); ec = eng.GetMyMutableContextFromRoot(h.root)
q, p_o, w_I, v_o = legacy_to_drake(x0)
plant.SetFreeBodyPose(pc, body, RigidTransform(Quaternion(q), p_o)); plant.SetFreeBodySpatialVelocity(pc, body, SpatialVelocity(w_I, v_o))
ec.get_mutable_continuous_state_vector().SetFromVector(np.array([20.0, 0, 0, 0, 0, 0]))
sim.Initialize()
t0 = time.time(); sim.AdvanceTo(T); tb = time.time() - t0
s = plant.GetPositionsAndVelocities(pc); a = ec.get_continuous_state_vector().CopyToVector()
p, v, w = drake_to_legacy(s[:4], s[4:7], s[7:10], s[10:13], a[0])
xb = truth[T]
print("(b) publish callback ", "pos err", np.linalg.norm(p - xb[POS]), "vel err", np.linalg.norm(v - xb[VEL]), f"{T/DT/tb:.0f} steps/s",
      "(mass lags a half step, no CoM correction)")

# (c) custom rigid body: legacy ODE as a LeafSystem (19 continuous states)
class Body(LeafSystem):
    def __init__(self):
        super().__init__()
        self.DeclareContinuousState(N_STATE)
        self.cmd = sc(0)

    def DoCalcTimeDerivatives(self, c, d):
        x = c.get_continuous_state_vector().CopyToVector()
        d.get_mutable_vector().SetFromVector(vehicle_derivative(c.get_time(), x, self.cmd, CFG, Environment(aero_enabled=False)))
bb = DiagramBuilder(); bo = bb.AddSystem(Body()); dia2 = bb.Build(); sim2 = Simulator(dia2)
ResetIntegratorFromFlags(sim2, "runge_kutta3", DT).set_fixed_step_mode(True)
bo.GetMyMutableContextFromRoot(sim2.get_mutable_context()).get_mutable_continuous_state_vector().SetFromVector(x0)
sim2.Initialize(); t0 = time.time(); sim2.AdvanceTo(T); tc = time.time() - t0
xc = bo.GetMyContextFromRoot(sim2.get_context()).get_continuous_state_vector().CopyToVector()
print("(c) custom body      ", "pos err", np.linalg.norm(xc[POS] - xb[POS]), "vel err", np.linalg.norm(xc[VEL] - xb[VEL]), f"{T/DT/tc:.0f} steps/s")
