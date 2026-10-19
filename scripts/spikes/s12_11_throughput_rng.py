"""Q7 (determinism with explicit Generators) and Q8 (throughput of Python systems on the hot path)."""
import time
from s12_common import *

class Gust(LeafSystem):
    """OU gust: noise drawn from a per-instance Generator in a periodic discrete update (never in derivatives)."""
    def __init__(self, body, seed, dt=1e-3, tau=1.0, sigma=2.0):
        super().__init__()
        self.rng = np.random.default_rng(seed); self.bi = body.index()
        self.a = np.exp(-dt / tau); self.b = sigma * np.sqrt(1 - self.a**2)
        self.DeclareDiscreteState(3)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self.upd)
        self.DeclareAbstractOutputPort("f", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self.calc)
    def upd(self, c, ds):
        ds.set_value(self.a * c.get_discrete_state_vector().CopyToVector() + self.b * self.rng.standard_normal(3))
    def calc(self, c, out):
        f = ExternallyAppliedSpatialForce(); f.body_index = self.bi; f.p_BoBq_B = np.zeros(3)
        f.F_Bq_W = SpatialForce(np.zeros(3), 5.0 * c.get_discrete_state_vector().CopyToVector()); out.set_value([f])

class Dummy(LeafSystem):
    """Stand-in for a propulsion-like Python force system: reads the plant state, does small numpy work."""
    def __init__(self, body):
        super().__init__()
        self.bi = body.index(); self.DeclareVectorInputPort("s", 13)
        self.DeclareAbstractOutputPort("f", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self.calc)
    def calc(self, c, out):
        s = self.get_input_port().Eval(c); R = quat_to_dcm(s[:4])
        f = ExternallyAppliedSpatialForce(); f.body_index = self.bi; f.p_BoBq_B = np.zeros(3)
        f.F_Bq_W = SpatialForce(np.zeros(3), R @ np.array([0, 0, 500.0]) - 0.1 * s[10:13]); out.set_value([f])

class Combiner(LeafSystem):
    def __init__(self, n):
        super().__init__()
        self.ports = [self.DeclareAbstractInputPort(f"f{i}", AbstractValue.Make([ExternallyAppliedSpatialForce()])) for i in range(n)]
        self.DeclareAbstractOutputPort("f", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce()]), self.calc)
    def calc(self, c, out):
        out.set_value([f for p in self.ports for f in p.Eval(c)])

def build(n_py, seed=None, dt=1e-3, T=5.0, mass_update=False):
    b = DiagramBuilder(); plant = b.AddSystem(MultibodyPlant(0.0))
    body = plant.AddRigidBody("hopper", spatial_inertia(20.0)); plant.Finalize()
    systems = [b.AddSystem(Gust(body, seed))] if seed is not None else []
    systems += [b.AddSystem(Dummy(body)) for _ in range(n_py)]
    for s in systems:
        if isinstance(s, Dummy): b.Connect(plant.get_state_output_port(), s.get_input_port())
    comb = b.AddSystem(Combiner(len(systems)))
    for i, s in enumerate(systems): b.Connect(s.GetOutputPort("f"), comb.ports[i])
    b.Connect(comb.get_output_port(), plant.get_applied_spatial_force_input_port())
    dia = b.Build(); sim = Simulator(dia); ResetIntegratorFromFlags(sim, "runge_kutta3", dt).set_fixed_step_mode(True)
    pc = plant.GetMyMutableContextFromRoot(sim.get_mutable_context())
    plant.SetFreeBodyPose(pc, body, RigidTransform([0, 0, 10.0])); sim.Initialize()
    return sim, plant, pc, body

def run(sim, plant, pc, body, T=5.0, dt=1e-3, outer=False):
    t0 = time.time()
    if outer:
        for k in range(int(T / dt)):
            sim.AdvanceTo((k + 1) * dt); body.SetSpatialInertiaInBodyFrame(pc, spatial_inertia(20.0 - 1e-4 * k))
    else:
        sim.AdvanceTo(T)
    return plant.GetPositionsAndVelocities(pc).copy(), T / dt / (time.time() - t0)

if __name__ == "__main__":
    print("== Q7 determinism")
    a, _ = run(*build(0, seed=123)); b_, _ = run(*build(0, seed=123))
    other = build(2, seed=999); run(*other)                       # a different simulation in the same process
    c, _ = run(*build(0, seed=123))
    print("run A == run B (same seed):", np.array_equal(a, b_), "; A == C after an unrelated sim ran:", np.array_equal(a, c),
          "; seed 124 differs:", not np.array_equal(a, run(*build(0, seed=124))[0]))

    print("== Q8 throughput (steps/s, RK3 1 ms, free flight)")
    for n in (0, 1, 3, 6):
        r1 = run(*build(n, seed=1))[1]; r2 = run(*build(n, seed=1), outer=True)[1]
        print(f"plant + gust + {n} python force systems + combiner: AdvanceTo(T) {r1:.0f}, outer loop with per-step mass write {r2:.0f}")
    t0 = time.time(); legacy_run(5.0, 1e-3); print("legacy open loop 5 s:", int(5000 / (time.time() - t0)), "steps/s")
