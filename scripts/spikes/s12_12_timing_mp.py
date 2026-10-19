"""Q9a: controller + command delay ordering. Q9: Monte Carlo workers build the diagram per process."""
import numpy as np
from pydrake.all import DiagramBuilder, LeafSystem, Simulator, ResetIntegratorFromFlags

class Ctrl(LeafSystem):  # 100 Hz controller: counts its calls
    def __init__(self):
        super().__init__(); self.DeclareDiscreteState(1)
        self.DeclarePeriodicDiscreteUpdateEvent(0.01, 0.0, lambda c, ds: ds.set_value(c.get_discrete_state_vector().CopyToVector() + 1))
        self.DeclareVectorOutputPort("u", 1, lambda c, o: o.SetFromVector(c.get_discrete_state_vector().CopyToVector()))

class Hold(LeafSystem):  # separate 1 kHz delay-0 "command latch"
    def __init__(self):
        super().__init__(); self.DeclareDiscreteState(1); p = self.DeclareVectorInputPort("u", 1)
        self.DeclarePeriodicDiscreteUpdateEvent(1e-3, 0.0, lambda c, ds: ds.set_value(p.Eval(c)))
        self.DeclareVectorOutputPort("y", 1, lambda c, o: o.SetFromVector(c.get_discrete_state_vector().CopyToVector()))

class Merged(LeafSystem):  # controller + delay in one system: new command applies in the same step
    def __init__(self):
        super().__init__(); self.DeclareDiscreteState(2)  # [controller counter, applied command]
        self.DeclarePeriodicDiscreteUpdateEvent(1e-3, 0.0, self.upd)
        self.DeclareVectorOutputPort("y", 1, lambda c, o: o.SetFromVector(c.get_discrete_state_vector().CopyToVector()[1:]))
    def upd(self, c, ds):
        n, y = c.get_discrete_state_vector().CopyToVector()
        if abs(c.get_time() / 0.01 - round(c.get_time() / 0.01)) < 1e-9: n += 1; y = n   # controller tick, delay 0
        ds.set_value([n, y])

class Probe(LeafSystem):  # integrates the applied command: shows which command acted over each step
    def __init__(self):
        super().__init__(); self.DeclareContinuousState(1); self.p = self.DeclareVectorInputPort("y", 1)
    def DoCalcTimeDerivatives(self, c, d): d.get_mutable_vector().SetFromVector(self.p.Eval(c))

def trial(merged):
    b = DiagramBuilder(); pr = b.AddSystem(Probe())
    if merged: m = b.AddSystem(Merged()); b.Connect(m.get_output_port(), pr.p)
    else:
        c = b.AddSystem(Ctrl()); h = b.AddSystem(Hold()); b.Connect(c.get_output_port(), h.get_input_port()); b.Connect(h.get_output_port(), pr.p)
    s = Simulator(b.Build()); ResetIntegratorFromFlags(s, "runge_kutta3", 1e-3).set_fixed_step_mode(True); s.Initialize()
    s.AdvanceTo(0.011)
    return pr.GetMyContextFromRoot(s.get_context()).get_continuous_state_vector().GetAtIndex(0) / 1e-3

print("Integral of applied command over 0.011 s, in units of (command x 1 ms).")
print("Legacy semantics: tick at t=0 applies cmd 1 over [0,10ms) (10 units); tick at t=10ms applies cmd 2 over [10,11ms) (2 units) -> 12")
print("separate controller + latch (delay 0):", trial(False))
print("controller + delay in one system     :", trial(True))

# Q9: multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
def worker(seed):
    from s12_11_throughput_rng import build, run   # builds the Drake diagram inside the worker
    import numpy as np
    out, _ = run(*build(0, seed=seed, T=1.0), T=1.0)
    return os.getpid(), seed, out[4:7].tolist()
if __name__ == "__main__":
    import pickle, sys
    from pydrake.all import MultibodyPlant
    try: pickle.dumps(MultibodyPlant(0.0)); print("MultibodyPlant picklable: yes")
    except Exception as e: print("MultibodyPlant picklable: no ->", type(e).__name__)
    seeds = [1, 2, 3, 1, 2, 3]
    with ProcessPoolExecutor(3) as ex: res = list(ex.map(worker, seeds))
    ref = {s: worker(s)[2] for s in (1, 2, 3)}
    print("pids", sorted({r[0] for r in res}), "| each worker result == serial result:", all(r[2] == ref[r[1]] for r in res))
