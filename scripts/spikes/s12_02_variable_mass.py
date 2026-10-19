"""Q1/Q2/Q5: variable mass in one floating body vs legacy over a full burn; integrator convergence."""
import time
from s12_common import *

T = 12.0
truth = legacy_run(T, 2.5e-4)
leg1 = legacy_run(T, 1e-3); leg05 = legacy_run(T, 5e-4)
np.set_printoptions(precision=3, linewidth=140)
print("cols: pos m, vel m/s, att rad, omega rad/s, prop kg")
print("legacy 1ms   vs truth", compare(leg1, truth))
print("legacy 0.5ms vs truth", compare(leg05, truth))
print("legacy 1ms vs 0.5ms (self-convergence)", compare(leg1, leg05))
print("final prop (truth)", truth[T][PROP], "mass", CFG.mass.dry_mass + truth[T][PROP])
def drk(dt, integ, **kw):
    d = DrakeHopper(dt, integ); d.mid = kw.pop("mid", True); return d

for mid in (False, True):
    for mode in ("none", "vel", "pos"):
        out = drk(1e-3, "rk3", mid=mid).run(T, mass_mode=mode)
        print(f"rk3 1ms mid={mid} mass_mode={mode}: vs truth {compare(out, truth)}")
for integ, dt in (("rk2", 1e-3), ("rk3", 4e-3), ("rk3", 2e-3), ("rk3", 1e-3), ("rk3", 5e-4)):
    t0 = time.time(); d = drk(dt, integ); out = d.run(T, mass_mode="pos"); el = time.time() - t0
    print(f"drake {integ} dt={dt:g} mid pos: vs truth {compare(out, truth)}  {T/dt/el:.0f} steps/s")
