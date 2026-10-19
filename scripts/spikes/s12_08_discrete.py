"""Q3: discrete plant (time_step 1e-3) vs continuous: free-flight accuracy over the 12 s burn, and a drop."""
import time
from s12_common import *
import s12_04_contact as C
T = 12.0
truth = legacy_run(T, 2.5e-4)
np.set_printoptions(precision=3, linewidth=150)
for pdt, label in ((0.0, "continuous RK3 1ms"), (1e-3, "discrete 1e-3 (SAP)"), (2e-3, "discrete 2e-3")):
    d = DrakeHopper(min(1e-3, pdt) if pdt else 1e-3, "rk3", plant_dt=pdt); d.mid = True
    t0 = time.time(); out = d.run(T, mass_mode="pos"); el = time.time() - t0
    print(f"{label:22s}: vs truth {compare(out, truth)}  {T/1e-3/el:.0f} steps/s")
# drop with the discrete plant
x, lz, lf = C.legacy(); C.summarize("legacy drop", lz, lf)
for pdt in (1e-3, 2e-3):
    for d in (0.0, 1.0, 2.5):
        r = C.drake_run("point", dt=pdt, diss=d, plant_dt=pdt)
        C.summarize(f"discrete {pdt:g} diss={d}", r[0], r[1])
