"""Q4: touchdown at hop speeds (0.7 m/s) and a hard 5 m/s landing, legacy vs Drake point contact."""
import numpy as np
import s12_04_contact as C
from s12_common import *
import pydrake.multibody.plant as mp

x_rest, _, _ = C.legacy()
rest_z = x_rest[2]
old = mp.MultibodyPlant.Finalize
def with_stiction(tol):
    def fin(self, *a): self.set_stiction_tolerance(tol); return old(self, *a)
    mp.MultibodyPlant.Finalize = fin
for vz, tilt_vx in ((-0.7, 0.0), (-0.7, 0.3), (-5.0, 0.0)):
    C.X0 = make_state([0, 0, rest_z + 0.002 + 0.5 * 0], [tilt_vx, 0, vz], None, [0, 0, 0], prop=C.PM)
    C.T = 2.0
    x, lz, lf = C.legacy(); print(f"--- vz={vz} vx={tilt_vx}")
    C.summarize("legacy", lz, lf)
    for d in (2.0, 2.5, 3.0):
        for tol in (0.05,):
            with_stiction(tol)
            try:
                r = C.drake_run("point", diss=d)
            finally:
                mp.MultibodyPlant.Finalize = old
            C.summarize(f"(a) d={d} stic={tol}", r[0], r[1])
