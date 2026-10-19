"""Q4 cont.: tune Hunt-Crossley, then friction (slide distance, yaw limit cycle) and stiction tolerance."""
import numpy as np
import s12_04_contact as C
from s12_common import *

def force_rms(fs_a, fs_b):
    a, b = np.array(fs_a), np.array(fs_b)
    ia, ib = int(np.argmax(a > 0)), int(np.argmax(b > 0))
    n = 400  # 0.4 s after first contact
    return np.sqrt(np.mean((a[ia:ia+n] - b[ib:ib+n])**2)), np.abs(a[ia:ia+n] - b[ib:ib+n]).max()

x, lz, lf = C.legacy()
print("legacy: peak", max(lf), "max pen mm", -min(lz)*1000)
for d in (1.5, 2.0, 2.5, 3.0):
    r = C.drake_run("point", diss=d)
    pen = -min(r[0])*1000
    print(f"d={d}: peakF {max(r[1]):.0f}  pen {pen:.1f} mm  force rms/max diff vs legacy over 0.4 s {force_rms(r[1], lf)}")



L2 = CFG.legs.model_copy(update={"friction": 0.3}); C.CFG = CFG.model_copy(update={"legs": L2}); C.L = L2
rest_z = x[2]
import pydrake.multibody.plant as mp
old = mp.MultibodyPlant.Finalize
for name, v0, w0 in (("slide 1 m/s, no spin", 1.0, 0.0), ("spin 1 rad/s only", 0.0, 1.0)):
    C.X0 = make_state([0, 0, rest_z], [v0, 0, 0], None, [0, 0, w0], prop=C.PM)
    print("---", name, "(Coulomb distance %.4f m)" % (v0**2 / (2 * 0.3 * G)))
    xl, _, _ = C.legacy(); print("legacy : x %.4f vx %.2e wz %.2e" % (xl[0], xl[3], xl[12]))
    for tol in (1e-4, 1e-3, 1e-2, 5e-2):
        def fin(self, *a, tol=tol): self.set_stiction_tolerance(tol); return old(self, *a)
        mp.MultibodyPlant.Finalize = fin
        try:
            r = C.drake_run("point", diss=2.5); s = r[4]
            print(f"(a) stiction_tol={tol:g}: x {s[4]:.4f} vx {s[10]:.2e} wz {s[9]:.2e}")
        finally: mp.MultibodyPlant.Finalize = old
