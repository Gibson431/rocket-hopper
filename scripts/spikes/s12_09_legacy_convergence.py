"""Legacy hop (perfect state) at dt=1 ms vs 0.5 ms: self-convergence for the parity tolerances."""
import numpy as np
from s12_common import CFG
from hopper.scenario_glue import run_hop
res = {}
for dt in (1e-3, 5e-4):
    cfg = CFG.model_copy(update={"sim": CFG.sim.model_copy(update={"dt": dt})})
    r, s = run_hop(cfg, log_decimation=int(round(0.01 / dt)))
    res[dt] = (r, s); print(dt, r.outcome, "t_end", r.t_end if hasattr(r, "t_end") else None, s)
a, b = res[1e-3][0].log, res[5e-4][0].log
print(list(a.columns))
n = min(len(a), len(b))
print("rows", len(a), len(b))
for cols in (["pos_x","pos_y","pos_z"], ["vel_x","vel_y","vel_z"], ["omega_x","omega_y","omega_z"], ["prop"], ["thr"]):
    try:
        d = np.abs(a[cols].values[:n] - b[cols].values[:n]); print(cols, "max", d.max(), "rms", np.sqrt((d**2).mean()))
    except KeyError as e:
        print("missing", e)
