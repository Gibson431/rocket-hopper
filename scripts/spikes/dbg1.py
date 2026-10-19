from s12_common import *
np.set_printoptions(precision=4, linewidth=160)
const = lambda t: Command(throttle=0.7, gimbal=np.zeros(2), roll_torque=0.0)
for name, sc in (("const", const), ("sched", schedule)):
    tr = legacy_run(12.0, 2.5e-4, sched=sc, sample_dt=0.5)
    d = DrakeHopper(1e-3, "rk3", sched=sc); out = d.run(12.0)
    print(name, "prop err by time:", [f"{out[t][PROP]-tr[t][PROP]:.2e}" for t in sorted(out)])
    print(name, "thr  err by time:", [f"{out[t][THR]-tr[t][THR]:.2e}" for t in sorted(out)])
