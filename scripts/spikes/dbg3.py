from s12_common import *
np.set_printoptions(precision=3, linewidth=160)
mk = lambda thr: (lambda t: Command(throttle=thr, gimbal=np.zeros(2), roll_torque=0.0))
x_still = make_state([0,0,30],[0,0,0],None,[0,0,0],prop=20.0)
tr = legacy_run(6.0, 5e-4, x0=x_still, sched=mk(0.7))
for mid in (False, True):
    d = DrakeHopper(1e-3, "rk3", sched=mk(0.7), x0=x_still); d.mid = mid
    print("mid", mid, compare(d.run(6.0, mass_mode="pos"), tr))
