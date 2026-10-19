from s12_common import *
np.set_printoptions(precision=3, linewidth=160)
def mk(thr, gx=0.0, gy=0.0, roll=0.0):
    return lambda t: Command(throttle=thr, gimbal=np.array([gx, gy]), roll_torque=roll)
x0z = initial_state(); 
x_still = make_state([0,0,30],[0,0,0],None,[0,0,0],prop=20.0)
for name, sc, x0 in (("straight,w=0", mk(0.7), x_still), ("straight,w!=0", mk(0.7), None), ("gimbal", mk(0.7, 0.002, 0.001), x_still)):
    tr = legacy_run(6.0, 5e-4, x0=x0, sched=sc)
    for mode in ("none", "pos"):
        d = DrakeHopper(1e-3, "rk3", sched=sc, x0=x0); out = d.run(6.0, mass_mode=mode)
        print(name, mode, compare(out, tr))
