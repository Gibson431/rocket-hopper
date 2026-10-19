"""Q6: specific force for the IMU from the plant's body spatial acceleration output port."""
from s12_common import *
np.set_printoptions(precision=6, linewidth=150)

def specific_force(d):
    s = d.plant.GetPositionsAndVelocities(d.pctx)
    q, w_I = s[:4], s[7:10]
    A = d.plant.get_body_spatial_accelerations_output_port().Eval(d.pctx)[d.body.index()]
    R = quat_to_dcm(q); r = R @ np.array([0, 0, com_z(d.prop())])
    a_o, alpha = A.translational(), A.rotational()
    a_com = a_o + np.cross(alpha, r) + np.cross(w_I, np.cross(w_I, r))   # IMU at the CoM (legacy mount)
    return R.T @ (a_com + np.array([0, 0, G]))   # specific force = a - g (g = [0,0,-G])

d = DrakeHopper(1e-3, "rk3"); d.mid = True
worst = 0.0
for k in range(12000):
    d.sim.AdvanceTo((k + 1) * 1e-3); d.update_mass("pos")
    if (k + 1) % 500 == 0:
        t = (k + 1) * 1e-3; x = d.legacy_state(); info = {}
        vehicle_derivative(t, x, schedule(t - 1e-9), CFG, Environment(aero_enabled=False), info=info)
        diff = np.linalg.norm(specific_force(d) - info["specific_force_B"]); worst = max(worst, diff)
print("burn: max |f_drake - f_legacy| over 24 samples, m/s^2:", worst, " sample f_B =", info["specific_force_B"])

# Does the port exist/behave on a discrete plant? (acceleration = delta-v / dt from the contact solver)
d2 = DrakeHopper(1e-3, "rk3", plant_dt=1e-3); d2.mid = True; d2.sim.AdvanceTo(1.0)
try:
    A = d2.plant.get_body_spatial_accelerations_output_port().Eval(d2.pctx)[d2.body.index()]
    print("discrete plant spatial accel port ok:", A.translational())
except Exception as e:
    print("discrete plant spatial accel port FAILED:", type(e).__name__, str(e)[:160])
