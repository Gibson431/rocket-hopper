"""Print attitude PD gains and the hover plant gain for a chosen bandwidth and damping.

uv run python scripts/tune_attitude.py --wn 3 3 1.5 --zeta 0.8
"""

import argparse

import numpy as np
from hopper_control.attitude import AttitudeGains

from hopper.config import load_vehicle
from hopper.dynamics.mass_model import mass_properties
from hopper.scenario_glue import control_params_from_config


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vehicles/vehicle_default.yaml")
    ap.add_argument("--wn", type=float, nargs="+", default=[3.0, 3.0, 1.5], help="rad/s (1 or 3)")
    ap.add_argument("--zeta", type=float, default=0.8)
    args = ap.parse_args()

    cfg = load_vehicle(args.config)
    params = control_params_from_config(cfg)
    gains = AttitudeGains.from_bandwidth(args.wn if len(args.wn) == 3 else args.wn[0], args.zeta)
    np.set_printoptions(precision=4, suppress=True)
    print(f"wn = {np.broadcast_to(args.wn, 3)} rad/s, zeta = {args.zeta}")
    print(f"kp = {gains.kp} 1/s^2\nkd = {gains.kd} 1/s")
    print(f"gimbal bandwidth = {cfg.engine.gimbal_wn} rad/s")
    for prop in (0.0, cfg.tank.prop_mass_initial):
        mp = mass_properties(prop, cfg)
        k = mp.com_z * params.thrust_min_frac * params.thrust_max / mp.inertia[0, 0]
        print(
            f"prop {prop:5.1f} kg: plant gain K = z_com T / I_xx = {k:.2f} rad/s^2 per rad (T min)"
        )
        print(f"   gimbal for a 1 rad/s^2 demand = {1.0 / k:.4f} rad")


if __name__ == "__main__":
    main()
