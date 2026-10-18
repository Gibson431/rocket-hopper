"""Minimal v1 scenario runner: an open-loop burn, or `--mission hop` for the closed-loop hop.

S11 replaces it with scenario files.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from hopper.analysis.logger import save_frame
from hopper.config import load_vehicle
from hopper.control.interface import OpenLoopController
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.scenario_glue import run_hop
from hopper.sim.engine import Simulation
from hopper.types import Command, make_state


def run_mission(cfg, args) -> None:
    t0 = time.perf_counter()
    res, s = run_hop(
        cfg,
        seed=args.seed,
        command_delay=args.delay,
        wind_mean=tuple(args.wind),
        log_decimation=1,
        perfect_state=not args.ekf,
    )
    elapsed = time.perf_counter() - t0
    print(f"outcome: {s.outcome.name} at t = {s.t_end:.2f} s (seed {res.seed})")
    print(f"apogee:            {s.apogee:.2f} m")
    print(f"landing error:     {s.landing_error:.3f} m")
    print(f"touchdown speed:   {s.touchdown_speed:.2f} m/s")
    print(f"touchdown tilt:    {np.degrees(s.touchdown_tilt):.2f} deg")
    print(f"time of flight:    {s.flight_time:.2f} s")
    print(f"fuel used:         {s.fuel_used:.2f} kg")
    print(f"{len(res.log)} steps in {elapsed:.1f} s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_frame(res.log, args.out)
        print(f"log written to {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vehicle_default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="parquet path for the log")
    ap.add_argument("--mission", choices=["hop"], default=None, help="fly a closed-loop mission")
    ap.add_argument("--ekf", action="store_true", help="fly on sensors and the EKF estimate")
    ap.add_argument("--delay", type=float, default=0.0, help="command delay (s)")
    ap.add_argument("--wind", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("E", "N", "U"))
    args = ap.parse_args()

    cfg = load_vehicle(args.config)
    if args.mission == "hop":
        run_mission(cfg, args)
        return
    foot_height = mass_properties(cfg.tank.prop_mass_initial, cfg).com_z + cfg.legs.length
    x0 = make_state([0, 0, foot_height], [0, 0, 0], None, [0, 0, 0], cfg.tank.prop_mass_initial)
    # Short burn just above hover thrust, then coast and fall back.
    burn = Command(0.45, np.zeros(2), 0.0)
    off = Command(0.0, np.zeros(2), 0.0)
    controller = OpenLoopController(schedule=[(0.0, burn), (0.8, off)])

    sim = Simulation(cfg, controller, Environment.default(), x0, np.random.default_rng(args.seed))
    t0 = time.perf_counter()
    res = sim.run()
    elapsed = time.perf_counter() - t0

    last = res.log.iloc[-1]
    print(f"outcome: {res.outcome.name} at t = {res.t_end:.3f} s (seed {res.seed})")
    print(
        f"final pos = ({last.pos_x:.3f}, {last.pos_y:.3f}, {last.pos_z:.3f}) m, "
        f"vel = ({last.vel_x:.3f}, {last.vel_y:.3f}, {last.vel_z:.3f}) m/s, "
        f"propellant = {last.prop:.3f} kg"
    )
    print(f"{len(res.log)} steps in {elapsed:.2f} s ({len(res.log) / elapsed:.0f} steps/s)")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_frame(res.log, args.out)
        print(f"log written to {args.out}")


if __name__ == "__main__":
    main()
