"""Minimal v1 scenario runner: a built-in open-loop hop. S11 replaces it with scenario files."""

import argparse
import time
from pathlib import Path

import numpy as np

from hopper.analysis.logger import save_frame
from hopper.config import load_vehicle
from hopper.control.interface import OpenLoopController
from hopper.dynamics.environment import Environment
from hopper.dynamics.mass_model import mass_properties
from hopper.sim.engine import Simulation
from hopper.types import Command, make_state


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/vehicle_default.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="parquet path for the log")
    args = ap.parse_args()

    cfg = load_vehicle(args.config)
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
