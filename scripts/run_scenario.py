"""Run one scenario file and print the hop summary.

    python scripts/run_scenario.py configs/scenarios/hop_nominal.yaml [--seed S] [--dispersed] [--meshcat]

`--seed` overrides the scenario seed. `--dispersed` applies the scenario's Monte Carlo dispersions,
so a failed Monte Carlo seed is reproduced with `--seed <seed> --dispersed`.
`--meshcat` publishes the run to a live Meshcat view (open the printed URL); the script waits for
Enter before exiting so the view stays up.
"""

import argparse
import time
from pathlib import Path

import numpy as np

from hopper.analysis.logger import save_frame
from hopper.scenario import load_scenario, run_scenario


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dispersed", action="store_true")
    ap.add_argument("--out", default=None, help="parquet path for the log")
    ap.add_argument("--full-log", action="store_true", help="log every physics step")
    ap.add_argument("--meshcat", action="store_true", help="show the run in a live Meshcat view")
    args = ap.parse_args()

    meshcat = None
    if args.meshcat:
        from pydrake.all import Meshcat

        meshcat = Meshcat()
        print(f"Meshcat live view at {meshcat.web_url()}")
        input("open the URL, then press Enter to start the run ")

    scenario = load_scenario(args.scenario)
    t0 = time.perf_counter()
    run = run_scenario(
        scenario, args.seed, args.dispersed, log_decimation=1 if args.full_log else 10,
        meshcat=meshcat,
    )
    elapsed = time.perf_counter() - t0
    s, res = run.summary, run.result
    print(f"scenario {scenario.name}: {s.outcome.name} at t = {s.t_end:.2f} s (seed {run.seed})")
    if run.dispersions:
        print("dispersions: " + ", ".join(f"{k}={v:.4g}" for k, v in run.dispersions.items()))
    print(f"apogee:            {s.apogee:.2f} m")
    print(f"landing error:     {s.landing_error:.3f} m")
    print(f"touchdown speed:   {s.touchdown_speed:.2f} m/s vertical, {s.touchdown_hspeed:.2f} m/s horizontal")
    print(f"touchdown tilt:    {np.degrees(s.touchdown_tilt):.2f} deg (max {np.degrees(s.max_tilt):.1f})")
    print(f"max gimbal:        {np.degrees(s.max_gimbal):.2f} deg")
    print(f"time of flight:    {s.flight_time:.2f} s")
    print(f"fuel used:         {s.fuel_used:.2f} kg")
    print(f"{len(res.log)} rows in {elapsed:.1f} s")
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        save_frame(res.log, args.out)
        print(f"log written to {args.out}")
    if meshcat is not None:
        input("press Enter to close the viewer ")


if __name__ == "__main__":
    main()
