"""Run a dispersed scenario many times in parallel and print pass/fail statistics.

    python scripts/run_monte_carlo.py configs/scenarios/hop_dispersed.yaml -n 500 --workers 8
"""

import argparse
import os
import time

from hopper.analysis.monte_carlo import run_monte_carlo


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario")
    ap.add_argument("-n", type=int, default=100, help="number of runs")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--out", default=None, help="directory for summary.parquet and failure logs")
    args = ap.parse_args()

    t0 = time.perf_counter()
    mc = run_monte_carlo(args.scenario, args.n, args.base_seed, args.workers, args.out)
    elapsed = time.perf_counter() - t0
    print(mc.report())
    per_min = args.n / elapsed * 60 / max(1, min(args.workers, args.n))
    print(f"\n{args.n} runs in {elapsed:.0f} s on {args.workers} workers ({per_min:.1f} runs/min/worker)")
    if args.out:
        print(f"summary written to {args.out}/summary.parquet")


if __name__ == "__main__":
    main()
