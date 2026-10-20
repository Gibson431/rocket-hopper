"""Plot a logged run and optionally replay it in Meshcat.

    python scripts/plot_run.py runs/test.parquet [--replay] [--html out.html] [--out figs/]

`--replay` plays the run in a Meshcat browser view (open the printed URL), `--html` writes a static
animation that needs no server, `--out` is the figure directory (default: next to the log).
"""

import argparse
from pathlib import Path

from hopper.analysis.logger import Logger
from hopper.analysis.plots import save_all
from hopper.config import load_vehicle

DEFAULT_VEHICLE = Path(__file__).parent.parent / "configs" / "vehicle_default.yaml"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", help="parquet log from run_scenario.py --out")
    ap.add_argument("--out", default=None, help="figure directory (default: <log stem>_figs)")
    ap.add_argument("--replay", action="store_true", help="play the run in Meshcat")
    ap.add_argument("--html", default=None, help="write a static Meshcat HTML animation")
    ap.add_argument("--speed", type=float, default=1.0, help="replay speed (1 = real time)")
    ap.add_argument("--vehicle", default=str(DEFAULT_VEHICLE), help="vehicle config for the geometry")
    args = ap.parse_args(argv)

    log = Logger.load(args.log)
    out = Path(args.out) if args.out else Path(args.log).with_suffix("").with_name(
        Path(args.log).stem + "_figs"
    )
    for path in save_all(log, out):
        print(f"wrote {path}")
    if args.replay or args.html:
        from hopper.drake.viz import record_replay, replay

        cfg = load_vehicle(args.vehicle)
        if args.html:
            record_replay(log, cfg, args.html)
            print(f"wrote {args.html}")
        if args.replay:
            rp = replay(log, cfg, speed=args.speed)
            print(f"replayed in Meshcat at {rp.meshcat.web_url()}")
            input("press Enter to close the viewer ")


if __name__ == "__main__":
    main()
