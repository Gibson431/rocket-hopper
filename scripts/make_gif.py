"""Render a scenario to a GIF of the Meshcat view (used for the README).

    uv run --with playwright python -m playwright install firefox   # once
    uv run --with playwright scripts/make_gif.py SCENARIO.yaml docs/media/hop_nominal.gif

Runs the scenario, replays the log in a live Meshcat server, screenshots each frame in headless
Firefox and assembles them with ffmpeg. Playwright is not a project dependency, hence `--with`.
"""

import argparse
import subprocess
import tempfile
import time
from pathlib import Path

from pydrake.all import Meshcat

from hopper.config import load_vehicle
from hopper.drake.viz import Replay, frame_indices
from hopper.scenario import load_scenario, run_scenario

CAMERA = ([16, -16, 7], [1.5, 0, 5])  # position, target (m)
VIEWPORT = 640
CROP = "crop=420:420:110:90"  # drops the Meshcat stats widget and empty margin
SETTLE = 0.12  # s for the browser to draw a frame


def capture_frames(log, cfg, out: Path, fps: float) -> int:
    from playwright.sync_api import sync_playwright

    meshcat = Meshcat()
    rp = Replay(cfg, meshcat)
    meshcat.SetCameraPose(*CAMERA)
    with sync_playwright() as p:
        browser = p.firefox.launch()
        page = browser.new_page(viewport={"width": VIEWPORT, "height": VIEWPORT})
        page.goto(meshcat.web_url() + "?tracking=false")
        time.sleep(3)
        page.add_style_tag(content=".dg, #stats, .main {display:none !important}")
        rp.show(log, 0)
        meshcat.SetCameraPose(*CAMERA)  # the browser resets it on load
        time.sleep(1)
        n = 0
        for n, k in enumerate(frame_indices(log["t"].to_numpy(float), fps), 1):
            rp.show(log, k)
            time.sleep(SETTLE)
            page.screenshot(path=str(out / f"f{n - 1:04d}.png"))
        browser.close()
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scenario")
    ap.add_argument("out", help="output .gif path")
    ap.add_argument("--fps", type=float, default=12.0)
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()

    scenario = load_scenario(args.scenario)
    run = run_scenario(scenario, args.seed, False, log_decimation=10)
    cfg = load_vehicle(scenario.vehicle_path())
    log = run.result.log
    print(f"{scenario.name}: {run.summary.outcome.name}, {len(log)} rows")

    with tempfile.TemporaryDirectory() as tmp:
        frames = Path(tmp)
        n = capture_frames(log, cfg, frames, args.fps)
        print(f"{n} frames")
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        palette = "palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer:bayer_scale=4"
        vf = f"{CROP},split[a][b];[a]{palette}"
        subprocess.run(
            [
                "ffmpeg",
                "-loglevel",
                "error",
                "-y",
                "-framerate",
                str(args.fps),
                "-i",
                str(frames / "f%04d.png"),
                "-vf",
                vf,
                "-loop",
                "0",
                args.out,
            ],
            check=True,
        )
    print(f"wrote {args.out} ({Path(args.out).stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
