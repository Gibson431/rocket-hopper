# S12: Plots and 3D replay

**Depends on:** S06 (richer once S09-S11 exist). **Effort:** 2 days.

## Context

Make runs inspectable. Read `docs/CONVENTIONS.md` (logging schema) and `src/hopper/analysis/logger.py`.
Consumers: the developer debugging a controller, and Monte Carlo reports. Keep matplotlib as the
only hard dependency; Rerun is optional (do not add it unless a quick spike shows it is worthwhile,
and log the decision).

## Deliverables

### `src/hopper/analysis/plots.py`
Functions taking a log DataFrame (and returning matplotlib `Figure`s; they never call `plt.show()`):
- `plot_overview(log)`: 3x2 grid: position (xyz), velocity, attitude (Euler, display only), body rates,
  throttle and thrust, gimbal (actual vs commanded). Shaded spans for mission phases when a
  `mission_state` column exists, vertical lines for fault times, a marker for the final outcome.
- `plot_trajectory(log)`: x-z and x-y projections with the ground and leg-contact intervals marked.
- `plot_control(log)`: commanded vs actual, saturation highlights (throttle at limits, gimbal at limit/rate limit).
- `plot_estimation(log)`: estimate error with +/-3 sigma bands (when `est_*`/`sig_*` columns exist).
- `plot_mc_summary(summary_df)`: landing-error scatter, touchdown-speed histogram, outcome pie/bar, failure
  seeds annotated.
- `save_all(log, out_dir)` writes PNGs for all of the above.

### `src/hopper/analysis/viz.py`
- `replay(log, cfg, speed=1.0)`: matplotlib 3D animation of the vehicle (body cylinder, nose cone, legs,
  thrust plume scaled by thrust, trail of past positions, ground grid, text overlay with time/phase/outcome).
  Rendering uses the logged quaternion and CoM offset.
- `export_replay(log, cfg, path, fps=30)` writes mp4/gif if ffmpeg/pillow is available, otherwise
  prints a clear message.
- Subsample the log to the frame rate to keep rendering light.

### CLI
`python scripts/plot_run.py runs/test.parquet [--replay] [--out figs/]`.

## Tests (`tests/analysis/test_plots.py`)

- Using a short synthetic log (and one from a real S06 run), each plot function returns a `Figure` with the
  expected number of axes, with the Agg backend, no exceptions with missing optional columns.
- Replay: constructs frames for a 2 s log without error (render a single frame to a buffer and check
  that it is nonempty); do not require ffmpeg in CI.
- `save_all` writes the expected files.

## Definition of done

- Tests pass; ruff clean; the CLI produces figures for a real run (include one example image path in the progress notes).
- Progress tracker updated per `docs/AGENT_START.md`.
