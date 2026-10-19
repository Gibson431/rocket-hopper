# S21: Visualisation: Meshcat replay and log plots

**Depends on:** S18 (richer after S19-S20). **Effort:** 2 days. Replaces the scrapped matplotlib
"plots and 3D replay" step.

## Context

Drake gives a 3D viewer for free: Meshcat, driven by `SceneGraph` geometry already created in S14/S17.
Use it instead of a hand-written matplotlib animation. Plots of logged signals stay in matplotlib,
which is already a dependency. Read `docs/DRAKE_DESIGN.md`, `src/hopper/analysis/logger.py`, and
`docs/CONVENTIONS.md` (log schema).

## Deliverables

### `src/hopper/drake/viz.py`
- Visual geometry for the plant: body cylinder, nose cone, legs, ground plane (attached in
  `build_vehicle_plant`, no collision role).
- `replay(log, cfg, speed=1.0, meshcat=None)`: plays a logged run in Meshcat using the logged pose
  (position, quaternion, CoM offset from logged mass), with a thrust plume scaled by logged thrust and
  gimbal angle, a trail of past positions, and a text overlay of time, phase and outcome.
- `record_replay(log, cfg, path)`: writes a self-contained Meshcat HTML (static) so a run can be
  shared without a server.
- Live mode: `DrakeSimulation(..., meshcat=m)` publishes while running (off by default; never in
  tests or Monte Carlo).

### `src/hopper/analysis/plots.py`
Functions taking a log DataFrame and returning matplotlib `Figure`s (never `plt.show()`):
`plot_overview` (position, velocity, attitude Euler for display only, body rates, throttle and thrust,
gimbal actual vs commanded; shaded phases when a `mission_state` column exists; fault times; outcome
marker), `plot_trajectory`, `plot_control`, `plot_estimation` (+/-3 sigma from `est_*`/`sig_*` columns),
`plot_mc_summary`, and `save_all(log, out_dir)`.

### CLI
`python scripts/plot_run.py runs/test.parquet [--replay] [--html out.html] [--out figs/]`.

## Tests (`tests/analysis/test_plots.py`, `tests/drake/test_viz.py`)

- With the Agg backend each plot function returns a `Figure` with the expected axes count, on a short
  synthetic log and a real run, without error when optional columns are missing.
- `save_all` writes the expected files.
- Replay: poses sent to Meshcat for a 2 s log equal the logged poses (read back from the plant's
  `SceneGraph` query or the Meshcat scene tree) to 1e-9; `record_replay` writes a non-empty HTML file.
  No browser or server is needed in CI.
- Visual geometry does not change dynamics: a run with and without it is bit-identical.

## Definition of done

- Tests pass; ruff clean; the CLI produces figures and an HTML replay for a real hop (give example paths
  in the progress notes).
- Progress tracker updated per `docs/AGENT_START.md`.
