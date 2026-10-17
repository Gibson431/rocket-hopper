# S11: Scenarios, faults, Monte Carlo

**Depends on:** S09 (S10 optional but recommended; scenarios should work in either mode). **Effort:** 2-3 days.

## Context

Make runs declarative and repeatable, inject faults, and run many dispersed simulations in parallel for
pass/fail statistics. Read `docs/CONVENTIONS.md`, `src/hopper/sim/engine.py`,
`src/hopper/control/mission.py`, and `scripts/run_scenario.py`.

## Deliverables

### Scenario schema (`src/hopper/scenario.py`, pydantic, YAML in `configs/scenarios/`)
```yaml
name: hop_nominal
vehicle: configs/vehicle_default.yaml
seed: 0
estimation: ekf            # or perfect
t_max: 60
initial: {pos: [0, 0, 0], vel: [0, 0, 0], tilt_deg: 0, yaw_deg: 0}
mission: {target_alt: 10, hover_time: 3, target_xy: [3, 0], ...}
wind: {mean: [2, 0, 0], gust_sigma: 1.0, gust_tau: 3.0}
command_delay: 0.02
faults:
  - {type: gimbal_stuck, axis: y, at: 5.0, value_deg: 2.0}
  - {type: thrust_scale, at: 8.0, factor: 0.8}
  - {type: gps_outage, start: 4.0, duration: 5.0}
  - {type: gyro_bias_step, at: 6.0, value_dps: 0.5}
  - {type: engine_out, at: 7.0}
  - {type: gimbal_slow, at: 3.0, rate_factor: 0.3}
dispersions:                # used only by Monte Carlo
  dry_mass: {dist: normal, sigma_frac: 0.03}
  com_z: {dist: normal, sigma: 0.02}
  thrust_scale: {dist: normal, sigma_frac: 0.03}
  wind_mean: {dist: uniform, low: [-4, -4, 0], high: [4, 4, 0]}
  gyro_bias: {dist: normal, sigma_dps: 0.1}
  initial_pos: {dist: normal, sigma: [0.2, 0.2, 0]}
```
Make the schema strict (reject unknown keys). `load_scenario(path)`, `build_simulation(scenario, seed)`
returns a ready `Simulation`.

### Fault injection (`src/hopper/sim/faults.py`)
Faults are time-triggered modifiers applied inside the engine at defined seams: command modification
(gimbal stuck, engine out), propulsion parameter scaling (thrust scale, slower gimbal), and sensor modification
(GPS outage, bias step). Implement a small `Fault` base class with `apply_command`, `modify_params`,
`modify_sensors` hooks; the engine calls them each step. Keep the nominal path (no faults) free of overhead.

### Monte Carlo (`src/hopper/analysis/monte_carlo.py`)
- `run_monte_carlo(scenario_path, n, base_seed, workers, out_dir)` using `concurrent.futures.ProcessPoolExecutor`;
  per-run seed = `SeedSequence(base_seed).spawn(n)` so runs are independent and reproducible, independent of
  `workers`.
- Per run, apply dispersions then run, then compute metrics: outcome, landing position error, touchdown
  vertical/horizontal speed, max tilt, max gimbal use, fuel used, flight time, estimation RMS (when
  available).
- Results: `summary.parquet` (one row per run, including sampled dispersions and seed), plus optional
  per-run logs saved for failures only. Report success rate with a Wilson confidence interval, percentile
  tables, and the list of failed seeds so any failure is re-runnable with `run_scenario.py --seed`.
- CLI: `python scripts/run_monte_carlo.py configs/scenarios/hop_dispersed.yaml -n 500 --workers 8`.

### Provided scenarios (`configs/scenarios/`)
`hop_nominal.yaml`, `hop_wind.yaml`, `hop_gps_outage.yaml`, `hop_gimbal_fault.yaml`, `hop_dispersed.yaml`.

### `scripts/run_scenario.py` upgrade
Takes a scenario file instead of the built-in mission; `--seed` overrides.

## Tests (`tests/test_scenario.py`, `tests/sim/test_faults.py`, `tests/analysis/test_monte_carlo.py`)

- Valid scenario loads; unknown key raises; the provided scenarios all load.
- Fault timing: `gimbal_stuck` at t=5 holds the actual gimbal (in the log) from t=5; `thrust_scale` scales
  thrust by the factor; `engine_out` zeroes thrust; `gps_outage` yields `None` GPS fields in the window.
- No-fault scenario output is bit-identical to a run built directly through `Simulation` (overhead-free
  nominal path).
- Monte Carlo reproducibility: same base seed gives identical summary for `workers=1` and `workers=4`.
- Dispersions are applied (summary columns show the expected distribution moments for n=200, loose tolerance).
- Nominal hop: success rate of 100% over 20 runs with small dispersions (record actual rates).
- The failed seed reported by the MC can be re-run singly and reproduces the same outcome.

## Definition of done

- Tests pass; ruff clean; `scripts/run_monte_carlo.py` runs 100 runs end to end.
- Record the measured MC throughput (runs/min per worker) and the baseline success rates for each
  provided scenario in the progress notes.
- Progress tracker updated per `docs/AGENT_START.md`.
