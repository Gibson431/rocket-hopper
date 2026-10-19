# S20: Scenarios and Monte Carlo on Drake

**Depends on:** S19. **Effort:** 2 days.

## Context

Make the Drake backend selectable everywhere scenarios and Monte Carlo run, and show it reproduces the
S11 statistics. Read `src/hopper/scenario.py`, `src/hopper/scenario_glue.py`,
`src/hopper/analysis/monte_carlo.py`, `scripts/run_scenario.py`, `scripts/run_monte_carlo.py`, and the
S11 Decisions in `docs/PROGRESS.md`.

## Deliverables

- Backend choice: `backend: legacy | drake` (scenario field, default `legacy` until S22) and a
  `--backend` flag on both scripts; `build_simulation` returns the matching object. No scenario YAML
  other than the new field changes.
- `run_hop` and the dispersion machinery work on both backends; dispersions (`dry_mass`, `com_z`,
  `thrust_scale`, wind, initial position, gyro bias) change only the true vehicle, applied to the Drake
  plant, engine and mass updater parameters, and are still a pure function of the run seed.
- Monte Carlo builds each Drake diagram inside the worker process (S12 finding), results independent
  of `workers`.
- `docs/` note comparing the two backends' baselines (below).

## Tests (`tests/test_scenario.py`, `tests/analysis/test_monte_carlo.py` parametrised over backends)

- All provided scenarios load and run for 1 s on both backends.
- `sample_dispersions` is unchanged; the same seed gives the same dispersed parameters on both backends.
- Reproducibility: same base seed gives an identical summary for `workers=1` and `workers=4`; a failed
  seed reruns singly with `run_scenario.py --seed S --dispersed --backend drake` and reproduces its outcome.
- Baselines (16 seeds each, EKF, noise only; mark `slow`): `hop_nominal`, `hop_wind`, `hop_gps_outage`,
  `hop_gimbal_fault` land 16/16, `hop_gimbal_stuck` 0/16 CRASHED, all as in S11.
- `hop_dispersed` (100 seeds, mark `slow`): success count within the Wilson interval of the legacy
  36/100 (27-46%). The known gyro-bias problem in the S11 backlog is **not** fixed here; the port
  must not hide or cure it. Also compare per-seed outcomes and report how many differ.
- Key metrics (landing error, touchdown speed, fuel) medians within the S12 tolerances of the legacy
  medians.

## Definition of done

- Tests pass (including the `slow` ones, run once and recorded); ruff clean.
- Record Drake MC throughput (runs/min/worker) on an idle machine next to legacy, with both numbers in
  the progress notes. If Drake is slower by more than 2x, add the profile and a Backlog item.
- Progress tracker updated per `docs/AGENT_START.md`.
