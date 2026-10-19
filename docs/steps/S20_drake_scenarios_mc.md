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

- All provided scenarios load and run for 1 s on the Drake backend.
- `sample_dispersions` is unchanged and a pure function of the seed.
- Reproducibility: the same base seed gives an identical summary for `workers=1` and `workers=4`; a
  failed seed reruns singly with `run_scenario.py --seed S --dispersed` and reproduces its outcome.
- Baselines (16 seeds each, EKF, noise only; mark `slow`; record the result, do not compare with the
  legacy numbers): `hop_nominal`, `hop_wind`, `hop_gps_outage`, `hop_gimbal_fault` are expected to
  land, `hop_gimbal_stuck` to crash. Investigate a controller regression if not, and do not tune the
  simulator to make it pass.
- `hop_dispersed` (100 seeds, mark `slow`): record the success rate and Wilson interval. The known
  gyro-bias problem in the S11 backlog is a controller/estimator issue and is not fixed here.

## Definition of done

- Tests pass (including the `slow` ones, run once and recorded); ruff clean.
- Record Drake MC throughput (runs/min/worker) on an idle machine in the progress notes. If a 100-run
  set takes more than an hour on the available cores, add the profile and a Backlog item.
- Progress tracker updated per `docs/AGENT_START.md`.
