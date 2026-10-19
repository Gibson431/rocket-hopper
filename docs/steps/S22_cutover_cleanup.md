# S22: Cutover to Drake and legacy removal

**Depends on:** S20, S21. **Effort:** 1-2 days.

## Context

Make Drake the only simulation backend and delete the hand-rolled one. Do this once the S18-S20 closed-loop
tests pass on the Drake backend. No comparison with the legacy sim is needed: Drake is the source of
truth. If those tests fail, do not delete anything: set the step `blocked` with what fails. Read
`docs/DRAKE_DESIGN.md` and the git history of the legacy modules.

## Deliverables

- Default backend is Drake; the `backend` scenario field and `--backend` flags are removed (or reduced
  to an error for `legacy`). Existing scenario YAML still loads.
- Delete the legacy simulation code: `dynamics/{rigid_body,propulsion,aero,contact,vehicle}.py`,
  `sim/{integrator,engine}.py`, the legacy `Simulation`, and their now-redundant tests. Keep what is
  still used: config, `mass_model` (used by `MassUpdater` and the controller glue),
  `environment.atmosphere`/ISA helpers if reused, `sim/events.py` if S17 reused it, sensors,
  `sim/faults.py` if S19 reused it, `types`, `math3d`, `analysis/logger.py`. Delete only what a search
  shows is unreferenced; move anything kept into its final home rather than leaving a dead package.
- Tests that exercised the legacy simulator are deleted with it; tests of flight software stay.
  Nothing imports from a deleted module.
- Docs: `docs/PLAN.md` architecture and steps table, `docs/CONVENTIONS.md` (tooling: `drake` dependency,
  Drake package layout, integrator and body-frame rules), `docs/AGENT_START.md` rules (no RK4 wording,
  `pydrake` import ban for flight software), `README.md`, CI.
- Boundary test: `hopper.control`, `hopper.estimation`, `hopper.types`, `hopper.math3d` import neither
  `pydrake` nor any simulator module (allowlist form, as in the S08 backlog suggestion).
- Final baselines table (outcomes, key metrics, throughput) recorded in `docs/PROGRESS.md`.

## Tests

- Whole suite passes with the legacy modules gone.
- `scripts/run_scenario.py` and `scripts/run_monte_carlo.py` run end to end for a provided scenario.
- Import-boundary and "no references to deleted modules" tests.

## Definition of done

- `uv run pytest` and `uv run ruff check .` and `uv run ruff format --check .` pass; no dead code or
  stale docs referring to the hand-rolled integrator remain (search for `rk4_step`, `vehicle_derivative`,
  `contact_loads`, "fixed-step RK4").
- Progress tracker updated per `docs/AGENT_START.md`. Do not commit unless the user asks; the deletion
  is easiest to review as its own commit.
