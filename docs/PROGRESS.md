# Progress tracker

Source of truth for step status. Update it at the start and end of every step
(see `docs/AGENT_START.md`). `python3 scripts/next_step.py` parses the table below, so keep its format:
one row per step, six columns, status one of `todo`, `in_progress`, `done`, `blocked`.

Dependencies are in the "Depends" column (comma-separated step IDs, or `-`).

| ID | Title | Status | Depends | Updated | Notes |
|----|-------|--------|---------|---------|-------|
| S00 | Project scaffolding | done | - | 2026-10-17 | uv project (uv_build backend), py>=3.12, empty subpackages, smoke test, CI. Reviewed: PASS (round 2) |
| S01 | Math, types, config | in_progress | S00 | 2026-10-17 | |
| S02 | Rigid-body dynamics and RK4 | todo | S01 | - | |
| S03 | Mass model and propulsion | todo | S02 | - | |
| S04 | Environment and aerodynamics | todo | S03 | - | |
| S05 | Ground contact and events | todo | S04 | - | |
| S06 | Simulation engine, logging, delays | todo | S05 | - | |
| S07 | Sensor models | todo | S06 | - | |
| S08 | Control interface and attitude controller | todo | S06 | - | |
| S09 | Position cascade and mission state machine | todo | S08 | - | |
| S10 | State estimator (EKF) | todo | S07, S09 | - | |
| S11 | Scenarios, faults, Monte Carlo | todo | S09 | - | |
| S12 | Plots and 3D replay | todo | S06 | - | |
| S13 | Advanced control (LQR, MPC, Gym) | todo | S11 | - | |

## Decisions

Record choices that fix an unspecified detail or deviate from the plan. Newest last.
Format: `YYYY-MM-DD, step: decision. Reason.`

- 2026-10-17, S00: `requires-python` is `>=3.12`, not `>=3.11`. Current numpy (2.5.x) needs 3.12+, so uv cannot resolve for 3.11. CONVENTIONS.md and the S00 step file updated.
- 2026-10-17, S00: build backend is `uv_build` (default of `uv init --package`), not hatchling/setuptools; the step allows an equivalent.
- 2026-10-17, S00: ruff `extend-exclude = ["scripts", "docs", "skills"]`. `scripts/next_step.py` must stay untouched and ruff would otherwise lint it and reformat embedded Python in markdown.
- 2026-10-17, S00: no `.python-version` pin (uv init created one; removed per the step notes). No dependency lacks a 3.14 wheel.

## Backlog

Out-of-scope ideas found while working. Do not do these inside another step.

- Smoke test: also assert each subpackage has a real `__init__.py` (`__file__ is not None`), since a deleted one still imports as a namespace package.

## Session log

Append one entry per session that changes status. Newest last.
Format: `YYYY-MM-DD, step, status change, what was done, what the next agent should know.`

- 2026-10-17, plan: plan, conventions and step files written. No code yet.
- 2026-10-17, S00, todo -> done: scaffolding created (pyproject, uv.lock, subpackages, smoke test, CI, README). Reviewed: PASS (round 2; round 1 failed on unlogged decisions). Python floor is 3.12; run everything via `uv run`. Next: S01.
