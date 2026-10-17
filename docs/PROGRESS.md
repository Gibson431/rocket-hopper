# Progress tracker

Source of truth for step status. Update it at the start and end of every step
(see `docs/AGENT_START.md`). `python3 scripts/next_step.py` parses the table below, so keep its format:
one row per step, six columns, status one of `todo`, `in_progress`, `done`, `blocked`.

Dependencies are in the "Depends" column (comma-separated step IDs, or `-`).

| ID | Title | Status | Depends | Updated | Notes |
|----|-------|--------|---------|---------|-------|
| S00 | Project scaffolding | done | - | 2026-10-17 | uv project (uv_build backend), py>=3.12, empty subpackages, smoke test, CI. Reviewed: PASS (round 2) |
| S01 | Math, types, config | done | S00 | 2026-10-17 | math3d, types (slices POS/VEL/... and make_state), pydantic config (frozen, extra=forbid), configs/vehicle_default.yaml. Reviewed: PASS (round 2) |
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
- 2026-10-17, S01: config models use `extra="forbid", frozen=True`. Typos in YAML keys must fail loudly and configs must not be mutated mid-run.
- 2026-10-17, S01: `SimConfig` defaults are `dt=0.001`, `controller_rate=100` Hz, `max_time=60` s, and a validator rejects `controller_rate > 1/dt`. The step left defaults open; a controller faster than physics is meaningless.
- 2026-10-17, S01: `make_state` validates shapes (ValueError) and renormalises the quaternion; `Command.__post_init__` coerces `gimbal` to a float64 array of shape (2,). Keeps invalid state out of the integrator.
- 2026-10-17, S01: `Controller` and `StateFeedbackController` are `@runtime_checkable`, and `StateFeedbackController` extends `Controller`. Lets tests and the sim engine check structurally.
- 2026-10-17, S01: `rcs.max_roll_torque`, `legs.damping` and `legs.friction` allow 0 (strictly positive elsewhere), so RCS-less or frictionless variants can be configured.

## Backlog

Out-of-scope ideas found while working. Do not do these inside another step.

- S01 review suggestions (also: remove duplicate assertion in geometry test): test `quat_error` small-angle branch (n < 1e-12) and `dcm_to_quat` on non-unit input; add vehicle-length to config and check tank fits inside it; validate `Command.throttle` / `roll_torque` ranges (decide where in S03/S08).
- Smoke test: also assert each subpackage has a real `__init__.py` (`__file__ is not None`), since a deleted one still imports as a namespace package.

## Session log

Append one entry per session that changes status. Newest last.
Format: `YYYY-MM-DD, step, status change, what was done, what the next agent should know.`

- 2026-10-17, plan: plan, conventions and step files written. No code yet.
- 2026-10-17, S00, todo -> done: scaffolding created (pyproject, uv.lock, subpackages, smoke test, CI, README). Reviewed: PASS (round 2; round 1 failed on unlogged decisions). Python floor is 3.12; run everything via `uv run`. Next: S01.
- 2026-10-17, S01, todo -> done: math3d, types, config, default YAML, 65 tests. Reviewed: PASS (round 2; round 1 failed on unlogged decisions). State indices are slices (POS, VEL, QUAT, OMEGA, GIM, GIMR) or ints (PROP, THR). Next: S02.
