# S23: Split flight software into workspace packages

**Depends on:** S22. **Effort:** 0.5-1 day.

## Context

The controller and estimator must be usable on their own (and portable later), so each becomes its
own uv workspace member. Neither may depend on the simulator or on the other. The shared `types` and
`math3d` modules go to a small core package that both import. The root `hopper` package stays the
simulator and is the only place the pieces are composed (`estimated_controller.py`, `scenario_glue.py`,
closed-loop tests).

## Deliverables

| Package (dir) | Import name | Contents | Dependencies |
|---|---|---|---|
| `packages/hopper-core` | `hopper_core` | `types.py`, `math3d.py` | numpy |
| `packages/hopper-control` | `hopper_control` | old `hopper/control/*` | hopper-core, numpy |
| `packages/hopper-estimation` | `hopper_estimation` | old `hopper/estimation/*` | hopper-core, numpy, scipy |
| root | `hopper` | simulator, scenarios, analysis | the three above + drake etc. |

- Root `pyproject.toml` declares a `[tool.uv.workspace]` and `[tool.uv.sources]` for the members.
- Every import of the moved modules is rewritten (`hopper.types` -> `hopper_core.types`,
  `hopper.control.x` -> `hopper_control.x`, ...). No compatibility shims.
- Unit tests that need only one package move into it (`packages/<pkg>/tests`). Tests that need the
  simulator (closed loop, scenario glue, Drake) stay in the root `tests/`.
- Boundary test becomes per package: each package imports only stdlib, its allowed third-party
  libraries, itself and (control, estimation) `hopper_core`. Control and estimation never import each other.
- Docs (`CONVENTIONS.md`, `AGENT_START.md`, `PLAN.md`, `README.md`, `PROGRESS.md`) and CI updated.

## Tests

- Each package's own tests pass when run from its directory, with the sim package not importable by it.
- Whole-workspace `uv run pytest -m ""`, `uv run ruff check .`, `uv run ruff format --check .` pass.

## Definition of done

- All of the above, no references to `hopper.control`, `hopper.estimation`, `hopper.types`,
  `hopper.math3d` anywhere (code or docs), and a reviewer PASS.
