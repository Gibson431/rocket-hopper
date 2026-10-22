# Rocket Hopper Simulator

Python 6-DOF simulator for developing a thrust-vector-controlled (TVC) rocket-hopper flight controller.
The simulation runs on [Drake](https://drake.mit.edu/) (`pydrake`); the controller and estimator are
plain Python that never import it.

## Layout

uv workspace. The flight software is split into standalone packages under `packages/`; the root
`hopper` package is the simulator and the only place they are composed.

| Package | Import | Role |
|---|---|---|
| `packages/hopper-core` | `hopper_core` | shared `types` and `math3d` |
| `packages/hopper-control` | `hopper_control` | attitude/position/mission controller |
| `packages/hopper-estimation` | `hopper_estimation` | EKF and propellant estimator |
| root | `hopper` | Drake simulation, sensors, scenarios, analysis |

## Install and test

```
mise install        # provides uv, if missing
uv sync
uv run pytest                 # unit tests (fast)
uv run pytest -m ""           # everything, including end-to-end simulation runs
uv run ruff check . && uv run ruff format --check .
```

## Run

```
uv run python scripts/run_scenario.py configs/scenarios/hop_nominal.yaml [--seed S] [--dispersed]
uv run python scripts/run_monte_carlo.py configs/scenarios/hop_dispersed.yaml -n 100 --workers 4
uv run python scripts/plot_run.py LOG.parquet [--replay]     # plots and Meshcat replay
```

## Development

The project is built step by step. Start with [docs/AGENT_START.md](docs/AGENT_START.md);
conventions are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md) and status in [docs/PROGRESS.md](docs/PROGRESS.md).
