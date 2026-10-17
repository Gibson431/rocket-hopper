# Rocket Hopper Simulator

Python 6-DOF simulator for developing a thrust-vector-controlled (TVC) rocket-hopper flight controller.

## Install and test

```
mise install        # provides uv, if missing
uv sync
uv run pytest
uv run ruff check . && uv run ruff format --check .
```

## Development

The project is built step by step. Start with [docs/AGENT_START.md](docs/AGENT_START.md);
conventions are in [docs/CONVENTIONS.md](docs/CONVENTIONS.md) and status in [docs/PROGRESS.md](docs/PROGRESS.md).
