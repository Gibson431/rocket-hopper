# S00: Project scaffolding

**Depends on:** nothing. **Effort:** 0.5 day.

## Context

Repo `rocket-hopper` builds a Python 6-DOF simulator for a TVC rocket hopper. Read
`docs/CONVENTIONS.md` first. This step creates the empty, working project skeleton so later steps
can add code and tests.

## Deliverables

- `pyproject.toml`: project `hopper`, Python >= 3.11, `src/` layout (hatchling or setuptools),
  runtime and dev dependencies from CONVENTIONS.md, ruff config (line length 100, rules `E,F,I,UP,B,N`),
  pytest config (`testpaths = ["tests"]`).
- `uv sync` creates the environment (`uv.lock` committed to the tree).
- Package skeleton with empty `__init__.py`: `src/hopper/{dynamics,sim,sensors,estimation,control,analysis}/`.
- `tests/` with `tests/test_smoke.py` importing `hopper` and every subpackage.
- `configs/` and `configs/scenarios/` directories (add `.gitkeep`).
- `.gitignore`: `.venv/`, `__pycache__/`, `.pytest_cache/`, `.ruff_cache/`, `runs/`, `*.parquet`, `.hypothesis/`.
- `.github/workflows/ci.yml`: on push/PR, set up uv, `uv sync`, `uv run ruff check .`,
  `uv run ruff format --check .`, `uv run pytest`.
- `README.md` (short): what the project is, how to install and test, pointer to `docs/AGENT_START.md`.
- `scripts/` already contains `next_step.py`; leave it alone.

## Notes

- uv is provided by mise (`.mise.toml` already lists `uv = "latest"`; run `mise install` if needed).
  Create the project with `uv init --package --name hopper` (or equivalent) and add dependencies only with
  `uv add` / `uv add --dev`, so `pyproject.toml` and `uv.lock` stay consistent.
- Python on this machine is 3.14. Use `uv python pin` only if a dependency has no wheel yet; otherwise
  leave the default.
- Do not write any simulation code here.

## Definition of done

- `uv sync && uv run pytest && uv run ruff check . && uv run ruff format --check .` all succeed.
- `python3 scripts/next_step.py` still runs.
- Progress tracker updated per `docs/AGENT_START.md`.
