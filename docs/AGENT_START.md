# Agent start guide

You may be a fresh agent with no memory of earlier sessions. Everything you need is in `docs/`.

## Workflow

1. Read `docs/CONVENTIONS.md` (frames, units, state layout, interfaces). It is short and binding.
2. Find your step:
   - If the user named a step (e.g. "do S03"), use that one.
   - Otherwise run `python3 scripts/next_step.py` (stdlib only, works before the project is installed).
     It prints the next `todo` step whose dependencies are all `done`, and any step left `in_progress`.
     An `in_progress` step means a previous session stopped midway: inspect the repo and its notes
     in `docs/PROGRESS.md`, then resume it instead of starting a new one.
3. Read `docs/steps/<ID>_*.md`. Each step file is self-contained: context, prerequisites, files to
   create, interfaces, tests, and a definition of done. You should not need earlier chat history.
4. Check the prerequisites actually hold (the listed files exist, `pytest` passes). If a dependency
   is marked `done` but is broken, fix it or mark it `blocked` with a note. Do not silently work around it.
5. **Start protocol:** set the step's status to `in_progress` in `docs/PROGRESS.md` and set the
   "Updated" date.
6. Read `skills/test-driven-developer/SKILL.md` and follow it: this project is developed test-first
   (red, green, refactor). Its examples are TypeScript/Jest; translate them to Python/pytest
   (plain `assert`, `pytest.approx`, `pytest.raises`, `hypothesis` for property tests). Write the failing
   tests from the step's "Tests" section first, watch them fail, then implement. Mocking is rarely needed
   here: prefer real physics objects and analytic expected values, and only fake slow or random
   collaborators.
   Implement within the step's scope. If you discover
   something out of scope, add a line under "Backlog" in `docs/PROGRESS.md` instead of doing it.
7. **Finish protocol** (only when every item in the step's "Definition of done" is met):
   - `uv run pytest` and `uv run ruff check .` pass (see S00 for setup; before S00 is done,
     these do not exist yet).
   - **Independent review:** follow `docs/REVIEW.md` Part 1. Spawn a fresh subagent that checks your
     work against the step file with no knowledge of your session. The step cannot be marked `done`
     until the reviewer returns `VERDICT: PASS`.
   - Set status to `done`, fill in "Updated" and a one-line note (what exists, anything the next step
     must know).
   - Append an entry to the "Session log" in `docs/PROGRESS.md`.
   - Record any decision that deviates from the plan or fixes an unspecified detail under
     "Decisions" in `docs/PROGRESS.md`, and update `docs/CONVENTIONS.md` if an interface changed.
   - Do not commit unless the user asked you to.
8. If you cannot finish: set the status to `blocked` (needs a human decision or an upstream fix) or
   leave it `in_progress`, and write what is done and what remains in the Notes column and the log.

## Rules

- `docs/PROGRESS.md` is the single source of truth for status. Never mark a step `done` without
  passing tests **and** a `PASS` from the independent reviewer (`docs/REVIEW.md`).
- If you are the reviewer subagent, follow `docs/REVIEW.md` Part 2 only: you are read-only and must
  not touch `docs/PROGRESS.md` or start implementing anything.
- The flight-software packages live in `packages/` and never import `hopper` (the simulator),
  `pydrake`, pandas or each other except through `hopper_core`: `hopper_control` and
  `hopper_estimation` may import only the standard library, numpy (scipy for estimation) and
  `hopper_core` (`types`, `math3d`). This keeps them portable to Rust or C later and prevents cheating
  with true state. `tests/test_import_boundary.py` enforces it as an allowlist. They are composed only
  in the root `hopper` package (`estimated_controller.py`, `scenario_glue.py`).
- Use `uv` for all environment and dependency work (`uv add`, `uv sync`, `uv run`); never `pip` or a
  hand-made venv. See CONVENTIONS.md "Tooling".
- SI units everywhere. No hidden unit conversions.
- All randomness goes through a `numpy.random.Generator` passed in explicitly. Runs with the same
  seed must be bit-reproducible.
- Keep the simulation numerically honest: fixed-step integration only (Drake RK3 at `cfg.sim.dt`,
  see `drake/sim.py`). Never adaptive stepping, in Monte Carlo runs or anywhere else.
- The simulator is not the product: Drake is the source of truth. Do not write unit tests for the
  simulator or A/B comparisons against another simulator. Test the controller and
  estimator; for simulator code a smoke test (builds, runs, finite, deterministic) is enough.
- Match the style of existing code. Keep comments short and explain why, not what.
