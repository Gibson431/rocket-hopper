# S18: Closed-loop diagram, controller host, delay and logging

**Depends on:** S15, S16, S17. **Effort:** 3 days.

## Context

Assemble the S14-S17 pieces into a runnable diagram with the same interface as
`src/hopper/sim/engine.py`: multi-rate controller, command delay, perfect-state mode, event
termination and the S06 log. Sensor mode and faults come in S19. Read `engine.py`,
`src/hopper/analysis/logger.py`, `src/hopper/control/interface.py`, `docs/DRAKE_DESIGN.md`.

## Deliverables

- `hopper/drake/diagram.py`: `build_diagram(cfg, controller, env, rng, command_delay, perfect_state)`
  wiring controller -> delay -> engine -> force combiner -> plant, aero, wind, contact observers and
  the mass updater. Python wiring only; no physics here.
- `hopper/drake/controller_host.py`: `ControllerHost(LeafSystem)` calls the existing `Controller` /
  `StateFeedbackController` objects unchanged at `controller_rate` via a periodic discrete update
  (state estimate built from the plant state in perfect-state mode). It owns `reset()` and the
  optional `log_columns`/`log_values()` hook.
- Command delay quantised to `round(delay / dt)` steps, as in `engine.py` (delay 0 applies in
  the same step; commands held between calls). **Put it inside `ControllerHost`** (a command queue in
  the host's state, output read from it): a separate delay system would read the pre-update value and
  add one physics step of latency (S12 question 9a). If S12 found a different arrangement, follow it.
  `hopper/drake/delay.py` then holds only the queue helper.
- `hopper/drake/logging.py`: records on the physics grid with `log_decimation` and builds
  the exact S06 DataFrame (same columns, the terminal row always kept, `df.attrs` seed and config hash).
- `hopper/drake/simulation.py`: `DrakeSimulation` with the same constructor arguments and
  `run(t_end=None) -> SimResult` as `Simulation`, so callers and `scenario_glue` can switch backend.
  `env.wind.advance` equivalent is handled inside the wind system.

## Tests (`tests/drake/test_closed_loop.py`)

- Interface: `DrakeSimulation.run` returns a `SimResult` whose log has the S06 columns; the terminal
  row carries the `event`.
- Multi-rate: a 10 s run at 100 Hz makes exactly 1000 controller calls; the controller is not called
  on the terminal row; a delay of k steps applies the command k steps later; delay 0 acts on the same
  step.
- Determinism: same seed gives a bit-identical log; different seeds differ with wind on.
- Closed loop: the perfect-state hop (S09 `run_hop`, seed 0) ends LANDED (landing error < 0.5 m,
  touchdown speed < 1 m/s, upright), also with 20 ms delay and 3 m/s wind. These are the S09
  acceptance thresholds, not comparisons with the legacy sim.
- Speed: record steps/s and the wall time of a perfect-state hop in the progress notes (no pass/fail
  threshold; flag it if a Monte Carlo run becomes impractical).

## Definition of done

- Tests pass; ruff clean; speed recorded in the progress notes.
- Progress tracker updated per `docs/AGENT_START.md`; document the diagram and port names in
  `docs/CONVENTIONS.md`.
