# S18: Closed-loop diagram, controller host, delay and logging

**Depends on:** S15, S16, S17. **Effort:** 3 days.

## Context

Assemble the S14-S17 pieces into a runnable diagram with the same behaviour as
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
- Command delay quantised to `round(delay / dt)` steps, same semantics as legacy (delay 0 applies in
  the same step; commands held between calls). **Put it inside `ControllerHost`** (a command queue in
  the host's state, output read from it): a separate delay system would read the pre-update value and
  add one physics step of latency (S12 question 9a). If S12 found a different arrangement, follow it.
  `hopper/drake/delay.py` then holds only the queue helper.
- `hopper/drake/logging.py`: records on the physics grid with the legacy `log_decimation` and builds
  the exact S06 DataFrame (same columns, the terminal row always kept, `df.attrs` seed and config hash).
- `hopper/drake/simulation.py`: `DrakeSimulation` with the same constructor arguments and
  `run(t_end=None) -> SimResult` as `Simulation`, so callers and `scenario_glue` can switch backend.
  `env.wind.advance` equivalent is handled inside the wind system.

## Tests (`tests/drake/test_closed_loop.py`)

- Interface: `DrakeSimulation` and `Simulation` accept the same arguments; result has identical
  columns and dtypes; terminal row and `event` semantics as in `tests/sim/test_engine.py`
  (port those tests parametrised over both backends).
- Multi-rate: a 10 s run at 100 Hz makes exactly 1000 controller calls; the controller is not called
  on the terminal row; delay of k steps applies the command k steps later; command is constant
  across integrator stages.
- Determinism: same seed gives a bit-identical log; different seeds differ with wind on.
- Open-loop parity on S13 goldens 1-6 through the whole diagram.
- Closed loop: the perfect-state hop (S09 `run_hop`, seed 0) ends LANDED with landing error and
  touchdown speed within the S12 parity tolerances of the legacy numbers (err 0.002 m, 0.75 m/s,
  flight ~22.6 s, fuel ~11.1 kg), also with 20 ms delay and 3 m/s wind.
- Speed: record steps/s and a full perfect-state hop wall time against legacy (7.5 s per 10 s run).
  Fail below the budget agreed in S12 (expected: parity with legacy, not a speedup).
- Delay 0 versus legacy: the command computed at step k acts on the same step (no extra latency).

## Definition of done

- Tests pass; ruff clean; the speed budget met or the shortfall recorded as a blocker for S20.
- Progress tracker updated per `docs/AGENT_START.md`; document the diagram and port names in
  `docs/CONVENTIONS.md`.
