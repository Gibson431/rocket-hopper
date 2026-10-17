# S06: Simulation engine, logging, delays

**Depends on:** S05. **Effort:** 2 days.

## Context

Ties the dynamics together into a closed-loop simulation with multi-rate scheduling: physics at
`sim.dt`, a controller at `sim.controller_rate`, command delay, and logging. Sensors are added in S07, so
in this step the engine feeds controllers a perfect `StateEstimate`. Read `docs/CONVENTIONS.md`,
`src/hopper/dynamics/vehicle.py`, `src/hopper/sim/integrator.py`, `src/hopper/sim/events.py`.

## Deliverables

### `src/hopper/sim/engine.py`
```python
class Simulation:
    def __init__(self, cfg: VehicleConfig, controller: Controller | StateFeedbackController,
                 env: Environment, x0: np.ndarray, rng: np.random.Generator,
                 perfect_state: bool = True, command_delay: float = 0.0): ...
    def run(self, t_end: float | None = None) -> SimResult
```
Loop details:
- Fixed physics step `dt`; controller called every `round(1 / (controller_rate * dt))` steps with its
  last held command applied in between (zero-order hold).
- In `perfect_state` mode, build `StateEstimate` from the true state (mass from `mass_properties`) and call
  `controller.step_state`. Otherwise raise `NotImplementedError` pointing to S07/S10 (S07 will add the
  sensor path; keep a clean seam `_build_controller_input(...)` for it).
- `command_delay` (seconds): a FIFO buffer; the command computed at `t` takes effect at `t + delay`
  (quantised to dt). Before the first command arrives, apply an "engine off, zero gimbal" command.
- RK4 stages use the same held command (command held constant across a step).
- `wind.advance(dt)` once per physics step (outside RK4).
- After each step: update `EventMonitor`; stop at a terminal `Outcome`, `sim.max_time`, or `t_end`.
- Everything deterministic given the rng seed.

```python
@dataclass
class SimResult:
    outcome: Outcome
    t_end: float
    log: pandas.DataFrame   # see schema below
    seed: int | None
```

### `src/hopper/analysis/logger.py`
- `Logger(decimation: int = 1)` collects rows during the run; `to_dataframe()` returns the schema from
  CONVENTIONS.md: `t`, `x0..x18` named (`pos_x, pos_y, pos_z, vel_x, ..., quat_w, ..., omega_x, ..., prop,
  thr, gim_x, gim_y, gimr_x, gimr_y`), `mass`, `thrust`, `cmd_throttle`, `cmd_gim_x`, `cmd_gim_y`,
  `cmd_roll`, `contact_force`, `n_contacts`, `event`.
- `save(path)` / `load(path)` to parquet, round-tripping all columns and the metadata (seed, config hash)
  stored in the parquet file metadata.
- Pre-allocate numpy arrays rather than appending dicts, so 100 s runs at 1 kHz stay fast.

### `src/hopper/control/interface.py`
Re-export the protocols for convenient import, plus a trivial `OpenLoopController` (constant
command or time-scheduled command list) used by tests and examples.

### `scripts/run_scenario.py` (minimal v1)
CLI: `python scripts/run_scenario.py --config configs/vehicle_default.yaml --seed 0 --out runs/test.parquet`
runs a built-in open-loop hop and prints outcome, final state, and elapsed time. S11 will replace it with
scenario files.

## Tests (`tests/sim/test_engine.py`, `tests/analysis/test_logger.py`)

- Open-loop hover command for a vehicle on the ground at throttle slightly above `m g` lifts off, then
  the log shows the expected acceleration (compare against `vehicle_derivative` directly).
- Controller rate: a counting controller is called the expected number of times in 10 s.
- ZOH: a controller that alternates sign each call produces a piecewise-constant command in the log.
- Command delay: a step command appears in the log's applied command exactly `delay` later (within dt).
- Determinism: two runs with the same seed give identical logs; different seeds differ when wind is on.
- Termination: crash scenario ends with `CRASHED`, a short timeout with `TIMEOUT`.
- Energy bookkeeping sanity: an unpowered drop from 5 m lands (`LANDED` or `CRASHED` per speed) with
  kinetic energy at ground ~ `m g h`.
- Logger round trip through parquet preserves all columns and values; decimation works.
- Performance: a 10 s run at 1 kHz finishes in a few seconds. If it takes longer than ~10 s, profile and
  vectorise hot spots. Record the timing in the progress notes.

## Definition of done

- Tests pass; ruff clean; `scripts/run_scenario.py` runs.
- Record the measured simulation speed (steps/s) in the progress notes; later steps depend on it for
  Monte Carlo planning.
- Progress tracker updated per `docs/AGENT_START.md`.
