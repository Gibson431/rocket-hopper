# S09: Position cascade and mission state machine

**Depends on:** S08. **Effort:** 2-3 days.

## Context

Adds the outer loops (altitude and horizontal position) on top of the attitude controller, and a mission
state machine that flies a **hop-hover-land** profile. This step uses perfect-state feedback (S10 adds
the estimator later). It is the first milestone where the simulator flies a full mission. Read
`docs/CONVENTIONS.md`, `docs/AGENT_START.md` (controller package import rules), and
`src/hopper/control/attitude.py`.

Practical constraint to design around: minimum throttle is 40% (1000 N) and a full 100 kg vehicle weighs
~981 N, so hovering when full is marginal and the vehicle cannot reduce thrust below weight until propellant
burns off. The controller must cope with this (e.g. the altitude loop saturates at the minimum, descent
requires reduced weight or accepts a "suicide burn" style profile). For the placeholder vehicle consider
lowering `thrust_min_frac` to ~0.3 in `configs/vehicle_default.yaml` if the profile becomes infeasible, and log
the decision.

## Deliverables

### `src/hopper/control/position.py`
- `AltitudeController`: PID on altitude/vertical velocity producing a throttle command. Gravity
  feedforward `m g / (T_max cos(tilt))` using estimated mass, clamped to `[thrust_min_frac, 1]`. Output 0 to
  cut the engine when commanded.
- `HorizontalController`: PD on horizontal position/velocity producing a desired thrust tilt: `a_des =
  kp (p_des - p) + kd (v_des - v)`, converted to a desired attitude by tilting the thrust vector toward the
  required horizontal acceleration `(a_des_x, a_des_y)`. Clamp tilt (e.g. 15 degrees), and rate-limit the
  attitude setpoint. Desired yaw held (default 0), producing `q_des` including yaw.
- Both gain sets are dataclasses with documented design rationale (bandwidth separation: position loop
  about 3-5x slower than the attitude loop).

### `src/hopper/control/mission.py`
Mission state machine with explicit states and transitions, deterministic and unit-testable:
`IDLE -> IGNITION -> ASCEND -> HOVER -> TRANSLATE (optional) -> DESCEND -> TOUCHDOWN -> CUTOFF -> LANDED`.
- Mission parameters (dataclass): `target_alt`, `hover_time`, `target_xy` (landing site offset), `ascent_rate`,
  `descent_rate`, `final_descent_rate`, `touchdown_alt`, `cutoff_alt`.
- Vertical profile as velocity commands: ramp up, hold altitude, descend at `descent_rate` until
  `touchdown_alt`, then at `final_descent_rate` until contact is detected (altitude near zero or
  estimated vertical acceleration spike), then cut the engine.
- Landing altitude uses whichever altitude source the controller has; in this step it comes from the
  `StateEstimate`.
- State-transition guards use hysteresis and timeouts; abort path: if tilt exceeds a limit or altitude error
  grows past a bound, enter `ABORT` (cut engine if low, else try to level and descend).
- `MissionController` implements `StateFeedbackController`, wiring state machine + position loops + attitude controller.

### `scripts/run_scenario.py`
Add a `--mission hop` option that flies the default hop with `MissionController` and prints a
summary (apogee, landing error, touchdown speed, time of flight, fuel used, outcome).

## Tests (`tests/control/test_position.py`, `tests/control/test_mission.py`, `tests/sim/test_hop_scenario.py`)

- Altitude step: from hover at 10 m, a +5 m command settles within 1% of the target with overshoot < 15%.
- Horizontal step: a +5 m x command settles with the vehicle tilting no more than the clamp, no altitude loss > 1 m.
- State machine transitions: scripted state inputs drive each transition; timeouts fire; `ABORT` entered on
  an out-of-bounds condition.
- **Full hop:** 10 m up, hover 3 s, translate 3 m, descend, land. Outcome `LANDED`, landing position
  error < 0.5 m, touchdown vertical speed < 1 m/s, tilt < 5 degrees, with default config, no wind, and perfect state.
- Same hop with the S03 actuator lags and 20 ms command delay (still `LANDED`).
- Same hop with a 3 m/s steady wind: `LANDED` with a position error < 1.5 m (looser; record the actual numbers).
- Import-boundary test (from S08) still passes.

## Definition of done

- The full-hop test passes reproducibly (fixed seed) and the metrics are recorded in the progress notes.
- Ruff clean; gains and mission parameters documented.
- Progress tracker updated per `docs/AGENT_START.md`.
