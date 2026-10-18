# Progress tracker

Source of truth for step status. Update it at the start and end of every step
(see `docs/AGENT_START.md`). `python3 scripts/next_step.py` parses the table below, so keep its format:
one row per step, six columns, status one of `todo`, `in_progress`, `done`, `blocked`.

Dependencies are in the "Depends" column (comma-separated step IDs, or `-`).

| ID | Title | Status | Depends | Updated | Notes |
|----|-------|--------|---------|---------|-------|
| S00 | Project scaffolding | done | - | 2026-10-17 | uv project (uv_build backend), py>=3.12, empty subpackages, smoke test, CI. Reviewed: PASS (round 2) |
| S01 | Math, types, config | done | S00 | 2026-10-17 | math3d, types (slices POS/VEL/... and make_state), pydantic config (frozen, extra=forbid), configs/vehicle_default.yaml. Reviewed: PASS (round 2) |
| S02 | Rigid-body dynamics and RK4 | done | S01 | 2026-10-17 | rigid_body_derivative; rk4_step/integrate(f,x0,t0,t1,dt,callback,renorm=QUAT,record=False) -> (x, traj or None). Reviewed: PASS (round 2) |
| S03 | Mass model and propulsion | done | S02 | 2026-10-17 | mass_properties; propulsion -> PropulsionOutput; vehicle_derivative(t,x,cmd,cfg,env_force_I,env_torque_B,g); S04 replaced `g` with `env` (see S04 row). S05 adds contact via the env_force_I/env_torque_B hooks. Reviewed: PASS (round 2) |
| S04 | Environment and aerodynamics | done | S03 | 2026-10-18 | environment (gravity, ISA `atmosphere`->Atmosphere, pressure_to_altitude, Wind, Environment), aero_loads, `vehicle_derivative(t,x,cmd,cfg,env=None,env_force_I,env_torque_B)`. Sim loop (S06) must call `env.wind.advance(dt)` per step. Reviewed: PASS (round 2) |
| S05 | Ground contact and events | done | S04 | 2026-10-18 | contact_loads -> ContactOutput (incl. base_strike); vehicle_derivative(..., info=None) always applies contact; sim/events.py (Outcome, EventState, check_events, EventMonitor); thresholds in SimConfig. S06 must run EventMonitor each step. Reviewed: PASS (round 1) |
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

- 2026-10-17, S02: `rk4_step(f, t, x, dt, renorm=QUAT)` and `integrate(f, x0, t0, t1, dt, callback=None, renorm=QUAT, record=False)` return `(x_final, traj_or_None)`. `renorm` is the slice to rescale (None disables, for generic non-quaternion states); `record=True` returns a trajectory with the initial state plus one row per step. The step file left both open. `x0` is copied, never mutated.
- 2026-10-17, S02: `integrate` calls `callback(t, x)` after every step with the post-step time, snapped to exactly `t1` on the final (possibly shortened) step. Step times are `t0 + k*dt`, not accumulated, to avoid drift. A zero-length interval returns `x0` with no steps; `dt <= 0` raises ValueError.
- 2026-10-17, S02: `rigid_body_derivative` keeps the specified `force_I`/`torque_B` argument names with `# noqa: N803`; ruff's pep8-naming would otherwise reject the frame-suffix convention.

- 2026-10-17, S03: `propulsion` returns a `PropulsionOutput(force_B, torque_B, xdot_act)` dataclass; `vehicle_derivative` takes an extra keyword `g=9.80665` (gravity-off for tests). The step left both open.
- 2026-10-17, S03: the CoM is not monotonic over the whole burn for the default vehicle: it dips from 0.860 m to ~0.853 m (at ~13.9 kg) and then rises to the dry 0.9 m, because the filled column's centre falls faster than the mass shrinks. The step's "moves monotonically" test is therefore applied to the late burn only (<= 13 kg), plus a test pinning the early dip.
- 2026-10-17, S03: gimbal limits. The angle derivative uses the rate clamped to +/- `gimbal_rate_max` (so angle changes obey the rate limit exactly) and is zero at the angle limit when moving outward; the thrust direction clips the angle to +/- `gimbal_max`. The raw angle state may overshoot the limit by at most one step of rate*dt under RK4. The rate state itself is not clamped, only its outward acceleration is zeroed at the limit.
- 2026-10-17, S03: `tests/conftest.py` provides a shared `cfg` fixture (default vehicle YAML).

- 2026-10-18, S04: `vehicle_derivative(t, x, cmd, cfg, env=None, env_force_I=None, env_torque_B=None)`. `env=None` means `Environment.default()`; the S03 `g` keyword is removed (gravity comes from `environment.gravity`, disabled by `Environment(gravity_enabled=False)`). The contact hooks stay for S05.
- 2026-10-18, S04: `Environment` has an extra `aero_enabled` flag (default True). Rocket-equation tests need drag off, since drag at ~100+ m/s breaks the 1e-3 delta-v match.
- 2026-10-18, S04: `atmosphere` returns an `Atmosphere` NamedTuple (density, temperature, pressure, speed_of_sound); it still unpacks as the specified 4-tuple. Altitude is clamped to 0-11 km.
- 2026-10-18, S04: `Wind(mean_I, gust_sigma, gust_tau, rng)` plus `reset(seed)`, which replaces the rng with `default_rng(seed)` and zeroes the gust. `advance(dt)` uses the exact discretisation (valid for any dt), so there is no fixed internal step. Gust starts at zero. `Wind.none()` uses a dummy seeded rng.
- 2026-10-18, S04: the terminal-velocity test raises `cd_axial` to 5 so the vehicle settles within 30 s of simulated fall.

- 2026-10-18, S05: friction regularisation width is `V_EPS = 0.05 m/s`, not the specified 0.01. With 0.01, yaw friction (slope ~7000 /s on I_zz ~ 3 kg m^2) is unstable under RK4 at dt = 1 ms and leaves a permanent 0.024 rad/s yaw limit cycle. 0.05 is stable at the default dt; no contact substep is needed. Default stiffness/damping (30000 N/m, 1500 N s/m) are stable and unchanged.
- 2026-10-18, S05: `vehicle_derivative(..., info=None)`: contact is always applied inside it (not via the env hooks, which stay for other loads); `info["contact"]` receives the `ContactOutput`. `ContactOutput` also carries `base_strike`.
- 2026-10-18, S05: outcome thresholds are `SimConfig` fields: `v_crash`, `crash_tilt`, `tipover_tilt`, `land_speed`, `land_tilt`, `land_rate`, `land_hold`, `max_range` (500 m), `max_alt` (200 m); angles in rad. The step left `max_range`/`max_alt` defaults open.
- 2026-10-18, S05: `check_events(t, x, contact, cfg, prev_state: EventState)` mutates `prev_state`; `EventMonitor(cfg).update(t, x, contact)` wraps it. Touchdown crash speed is the vertical speed from the call *before* first contact (the post-contact speed is already cushioned). `TIMEOUT` is reported when `t >= sim.max_time`. `FLAMEOUT_FALLING` exists in `Outcome` but is not detected (the step does not define it); S06 or later can add it.
- 2026-10-18, S05: `tipover` uses `tilt > 60 deg` while any foot is in contact; `CRASHED` precedes `TIPPED_OVER`, which precedes `OUT_OF_BOUNDS`, `LANDED`, `TIMEOUT` when several hold at once.

## Backlog

Out-of-scope ideas found while working. Do not do these inside another step.

- S01 review suggestions (also: remove duplicate assertion in geometry test): test `quat_error` small-angle branch (n < 1e-12) and `dcm_to_quat` on non-unit input; add vehicle-length to config and check tank fits inside it; validate `Command.throttle` / `roll_torque` ranges (decide where in S03/S08).
- Smoke test: also assert each subpackage has a real `__init__.py` (`__file__ is not None`), since a deleted one still imports as a namespace package.
- S02 review suggestions: pin the callback's final time == t1 exactly for a non-divisible span; tighten the RK4 exponential-decay test tolerance to ~1e-7; hypothesis test over spans/dts for step count.
- S03 review suggestions: amend S03 step text (CoM is not monotonic over the full burn); add negative-side and gy-axis cases for the gimbal angle/rate-limit tests; test propellant crossing zero mid-burn; decide where to validate `Command` ranges (propulsion clamps for now).
- S04 review suggestions: test simultaneous vehicle motion plus wind in `aero_loads`; check `Wind.advance` stays a no-op with sigma 0 across a reset.; add a direct hand-computed mixed axial+lateral `aero_loads` test and a tilted-vehicle cross-flow torque test.
- S05 review suggestions: confirm the V_EPS = 0.05 deviation is acceptable; mention the SimConfig thresholds and `info` hook in CONVENTIONS.md; implement `FLAMEOUT_FALLING` detection when its rule is defined.

## Session log

Append one entry per session that changes status. Newest last.
Format: `YYYY-MM-DD, step, status change, what was done, what the next agent should know.`

- 2026-10-17, plan: plan, conventions and step files written. No code yet.
- 2026-10-17, S00, todo -> done: scaffolding created (pyproject, uv.lock, subpackages, smoke test, CI, README). Reviewed: PASS (round 2; round 1 failed on unlogged decisions). Python floor is 3.12; run everything via `uv run`. Next: S01.
- 2026-10-17, S01, todo -> done: math3d, types, config, default YAML, 65 tests. Reviewed: PASS (round 2; round 1 failed on unlogged decisions). State indices are slices (POS, VEL, QUAT, OMEGA, GIM, GIMR) or ints (PROP, THR). Next: S02.
- 2026-10-17, S02, todo -> done: rigid_body_derivative, rk4_step, integrate, 25 tests. Reviewed: PASS (round 2; round 1 failed on unlogged decisions). Integrator signature extras (renorm, record) are in Decisions; S03/S06 call `integrate(...)` which returns a `(x, traj)` tuple. Next: S03.
- 2026-10-17, S03, todo -> done: mass_properties, propulsion (PropulsionOutput), vehicle_derivative with env hooks and `g` kwarg, 133 tests total. Reviewed: PASS (round 2; round 1 failed on unpinned gimbal damping and a placeholder note). CoM dips early in the burn (see Decisions). Next: S04.
- 2026-10-18, S04, todo -> done: environment.py, aero.py, vehicle_derivative takes `env`; 164 tests. Reviewed: PASS (round 2; round 1 failed on unpinned aero torque in vehicle_derivative and unlogged decisions). Next: S05.
- 2026-10-18, S05, todo -> done: contact.py, events.py, SimConfig thresholds, 30 new tests (194 total). Reviewed: PASS (round 1). Friction V_EPS widened to 0.05 m/s for RK4 stability at dt = 1 ms (see Decisions). Next: S06 (call `EventMonitor.update` every step and `env.wind.advance(dt)`).
