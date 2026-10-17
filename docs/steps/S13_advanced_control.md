# S13: Advanced control (LQR, MPC, Gymnasium)

**Depends on:** S11. **Effort:** open-ended; treat each sub-item as optional and independently completable.
Record each sub-item as its own line in the Notes column and the session log. Mark the step `done`
when the sub-items you and the user agreed on are complete (ask the user which to do if unclear).

## Context

With the baseline PID cascade, the estimator, and Monte Carlo in place, evaluate more capable controllers
against the same scenarios and metrics. Read `docs/CONVENTIONS.md`, `src/hopper/control/`, and the S11
Monte Carlo tooling. All new controllers must follow the same rules: implement `StateFeedbackController` (or
`Controller` behind the EKF), and obey the import boundary (no simulator imports in `hopper.control`).
Add dependencies with `uv add` (`control`, `casadi`, `gymnasium`) and note them in `CONVENTIONS.md`.

## S13a: Linearisation, LQR, and margins

- `scripts/linearize.py` (may import the simulator since it is a tool, not flight code): numerically
  linearise `vehicle_derivative` about hover at several propellant masses and export `A`, `B`.
- `src/hopper/control/lqr.py`: LQR attitude + position controller on the linearised model with gain scheduling
  over estimated mass; optional integral action. Q/R documented.
- Margins with `python-control`: loop-transfer-function gain/phase margins for the attitude loop (with
  actuator lag + delay modelled as a Pade approximation), reported for the PID baseline and LQR, plotted in a script.
- Tests: linearisation matches finite differences; LQR hop passes the S09 test thresholds; margins above 6 dB / 30 degrees
  for the nominal case (record actual values).

## S13b: MPC / powered-descent guidance

- `src/hopper/control/mpc.py`: CasADi-based MPC (translational dynamics with attitude handled by the inner loop)
  or a convex powered-descent guidance for the final landing with thrust magnitude and tilt constraints
  (including the minimum throttle, which makes the problem non-convex; use a lossless-convexification or
  a simple sequential approach and document the choice).
- Real-time budget: solve time per call logged; fail the test if the mean solve time exceeds the controller period.
- Tests: tracks the hop reference; respects constraints in the log (tilt, throttle limits); lands with the same or
  better precision than PID on the nominal scenario; compare MC success rates against the baseline and record them.

## S13c: Gymnasium wrapper (optional)

- `src/hopper/gym_env.py`: `HopperEnv(gymnasium.Env)` over `Simulation` stepped at the controller rate;
  observation = estimate-like vector (or sensor-based), action = `Command`, reward shaped on landing error,
  speed, fuel, and tilt. Seeding via `reset(seed=...)`.
- Tests: `gymnasium.utils.env_checker.check_env` passes; reset determinism; episodes terminate with the `Outcome`.

## Definition of done (per sub-item)

- Tests pass; ruff clean; results compared with the PID baseline via the S11 Monte Carlo and recorded in
  the progress notes.
- Progress tracker updated per `docs/AGENT_START.md`.
