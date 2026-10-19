# Implementation plan: rocket hopper simulator

## Goal

A Python 6-DOF simulator for a thrust-vector-controlled (TVC) rocket hopper, used to design, tune and
stress-test the flight controller before hardware.

**In scope:** rigid-body 6-DOF with variable mass/inertia; throttled, gimballed thrust; roll control
via RCS; actuator, sensor, delay and wind models; ground contact with legs; closed-loop controllers;
scenarios with fault injection; Monte Carlo; plots and replay.

**Out of scope for v1:** CFD-grade aero, propellant slosh, structural flex, hardware-in-the-loop,
RL training (a Gymnasium wrapper is an optional late step).

**Vehicle numbers** (mass, inertia, thrust, gimbal limits, tank layout) are placeholders in
`configs/vehicle_default.yaml` until real values exist. Nothing in the plan depends on the specific values.

## Why a custom sim, and why now Drake

No existing rocket simulator (RocketPy, CamPyRoS, OpenRocket) supports TVC, throttling or powered
landing, so S00-S11 wrote one. Drake is a general multibody and systems framework rather than a rocket
tool, so it fits: the rocket-specific parts (variable mass, gimbal, actuator lag) become small systems
around a maintained rigid-body and contact engine, and we get composable diagrams and Meshcat
visualisation. Expect parity with the hand-rolled code on speed, not a speedup (Python systems stay on
the hot path), and note that linearisation/autodiff will not see through Python systems. Drake is the
source of truth for the physics; the hand-rolled code is deleted in S22 and not compared against.

## Architecture

The key rule: **the controller never sees true state.** It receives sensor outputs and returns
actuator commands. See `docs/CONVENTIONS.md` for exact types.

```
rocket-hopper/
  CLAUDE.md, pyproject.toml
  configs/
    vehicle_default.yaml
    scenarios/*.yaml
  docs/                     # this plan, conventions, progress, steps/
  scripts/
    next_step.py            # prints the next step to work on
    run_scenario.py
  src/hopper/
    types.py                # state layout, SensorData, Command
    config.py               # pydantic models + YAML loading
    math3d.py               # quaternion ops, frames
    dynamics/
      rigid_body.py
      mass_model.py
      propulsion.py
      aero.py
      environment.py
      contact.py
      vehicle.py            # combines the above into f(t, x, cmd) -> xdot
    sim/
      integrator.py         # fixed-step RK4
      events.py
      engine.py             # multi-rate loop, delay buffers
    sensors/
      imu.py, baro.py, gps.py, lidar.py, suite.py
    estimation/ekf.py
    control/
      interface.py, attitude.py, position.py, mission.py, mpc.py
    analysis/
      logger.py, plots.py, monte_carlo.py, viz.py
  tests/
```

Multi-rate loop: physics at 1 kHz, IMU at 200-500 Hz, GPS 5-10 Hz, baro 50 Hz, controller 100-200 Hz.
Commands pass through a delay buffer, and sensors through their own latency.

## Steps

Each step has a self-contained file in `docs/steps/`. Status lives in `docs/PROGRESS.md`.

| ID | Title | Depends on | Effort |
|----|-------|-----------|--------|
| S00 | Project scaffolding | none | 0.5 d |
| S01 | Math, types, config | S00 | 1 d |
| S02 | Rigid-body dynamics and RK4 | S01 | 2 d |
| S03 | Mass model and propulsion | S02 | 2 d |
| S04 | Environment and aerodynamics | S03 | 1-2 d |
| S05 | Ground contact and events | S04 | 2 d |
| S06 | Simulation engine, logging, delays | S05 | 2 d |
| S07 | Sensor models | S06 | 2 d |
| S08 | Control interface and attitude controller | S06 | 2 d |
| S09 | Position cascade and mission state machine | S08 | 2-3 d |
| S10 | State estimator (EKF) | S07, S09 | 3-4 d |
| S11 | Scenarios, faults, Monte Carlo | S09 (S10 optional) | 2-3 d |

S00-S11 built the hand-rolled simulator and flight software (all done). S12-S22 port the simulation
side to Drake; see "Drake port" below.

The first flyable milestone was **S09** (a hop with perfect state feedback).

## Drake port (S12-S22)

**Scope.** Dynamics, environment, contact, integration, the sim loop, fault injection, Monte Carlo
plumbing and visualisation move to Drake (`pydrake`, in the new `src/hopper/drake/` package). Flight
software does not move: `hopper.control`, `hopper.estimation`, `hopper.types` and `hopper.math3d` stay
hand-written, never import `pydrake`, and run unchanged inside a Drake `LeafSystem` host. The sensor
models are reused as they are.

**Method.** The project is about the controller, not the simulator. Drake is the source of truth, so
there are no golden files, no legacy-vs-Drake comparisons and no simulator unit tests. Each porting
step needs only a smoke test (builds, runs, outputs finite, same seed gives the same result), and the
real acceptance is that the S09-S11 controller scenarios fly on the Drake backend.

| ID | Title | Depends on | Effort |
|----|-------|-----------|--------|
| S12 | Drake feasibility spike and design decisions (gate) | S11 | 1-2 d |
| S14 | Drake vehicle plant, mass properties, state adapters | S12 | 1-2 d |
| S15 | Propulsion and actuators as Drake systems | S14 | 2 d |
| S16 | Environment, wind and aerodynamics | S14 | 1-2 d |
| S17 | Legs, ground contact and outcome events | S14 | 2-3 d |
| S18 | Closed-loop diagram, controller host, delay, logging | S15, S16, S17 | 3 d |
| S19 | Sensors and fault injection | S18 | 2 d |
| S20 | Scenarios and Monte Carlo on Drake | S19 | 2 d |
| S21 | Visualisation: Meshcat replay and log plots | S18 | 2 d |
| S22 | Cutover to Drake and legacy removal | S20, S21 | 1-2 d |

S15, S16 and S17 run in parallel after S14; S21 can run alongside S19-S20.

**Known design problems** (S12 answers them): Drake has no fixed-step RK4 (the "fixed-step RK4" rule is
amended in S12/S22); `MultibodyPlant` mass is a context parameter, not a state, so burning propellant
needs an updater system; the legacy body frame is at the CoM, Drake's is fixed on the body; Python
`LeafSystem` callbacks may be slower than the current 1300 steps/s; Drake objects are not picklable
(build the diagram inside each Monte Carlo worker). Found while checking feasibility: a `LeafSystem`
cannot write the plant's mass parameters (needs a sim-loop hook), a separate delay system adds a step
of latency, and Drake's contact (series stiffness, depth-proportional damping) will not reproduce the
legacy leg law exactly, so S17 has a fallback of keeping the legacy law as a force element.
Effort for S12-S22 sums to about 17-21 days; S17 and S18 carry the schedule risk.

**Removed from the plan.** The earlier S12 (matplotlib plots and 3D replay) and S13 (LQR/MPC/Gym) were
scrapped when the port was planned, and the later S13 (legacy goldens and parity harness) was dropped
because the simulator is not the focus. Plots and replay come back in S21, built on Meshcat. The advanced
control ideas are in the Backlog of `docs/PROGRESS.md` for after the port.

## Validation strategy

- Unit tests target flight software: controller, estimator, mission logic, math and types. Analytic
  cases and property tests (hypothesis) are fine there.
- The simulator (Drake) is trusted: smoke tests only (builds, runs, finite, deterministic for a seed).
  No simulator unit tests, no golden files, no A/B comparison between simulators.
- Closed-loop scenarios (S09-S11) and Monte Carlo are the acceptance for the controller.

## Risks

| Risk | Mitigation |
|------|-----------|
| Real vehicle parameters unknown | Monte Carlo dispersions; aim for robustness; update the config later |
| Contact dynamics unstable at 1 kHz | Substep contact, or use MuJoCo for contact only |
| EKF hides controller problems | Keep a perfect-state mode for A/B comparison |
| Scope creep into fidelity | Every step has a definition of done; extras go to Backlog |

## Later: Rust

The controller interface (sensor struct in, command struct out, fixed rate) is what would be
reimplemented in Rust, optionally hosted by the sim through PyO3.
