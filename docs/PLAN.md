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

## Why a custom sim

No existing simulator (RocketPy, CamPyRoS, OpenRocket) supports TVC, throttling or powered landing, and
none exists in Rust. A 6-DOF rigid body with quaternions is ~150 lines. The rocket-specific parts
(variable mass, gimbal, actuator lag, sensors) are easier to write than to bolt on.

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
| S12 | Plots and 3D replay | S06 | 2 d |
| S13 | Advanced control (LQR, MPC, Gym) | S11 | open-ended |

S07 and S08 can proceed in parallel after S06. S12 can start any time after S06.
The first flyable milestone is **S09** (a hop with perfect state feedback).

## Validation strategy

- Unit tests per module against analytic cases (free fall, torque-free spin, rocket equation).
- Conservation checks (energy, angular momentum) as regression tests.
- Integrator convergence test: halving dt reduces error at 4th order.
- Golden scenarios: fixed-seed runs whose key metrics are checked in CI.
- Property tests (hypothesis) for quaternion operations.
- Optional cross-check of a simplified case against RocketPy or MuJoCo.

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
