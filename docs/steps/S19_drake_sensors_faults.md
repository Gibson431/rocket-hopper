# S19: Sensors and fault injection on the Drake diagram

**Depends on:** S18. **Effort:** 2 days.

## Context

Port the sensor path and the S11 faults. The sensor *models* (`src/hopper/sensors/`) are already pure
numpy with explicit RNGs and are reused; what changes is where their inputs come from and how
samples reach the controller. Read `src/hopper/sensors/suite.py`, `src/hopper/sim/faults.py`,
`engine.py` (`_sense`, `_build_controller_input`), `docs/DRAKE_DESIGN.md` (S12 accelerometer source and
randomness decisions).

## Deliverables

- `hopper/drake/sensors.py`: `SensorsSystem(LeafSystem)` wrapping `SensorSuite`: each physics step it
  samples from the plant state, the specific force (source chosen in S12) and returns the delivered
  samples; the controller host keeps the newest sample per field between calls, as the engine does.
  Spawn order of sensor RNGs (`rng.spawn(4)`: imu, baro, gps, lidar) is unchanged so a seed produces the
  the same noise stream for a given seed.
- (No `faults.py` was written; the seams are in `ControllerHost`, `Engine`/`DrakeSimulation` and `SensorsSystem`, see PROGRESS Decisions.) The six faults plus `GyroBias` as diagram elements or wrappers around the
  command, parameter and sensor seams, with the same semantics as `hopper.sim.faults`:
  `apply_command` on the last commanded value every step (stuck gimbal ignores later commands and the
  delay), `modify_params` for thrust scale and gimbal rate (parameters written to the plant/engine at
  trigger time, with the same cache-by-epoch behaviour), `modify_sensors` on delivered samples. With no
  faults the diagram contains none of these systems.
- `DrakeSimulation(..., perfect_state=False, faults=())` drives `Controller.step(SensorData, t)` and the
  EKF-backed `EstimatedController` unchanged.

## Tests (`tests/drake/test_sensors.py`, `test_faults.py`)

The sensor models and faults are inputs to controller testing, so these stay; they are not simulator
verification.

- The existing S07 sensor and S11 fault tests keep passing when run through the diagram: rates,
  latencies, `None` fields between samples, GPS dropouts, `gimbal_stuck` holds the logged gimbal,
  `thrust_scale` scales logged thrust, `engine_out` zeroes thrust, `gps_outage` gives `None` GPS in the
  window, `gimbal_slow` limits the rate, `gyro_bias_step` offsets the gyro. A no-fault run is
  bit-identical to a run built without the fault machinery.
- The IMU specific force is finite and about `[0, 0, g]` for a vehicle at rest on the pad.
- EKF hop on the diagram (seed 0): LANDED, with estimation errors in the S10 acceptance bounds
  (`tests/estimation/test_closed_loop.py`).

## Definition of done

- Tests pass; ruff clean. Progress tracker updated per `docs/AGENT_START.md`.
