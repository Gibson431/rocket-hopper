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
  same noise as the legacy run.
- `hopper/drake/faults.py`: the six faults plus `GyroBias` as diagram elements or wrappers around the
  command, parameter and sensor seams, with the same semantics as `hopper.sim.faults`:
  `apply_command` on the last commanded value every step (stuck gimbal ignores later commands and the
  delay), `modify_params` for thrust scale and gimbal rate (parameters written to the plant/engine at
  trigger time, with the same cache-by-epoch behaviour), `modify_sensors` on delivered samples. With no
  faults the diagram contains none of these systems.
- `DrakeSimulation(..., perfect_state=False, faults=())` drives `Controller.step(SensorData, t)` and the
  EKF-backed `EstimatedController` unchanged.

## Tests (`tests/drake/test_sensors.py`, `test_faults.py`)

- Port the S07 sensor tests to run through the diagram: rates, latencies, `None` fields between
  samples, noise statistics, GPS dropouts, baro/lidar from the CoM height, lidar range along -z_B.
- The IMU specific force on the diagram equals `info["specific_force_B"]` from legacy
  `vehicle_derivative` at the same state and command to 1e-6 (hover on the pad, free flight under
  thrust, mid-landing).
- For equal seeds the noise sequences equal the legacy ones (same RNG spawn order), compared on the
  samples.
- Port the S11 fault tests: `gimbal_stuck` holds the logged gimbal from the trigger time,
  `thrust_scale` scales logged thrust, `engine_out` zeroes thrust through the throttle lag,
  `gps_outage` gives `None` GPS in the window, `gimbal_slow` limits the rate, `gyro_bias_step` offsets
  the gyro; a no-fault run is bit-identical to a run built without the fault machinery.
- EKF hop on the diagram (seed 0, S10 numbers): LANDED, RMS position/velocity/attitude errors within the
  S12 tolerances of the legacy figures (0.108 m, 0.019 m/s, 0.20 deg).

## Definition of done

- Tests pass; ruff clean. Progress tracker updated per `docs/AGENT_START.md`.
