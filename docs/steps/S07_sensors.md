# S07: Sensor models

**Depends on:** S06. **Effort:** 2 days.

## Context

Replace perfect-state input with realistic sensors. Read `docs/CONVENTIONS.md` (the `SensorData`
contract: a `None` field means "no new sample this tick") and `src/hopper/sim/engine.py`, especially
the `_build_controller_input` seam. Sensors only read true state; they never feed back into dynamics.

## Deliverables

### Config (`SensorsConfig` in `src/hopper/config.py`, filled into `configs/vehicle_default.yaml`)
Per sensor: `rate`, `latency`, noise parameters, plus mounting. Defaults (placeholders):
- IMU: `rate 400 Hz`, gyro noise density 0.005 deg/s/sqrt(Hz), gyro bias instability 5 deg/h, accel noise
  density 100 ug/sqrt(Hz), accel bias 0.02 m/s^2, gyro range 2000 deg/s, accel range 16 g, latency 2 ms.
- Baro: 50 Hz, noise sigma 0.5 m, bias random walk 0.05 m/sqrt(s), latency 20 ms.
- GPS: 10 Hz, position sigma 0.5 m horizontal / 1 m vertical, velocity sigma 0.05 m/s, latency 150 ms,
  dropout probability and outage windows configurable.
- Lidar: 50 Hz, range 0.2-40 m, sigma 0.02 m + 0.1% of range, latency 10 ms, measures distance along -z_B
  to the ground plane (returns `None` if the ray misses the ground or is out of range; tilt > 60 deg also
  misses).

### Modules in `src/hopper/sensors/`
- `imu.py`: gyro measures `omega + bias + noise`; accelerometer measures **specific force in the body
  frame** `R^T (a_I - g_I)` computed from the true (non-gravitational) acceleration. Take the specific force
  from thrust, drag and contact forces divided by mass, so provide it from the vehicle model
  (return `specific_force_B` from `vehicle_derivative`'s `info` dict). Bias as a random walk; noise discretised
  as `sigma = density * sqrt(rate)`. Saturation clamps.
- `baro.py`: altitude from true z via ISA pressure with noise and bias walk (`SensorData.baro_alt` is in metres).
- `gps.py`: position and velocity with noise, latency, dropouts.
- `lidar.py`: slant range along -z_B.
- `suite.py`: `SensorSuite(cfg, rng)` with `sample(t, true_state, info) -> SensorData`, internal per-sensor
  scheduling by rate, and latency via per-sensor delay queues (a sample taken at `t` is delivered at
  `t + latency` with its original timestamp in a separate `sample_t` internal field; `SensorData.t` is
  the delivery time). Each sensor uses its own child `Generator` spawned from the parent so adding a sensor
  does not change other sensors' noise.

### Engine integration
`Simulation(perfect_state=False)` routes through `SensorSuite` and calls `controller.step(sensors, t)`.
The controller sees only the latest sample per sensor since the last controller call (fields are `None`
if nothing new arrived). Keep `perfect_state=True` working.

## Tests (`tests/sensors/`)

- Statistics: with a fixed seed and a static vehicle, gyro/accel/baro/GPS sample means/stds match the
  configured noise (5% for 10 k samples). Bias random walk variance grows linearly with time.
- Static IMU: upright and at rest, accelerometer reads `[0, 0, +g]` (specific force upward), gyro reads
  bias only.
- Free fall: accelerometer reads ~0 (no thrust, no drag).
- Rate scheduling: the number of samples over 10 s equals `rate * 10` (+/- 1).
- Latency: a step in true altitude appears in `baro_alt` no earlier than `latency` later.
- Saturation clamps; lidar returns `None` out of range or when the vehicle is on its side.
- GPS dropout windows produce `None` for the specified duration.
- Same seed -> identical sensor streams; changing one sensor's config does not change another's samples.
- Engine test: with `perfect_state=False`, a trivial controller receives `SensorData` fields at the
  expected rates and with the expected `None` pattern.

## Definition of done

- Tests pass; ruff clean; `CONVENTIONS.md` `SensorData` section matches the implementation.
- Progress tracker updated per `docs/AGENT_START.md`.
