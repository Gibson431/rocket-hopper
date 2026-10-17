# S10: State estimator (EKF)

**Depends on:** S07, S09. **Effort:** 3-4 days.

## Context

Replace perfect-state feedback with an estimate built from the S07 sensors. The estimator sits between
`SensorData` and the controller (it is flight software, so it lives under `hopper.estimation` and may only
import `hopper.types` and `hopper.math3d`). Read `docs/CONVENTIONS.md` (frames, quaternion convention,
`SensorData`, `StateEstimate`), `src/hopper/sensors/`, and `src/hopper/control/mission.py`.

## Deliverables

### `src/hopper/estimation/ekf.py`
Error-state (multiplicative) EKF, written by hand with numpy.
- Nominal state: position, velocity (I), quaternion, gyro bias (3), accel bias (3), baro bias (1).
  Error state (dimension 16): `dp, dv, dtheta, dbg, dba, dbaro`. Document the definitions.
- Propagation at each IMU sample: integrate gyro (bias-corrected) into the quaternion, rotate the
  bias-corrected specific force into I, add gravity, integrate velocity/position (midpoint or RK2).
  Covariance propagation `P = F P F^T + Q`, with `F` and `Q` from the noise densities in config.
- Updates (sequential, scalar or small-vector, Joseph form):
  - Baro altitude (with bias state).
  - GPS position and velocity (handle latency by buffering state history and re-applying the update at
    the correct historical time, or simply by discarding GPS in the first version and logging the decision;
    prefer the buffered approach).
  - Lidar range -> altitude `h = range * cos(tilt)` (only when valid, and when the estimated tilt is < 45 degrees);
    reject outliers with a chi-square innovation gate.
  - Optional: a "ground contact" zero-velocity pseudo-measurement when the estimated altitude is near zero and the
    mission state is `LANDED`/`IDLE` (pass a flag).
- Initialisation: static alignment (average accel for roll/pitch, yaw assumed zero in the test setup),
  initial covariance from config.
- Mass estimation: not an EKF state. Estimate propellant by integrating `thrust_cmd / (isp g0)` using the
  controller's own commanded throttle (documented dead-reckoning), exposed via `StateEstimate.mass`.
- API: `EKF(params, ...)`, `update_imu(...)`, `update_baro(...)`, `update_gps(...)`, `update_lidar(...)`,
  and `process(sensors: SensorData) -> StateEstimate | None` which applies everything available in a
  `SensorData` and returns the best estimate (None until initialised).
- Expose `P` diagonal and innovation values for logging.

### Integration
- `EstimatedController` wrapper: `Controller` whose `step(sensors, t)` calls the EKF, then
  `MissionController.step_state(estimate, t)`. The sim engine runs with `perfect_state=False`.
- Log estimate and covariance next to truth in the sim log (extra columns `est_*`, `sig_*`) via an
  optional hook so analyses can plot estimation error.

## Tests (`tests/estimation/`)

- Noise-free, bias-free sensors: the estimate tracks truth to within numerical error across a hop trajectory.
- With nominal noise: RMS position error < 0.5 m, velocity < 0.1 m/s, attitude < 0.5 degree over a full hop
  (record actual numbers in notes).
- Consistency (NEES): over 20 seeded runs, the average normalised estimation error squared falls within
  the chi-square bounds for 15 or so dof (allow a liberal 95% bound); innovations (NIS) for baro/GPS also in bounds.
- Bias estimation: with a 0.5 deg/s constant gyro bias, the estimate converges to the bias within 10%
  after 30 s of motion.
- GPS latency: with 150 ms latency, buffered update beats naive application (lower RMS), asserted on a
  maneuvering trajectory.
- Outlier robustness: a single 50 m GPS spike is rejected by the gate; a 5 s GPS outage grows the covariance
  and the estimate stays bounded.
- Closed-loop: the full hop (S09 test) lands with `perfect_state=False`, error < 1 m, touchdown speed < 1.5 m/s.
- Import-boundary test extended to `hopper.estimation`.

## Definition of done

- Closed-loop hop with sensors and EKF passes; the consistency test passes.
- Ruff clean; state definition and noise tuning documented.
- Progress tracker updated per `docs/AGENT_START.md` (record RMS error numbers).
