# S01: Math, types, config

**Depends on:** S00. **Effort:** 1 day.

## Context

Foundation modules everything else imports. Read `docs/CONVENTIONS.md` fully: it defines frames,
quaternion convention (scalar-first, body -> inertial), the 19-element state layout, and the
`Command` / `SensorData` / `StateEstimate` / `Controller` interfaces. Implement them exactly.

## Deliverables

### `src/hopper/math3d.py`
Pure numpy, no state. All functions accept/return `np.ndarray` of float64.
- `quat_normalize(q)`, `quat_mul(a, b)` (Hamilton product, scalar-first), `quat_conj(q)`,
  `quat_to_dcm(q)` (returns R with `v_I = R @ v_B`), `dcm_to_quat(R)`.
- `quat_rotate(q, v)` (body -> inertial), `quat_rotate_inv(q, v)` (inertial -> body).
- `quat_from_axis_angle(axis, angle)`, `quat_from_euler_zyx(yaw, pitch, roll)`,
  `quat_to_euler_zyx(q)` (display only).
- `quat_error(q_des, q)` returning the rotation-vector error (3,) in the body frame, taking the shortest
  path (flip sign when w < 0). This is what attitude control will use.
- `quat_derivative(q, omega_body)` = `0.5 * q (x) [0, omega]`.
- `skew(v)`.

### `src/hopper/types.py`
- Index constants `POS, VEL, QUAT, OMEGA, PROP, THR, GIM, GIMR` and `N_STATE = 19` per CONVENTIONS.md.
- `Command`, `SensorData`, `StateEstimate` frozen dataclasses; `Controller` Protocol;
  `StateFeedbackController` Protocol (adds `step_state`).
- `make_state(pos, vel, quat, omega, prop, thr=0.0, gim=(0, 0), gimr=(0, 0)) -> np.ndarray`
  helper with sensible defaults (unit quaternion if none given).
- `Command` validates shapes in `__post_init__` (gimbal shape (2,)).

### `src/hopper/config.py`
- Pydantic v2 models for the vehicle config groups listed in CONVENTIONS.md (`MassConfig`,
  `TankConfig`, `EngineConfig`, `RcsConfig`, `LegsConfig`, `AeroConfig`, `SimConfig`, and
  `VehicleConfig` aggregating them). `sensors` group: placeholder empty model for S07.
- `load_vehicle(path) -> VehicleConfig` from YAML. Validate positivity and sane ranges
  (e.g. `0 < thrust_min_frac <= 1`, `gimbal_max < pi/4`).
- `configs/vehicle_default.yaml` with the placeholder vehicle described in CONVENTIONS.md.
  Make the numbers self-consistent: tank inside the vehicle, legs long enough to clear the engine.

## Tests (`tests/test_math3d.py`, `tests/test_types.py`, `tests/test_config.py`)

- Quaternion multiply identity, conjugate inverse, `quat_to_dcm` orthonormal with det 1.
- `dcm_to_quat(quat_to_dcm(q))` round trip (compare up to sign) using hypothesis random unit quaternions.
- 90 degree rotation about z maps x_B to y_I (right-handed, ENU).
- `quat_rotate_inv` is the inverse of `quat_rotate`.
- `quat_error` is zero for identical attitudes, picks the shortest path for q and -q, and gives
  a rotation vector of magnitude equal to the rotation angle.
- `quat_derivative` integrates a constant omega to the analytic quaternion (small-step check).
- `make_state` returns shape (19,) with correct slices.
- Config loads the default YAML; invalid values raise `ValidationError`.

## Definition of done

- All tests above pass; `ruff check` and `ruff format --check` clean.
- `docs/CONVENTIONS.md` matches the implemented interfaces (update it if you had to deviate, and log the decision).
- Progress tracker updated per `docs/AGENT_START.md`.
