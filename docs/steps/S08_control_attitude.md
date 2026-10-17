# S08: Control interface and attitude controller

**Depends on:** S06. **Effort:** 2 days.

## Context

First closed-loop controller: stabilise attitude using TVC gimbal and RCS roll, with perfect-state
feedback via `StateFeedbackController.step_state`. Read `docs/CONVENTIONS.md`
(controller interfaces, gimbal geometry and torque sign convention) and `docs/AGENT_START.md` (the
controller package must not import from `hopper.dynamics`, `hopper.sim`, or `hopper.sensors`; only
`hopper.types` and `hopper.math3d`). The vehicle parameters the controller needs (thrust max, arm, inertia)
are passed through a controller-specific config dataclass, not by importing the simulator.

## Deliverables

### `src/hopper/control/attitude.py`
```python
@dataclass
class AttitudeGains:
    kp: np.ndarray  # (3,) rad -> torque scale
    kd: np.ndarray  # (3,)
    ...

class AttitudeController:
    def __init__(self, params: ControlParams, gains: AttitudeGains): ...
    def reset(self): ...
    def update(self, q_des, omega_des, quat, omega, thrust, mass, com_z, dt) -> AttitudeOutput
```
- `ControlParams` (in `src/hopper/control/params.py`): `thrust_max`, `thrust_min_frac`, `gimbal_max`,
  `gimbal_rate_max`, `max_roll_torque`, `inertia_diag_dry`, `dry_mass`, `dry_com_z`, `prop_mass_initial`,
  `tank` geometry for approximating inertia/CoM vs estimated mass; `from_vehicle_config(cfg)` classmethod
  lives in a separate bridging module (e.g. `hopper/scenario_glue.py`), not in `control/`.
- Control law: attitude error from `quat_error(q_des, q)` (rotation vector, body frame) -> desired torque
  `tau = I (kp * e + kd * (omega_des - omega))`, optionally with a feedforward term. Map to actuators:
  - Pitch/yaw torque from TVC: from the S03 convention `tau = [z_com F_y, -z_com F_x, 0]`, solve for the
    lateral thrust components `F_x = -tau_y / z_com`, `F_y = tau_x / z_com`, then gimbal angles
    `gy = asin(F_x / T)`, `gx = -asin(F_y / T)` (small-angle approximations are fine). Protect against
    `T ~ 0` (hold gimbal at 0 when thrust is below a floor) and clamp to limits.
  - Roll torque from the RCS: `roll_torque = clip(tau_z / max_roll_torque, -1, 1)`.
- Tilt-aware inertia: use scheduled inertia from the estimated mass.
- Anti-windup (if an integral term is used) and rate-limit awareness: back off the gain when the
  gimbal is saturated, so there is no limit-cycling.

### `src/hopper/control/attitude_test_controller.py` (or in tests)
A `StateFeedbackController` that wraps `AttitudeController` with a fixed throttle and desired attitude,
used for the tests below and as the base for S09.

## Gain tuning

Document the design in the module docstring: linearise about hover, `I theta_ddot = z_com T gimbal`, giving
a double-integrator with plant gain `K = T z_com / I`. Pick `wn` and `zeta` (suggest bandwidth about 3 rad/s
at the start, well below the gimbal bandwidth of `gimbal_wn`) and set gains accordingly. Include a
small script or notebook cell `scripts/tune_attitude.py` that computes gains from `wn`, `zeta` and prints
them. (A numeric check of margins comes with `python-control` in S13; not needed here.)

## Tests (`tests/control/test_attitude.py`, closed-loop tests via `Simulation`)

- Unit: zero error gives zero torque; a small positive error about x/y/z gives torque of the correct sign.
- Torque-to-gimbal mapping inverts the S03 propulsion model: apply the commanded gimbal in the propulsion
  model and check the produced torque equals the demanded one (within the small-angle error).
- Closed loop, airborne (set start at 50 m altitude with a fixed throttle producing hover thrust; gravity
  on): from a 10 degree initial tilt, attitude converges to < 0.5 degree within 5 s with no overshoot > 30%.
- Closed loop roll: 30 degree initial roll error is removed using RCS.
- Closed loop with command delay of 20 ms and the S03 actuator lag still stable (tilt converges); with a very large
  delay (e.g. 300 ms) the test documents the instability margin (assert the controller degrades rather than
  asserting stability, or skip with a comment on the measured margin).
- Saturation: a 60 degree tilt with limited gimbal does not produce NaN or chatter; the output stays within limits.
- The controller module passes an import-boundary test: `tests/control/test_import_boundary.py` asserts that
  `hopper.control` modules do not import `hopper.dynamics`, `hopper.sim`, or `hopper.sensors`
  (inspect `ast` of the source files).

## Definition of done

- Tests pass; ruff clean; gains documented; import-boundary test in place.
- Progress tracker updated per `docs/AGENT_START.md`.
