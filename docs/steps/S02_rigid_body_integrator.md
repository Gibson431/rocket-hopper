# S02: Rigid-body dynamics and RK4

**Depends on:** S01. **Effort:** 2 days.

## Context

Core 6-DOF equations of motion and the fixed-step integrator. Read `docs/CONVENTIONS.md`
(frames, state layout). This step knows nothing about rockets yet: it integrates a rigid body
given a total force and torque, so it can be verified against analytic solutions.

## Deliverables

### `src/hopper/dynamics/rigid_body.py`
```python
def rigid_body_derivative(
    x: np.ndarray,        # full 19-state; only POS, VEL, QUAT, OMEGA used here
    mass: float,
    inertia: np.ndarray,  # (3, 3) about CoM, body frame
    force_I: np.ndarray,  # total external force in inertial frame (N), including gravity
    torque_B: np.ndarray, # total torque about CoM in body frame (N m)
) -> np.ndarray:          # returns xdot (19,), with entries outside the rigid-body block = 0
```
Equations:
- `pos_dot = vel`; `vel_dot = force_I / mass`.
- `quat_dot = 0.5 * q (x) [0, omega]` (use `quat_derivative`).
- `omega_dot = I^-1 (torque - omega x (I omega))`.

Mass-rate and inertia-rate terms are handled in S03 via `PROP` dynamics; document that omega_dot here
ignores d(I)/dt and note the neglect in the docstring.

### `src/hopper/sim/integrator.py`
- `rk4_step(f, t, x, dt)` where `f(t, x) -> xdot`. After the step, renormalise the quaternion slice
  (`QUAT`). Do not touch other entries.
- `integrate(f, x0, t0, t1, dt, callback=None)` loops fixed steps, returning final state and an
  optional trajectory array. The final step is shortened so the end time is hit exactly.
- Keep the integrator generic (works on any 1-D numpy state) except for the optional renormalisation,
  which takes the slice as a parameter (default: `QUAT`).

## Tests (`tests/dynamics/test_rigid_body.py`, `tests/sim/test_integrator.py`)

- Free fall (force = `m g`): position and velocity match `z0 - g t^2 / 2` to 1e-9 over 5 s at dt=1e-3.
- Constant force: `x = x0 + v0 t + F t^2 / (2 m)` in every axis.
- Torque-free symmetric top: spin about its symmetry axis stays constant; angular momentum norm in the
  inertial frame is conserved to 1e-9 for an asymmetric body with initial spin about all axes (use
  `L_I = R I omega`).
- Torque-free kinetic energy `0.5 omega . I omega` is conserved to 1e-8.
- Constant torque about a principal axis: `omega = tau t / I`, and the attitude angle equals `tau t^2 / (2 I)`.
- Quaternion norm stays within 1e-12 of 1 after 10 s of tumbling.
- **Convergence:** integrate a nonlinear case (asymmetric tumble) at dt, dt/2, dt/4 against a very fine
  reference. The error ratio between successive halvings should be near 16 (4th order); assert > 10.
- Integration to a time not divisible by dt hits the end time exactly.

## Definition of done

- All tests pass; ruff clean.
- The function signatures above are stable (S03 and S06 call them). Log any change.
- Progress tracker updated per `docs/AGENT_START.md`.
