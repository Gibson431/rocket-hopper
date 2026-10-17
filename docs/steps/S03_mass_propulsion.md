# S03: Mass model and propulsion

**Depends on:** S02. **Effort:** 2 days.

## Context

Adds the rocket-specific parts: propellant-dependent mass/CoM/inertia, and the engine with throttle
lag, gimbal dynamics, and mass flow. Read `docs/CONVENTIONS.md`, especially "Gimbal and thrust geometry"
and the vehicle config groups. The thrust torque sign convention is pinned by tests here.

## Deliverables

### `src/hopper/dynamics/mass_model.py`
```python
@dataclass(frozen=True)
class MassProperties:
    mass: float
    com_z: float            # CoM height above base along z_B (m)
    inertia: np.ndarray     # (3, 3) about the CoM, body frame

def mass_properties(prop_mass: float, cfg: VehicleConfig) -> MassProperties
```
- Dry structure: point CoM at `dry_com_z`, diagonal `dry_inertia` about its own CoM.
- Propellant: uniform solid cylinder of radius `tank.radius`, filled from the tank bottom upward to a
  height proportional to `prop_mass / prop_mass_initial * tank.height`. Its CoM is at the middle of the
  filled column; inertia about its own CoM from the cylinder formulas.
- Combine with the parallel-axis theorem about the total CoM. Clamp `prop_mass` to `>= 0`.

### `src/hopper/dynamics/propulsion.py`
```python
def propulsion(x: np.ndarray, cmd: Command, cfg: VehicleConfig, mp: MassProperties) -> PropulsionOutput
```
Returns `force_B (3,)`, `torque_B (3,)`, and the derivatives of the actuator states
(`PROP`, `THR`, `GIM`, `GIMR` slots) as a 19-vector `xdot_act` (other entries 0).
- Commanded throttle: `0` means off; otherwise clamp to `[thrust_min_frac, 1]`. A commanded value in
  `(0, thrust_min_frac)` is clamped up to the minimum (the engine cannot throttle below it).
- Thrust fraction state `THR`: first-order lag toward the commanded fraction, `thr_dot = (cmd - thr) / throttle_tau`.
  Thrust magnitude `= thrust_max * THR`. Thrust is zero when `PROP <= 0` (flameout); the lag state decays to 0.
- Gimbal: second-order per axis, `th_ddot = wn^2 (cmd - th) - 2 zeta wn th_dot`, commanded angle
  clamped to `+/- gimbal_max`, rate clamped to `+/- gimbal_rate_max` (when at the rate limit and still
  accelerating outward, zero the acceleration). Also clamp the angle to its limit.
- Thrust direction `d_B` and force/torque from CONVENTIONS.md: `F_B = T d_B`, `tau = r x F_B` with
  `r = [0, 0, -com_z]`. Add the RCS roll torque: `tau_z += roll_torque_cmd * max_roll_torque`
  (clamped to [-1, 1]; RCS needs no propellant in v1).
- Mass flow: `PROP_dot = -T / (isp * g0)` with `g0 = 9.80665`.

### `src/hopper/dynamics/vehicle.py` (first version)
`vehicle_derivative(t, x, cmd, cfg, env_force_I=None, env_torque_B=None)` that:
1. computes `mass_properties(x[PROP])`,
2. gets propulsion force/torque,
3. adds gravity `[0, 0, -mass*g]` (flat, constant `g = 9.80665` for now; S04 generalises),
4. calls `rigid_body_derivative`, and merges the actuator derivatives.
Rotates `force_B` into the inertial frame with `quat_rotate`. S04 and S05 extend this function with
aero, wind, and contact through the optional hooks, so keep the structure simple and documented.

Known approximation (document it): inertia rate and CoM motion terms are neglected. They are small
for a slow burn.

## Tests (`tests/dynamics/test_mass_model.py`, `tests/dynamics/test_propulsion.py`, `tests/dynamics/test_vehicle.py`)

- Full tank: total mass = dry + propellant; empty tank: equals dry properties exactly.
- CoM lies between the dry CoM and the tank centre; moves monotonically as propellant burns.
- Inertia is symmetric positive definite and increases with propellant (about x/y, z) in the expected direction.
- Parallel-axis check: place the propellant at a known offset and compare against a hand calculation.
- Thrust at steady throttle `t` equals `thrust_max * t`; zero with zero propellant.
- Throttle step response: after `throttle_tau`, `THR` reaches ~63% of the step (analytic, 1%).
- Gimbal step response is bounded by the rate and angle limits; settles to the commanded value for small steps.
- **Torque sign convention:** with `gy > 0`, torque about y_B is negative; with `gx > 0` torque about x_B is
  negative (matches CONVENTIONS.md); zero gimbal gives zero torque.
- Thrust vector magnitude equals `thrust_max * THR` for any gimbal angles.
- **Rocket equation:** vertical burn with fixed throttle and no gravity: delta-v matches
  `Isp g0 ln(m0/m1)` within 0.1%. (Gravity-off flag for tests, e.g. `cfg`-independent argument `g=0`.)
- Hover: choose throttle so `thrust = m g` with the vehicle upright; the vertical acceleration is ~0.
- Roll torque produces a pure z-axis angular acceleration `tau / Izz`.

## Definition of done

- All tests pass; ruff clean.
- `vehicle_derivative` has the documented signature and a docstring listing what it neglects.
- Progress tracker updated per `docs/AGENT_START.md`.
