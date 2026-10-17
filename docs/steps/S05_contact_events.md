# S05: Ground contact and events

**Depends on:** S04. **Effort:** 2 days.

## Context

The hopper must take off from and land on the ground, so legs and ground contact are needed, plus
event detection for outcomes (touchdown, crash, tipover). Read `docs/CONVENTIONS.md` and
`src/hopper/dynamics/vehicle.py`. Ground is the plane z = 0 in ENU.

## Deliverables

### `src/hopper/dynamics/contact.py`
```python
def contact_loads(x, mp: MassProperties, cfg: LegsConfig) -> ContactOutput
# ContactOutput: force_I (3,), torque_B (3,), foot_forces (n,), any_contact: bool
```
- Feet are at body positions (relative to the CoM) `p_i = [r cos(a_i), r sin(a_i), -(com_z + legs.length)]`
  with `r = legs.radius`, `a_i = 2 pi i / count`. The base (engine pivot) is at `-com_z`, so the feet sit
  `legs.length` below it. Document the geometry in the docstring.
- Foot position in I: `pos + R @ p_i`; foot velocity: `vel + R @ (omega x p_i)`.
- Penetration `d = -z_foot` (positive when below ground). If `d > 0`: normal force
  `N = max(0, k d - c v_z_foot)` upward (z), no tension (clamp at zero).
- Friction: regularised Coulomb, tangential force opposes foot slip velocity, magnitude
  `mu * N * tanh(|v_t| / v_eps)` with `v_eps = 0.01 m/s`.
- Torque about the CoM: `r_i x F_i` per foot, rotated into the body frame.
- Also check the base (engine bell) for ground strikes: if the base point (`z = 0` on the vehicle, at the
  axis) goes below ground, report `base_strike=True` (used as a crash event; no contact force needed).

### Extend `vehicle_derivative`
Add `contact_loads` forces and torque. Expose `ContactOutput` info for logging via an optional
output mutable (`info` dict) to avoid recomputation by the logger.

### `src/hopper/sim/events.py`
```python
class Outcome(Enum): RUNNING, LANDED, CRASHED, TIPPED_OVER, OUT_OF_BOUNDS, TIMEOUT, FLAMEOUT_FALLING
def check_events(t, x, contact: ContactOutput, cfg, prev_state) -> Outcome | None
```
- `LANDED`: all feet in contact, speed `< 0.2 m/s`, tilt `< 10 deg`, angular rate `< 0.2 rad/s`, held for 1.0 s.
  (Hold-time tracking needs state; implement as a small class `EventMonitor` with `update(t, x, contact)`.)
- `CRASHED`: any touchdown with vertical speed `> v_crash` (default 3 m/s) or tilt `> 30 deg` at first
  contact, or `base_strike`.
- `TIPPED_OVER`: tilt `> 60 deg` while in contact.
- `OUT_OF_BOUNDS`: horizontal distance from the origin `> max_range` or altitude `> max_alt`
  (config values in `SimConfig`, add them).
- Thresholds live in `SimConfig` (add fields with defaults), not hard-coded.

## Tests (`tests/dynamics/test_contact.py`, `tests/sim/test_events.py`)

- Drop test: vehicle placed with feet just above the ground and zero velocity, no thrust, integrated for
  5 s. It settles; the steady foot deflection satisfies `sum(k d) = m g` (1%). It does not drift laterally.
- No contact force when all feet are above ground; no tension when separating.
- Friction: a vehicle pushed sideways while standing decelerates and stops; static hold on a (flat) plane
  produces no creep beyond a small tolerance.
- Tilted vehicle: tilting past the leg-footprint angle (`atan(radius / cg_height)`) while released from rest
  tips over (tilt keeps growing).
- Energy: landing from a small drop, total mechanical energy never increases (damping), within tolerance.
- Events: scripted state sequences produce each `Outcome`; the `LANDED` hold timer resets if a condition
  is violated; `CRASHED` triggers on a fast touchdown but not a slow one.
- A thrust-assisted lift-off at throttle above weight leaves the ground (contact goes to zero).

## Definition of done

- Tests pass; ruff clean. Contact is stable at the default dt = 1 ms with the default config
  (no jitter: foot normal force at rest varies < 1%).
- If dt = 1 ms is not stable for the chosen stiffness/damping, document the recommended contact substep or
  adjust the default stiffness and damping in the default YAML and log the decision.
- Progress tracker updated per `docs/AGENT_START.md`.
