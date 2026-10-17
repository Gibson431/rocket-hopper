# S04: Environment and aerodynamics

**Depends on:** S03. **Effort:** 1-2 days.

## Context

Adds gravity model options, atmosphere, wind, and simple drag/weathercock aerodynamics to
`vehicle_derivative`. Read `docs/CONVENTIONS.md` and the existing `src/hopper/dynamics/vehicle.py`
(from S03). For a hopper (low altitude, low speed) the model should be simple but correctly oriented.

## Deliverables

### `src/hopper/dynamics/environment.py`
- `gravity(pos_I) -> np.ndarray`: constant `[0, 0, -9.80665]` (flat Earth; keep a function so it can change).
- `atmosphere(alt_m) -> (rho, temperature, pressure, speed_of_sound)`: ISA troposphere (0-11 km).
  Also provide `pressure_to_altitude(p)` inverse for the baro sensor in S07.
- `Wind` class: `Wind(mean_I: np.ndarray, gust_sigma: float, gust_tau: float, rng)`; method
  `velocity(t, pos_I) -> np.ndarray` returns mean + a first-order Gauss-Markov gust per axis (discrete-time
  update with a fixed internal step; deterministic given the rng). `Wind.reset(seed)` to restart. A
  `Wind.none()` constructor returns zero wind. Gust update must be callable from inside RK4 stages
  without double-stepping: update the gust state in the sim loop (S06) via `wind.advance(dt)`, and make
  `velocity()` a pure read of the current state.

### `src/hopper/dynamics/aero.py`
```python
def aero_loads(vel_I, quat, wind_I, rho, mp: MassProperties, cfg: AeroConfig) -> tuple[np.ndarray, np.ndarray]
# returns (force_I, torque_B)
```
- Relative air velocity `v_rel_I = vel_I - wind_I`, in body: `v_B`.
- Axial/lateral decomposition along z_B. Quasi-steady drag: `F = -0.5 rho |v| (Cd_axial A_axial v_z z_hat + Cd_lateral A_lateral v_lat)`
  (acts opposite to the relative velocity components; magnitude `0.5 rho Cd A v^2`).
- Torque from the lateral force acting at the centre of pressure `cp_z` (above base) relative to the CoM:
  `r = [0, 0, cp_z - com_z]`, `tau = r x F_B`. If CP is above the CoM, the vehicle weathercocks (stable).
- Zero loads at zero relative velocity; no NaN at v=0.

### `src/hopper/dynamics/vehicle.py` (extend)
Replace constant gravity with `environment.gravity`, add aero and wind. New signature:
`vehicle_derivative(t, x, cmd, cfg, env: Environment)` where `Environment` is a small dataclass holding
`wind: Wind` and a `gravity_enabled` flag (useful for rocket-equation tests). Keep S03 tests passing by
providing `Environment.default()` (no wind, gravity on) and updating the callers in those tests.

## Tests (`tests/dynamics/test_environment.py`, `tests/dynamics/test_aero.py`, update `test_vehicle.py`)

- ISA sea level: rho 1.225, T 288.15 K, p 101325 Pa (1e-3 relative); at 1000 m matches published values (1%).
  `pressure_to_altitude(atmosphere(h).pressure) == h` to 1e-6.
- Wind with zero sigma returns exactly the mean; with sigma > 0, over many samples the std approaches
  sigma (fixed seed, 10% tolerance) and the autocorrelation decays with tau.
- Drag opposes velocity; magnitude `0.5 rho Cd A v^2` (1e-9). Axial and lateral cases separately.
- Weathercock: lateral wind on an upright vehicle with CP above the CoM produces a torque turning the nose
  into the relative wind; CP below the CoM flips the sign.
- Terminal velocity of a dropped vehicle (no thrust) matches `sqrt(2 m g / (rho Cd A))` within 1% after
  settling (short vehicle, big dt tolerable).
- A constant crosswind on a hovering vehicle pushes it downwind.
- Previous S03 tests still pass.

## Definition of done

- Tests pass; ruff clean.
- `Environment` and updated `vehicle_derivative` signature documented in the module docstring.
- Progress tracker updated per `docs/AGENT_START.md`.
