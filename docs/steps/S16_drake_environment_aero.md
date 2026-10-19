# S16: Environment, wind and aerodynamics on Drake

**Depends on:** S14. **Effort:** 1-2 days. Can run in parallel with S15 and S17.

## Context

Port gravity, the ISA atmosphere, the Gauss-Markov wind and the aero loads. Read
`src/hopper/dynamics/environment.py`, `src/hopper/dynamics/aero.py`, their tests,
`docs/DRAKE_DESIGN.md` (S12 decisions on randomness and the atmosphere).

## Deliverables

- Gravity: the plant's own uniform gravity field, with a switch matching
  `Environment(gravity_enabled=False)`.
- `hopper/drake/wind.py`: `WindSource(LeafSystem)` producing the wind vector at time t for the
  Ornstein-Uhlenbeck gust plus mean wind. Randomness follows the S12 decision (explicit seed, no
  global state). The exact discretisation of `Wind.advance` is kept; the gust must be a function of
  time only (read-only inside integrator stages).
- `hopper/drake/aero.py`: `AeroLoads(LeafSystem)`: inputs plant state, wind and propellant mass;
  output an `ExternallyAppliedSpatialForce` (force at the CP, converted to the plant body frame).
  Atmosphere density as in `atmosphere()` (0-11 km, clamped); `aero_enabled` switch.
- Combining several force sources onto the plant's single applied-force port is done by a small
  `ForceCombiner` system (engine, aero, ...); S17 does not need it, S18 wires it.

## Tests (`tests/drake/test_wind.py`, `test_aero.py`)

- Gravity: free fall acceleration is `g` to 1e-9; disabled gives zero.
- Wind: same seed gives bit-identical sequences; different seeds differ; mean and variance over 200
  samples match `gust_sigma` and `gust_tau` (loose, as in the S04 tests); zero sigma gives exactly the
  mean wind; independent of the integrator step used.
- Aero: 100 random (velocity, attitude, wind) cases equal `aero_loads` (force and torque about the
  CoM) to 1e-10; moving vehicle in wind and tilted cross-flow torque cases from the S04 backlog are
  included.
- Terminal velocity (with the S04 `cd_axial = 5` setup) matches the analytic value within 1%.
- S13 golden 1 (free fall with aero and wind) matches within the S12 tolerance.

## Definition of done

- Tests pass; ruff clean. Progress tracker updated per `docs/AGENT_START.md`.
