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

Smoke tests.

- Wind: the same seed gives bit-identical sequences, different seeds differ, zero sigma gives exactly
  the mean wind, and the gust depends on time only (not on the integrator step).
- Aero builds and gives finite loads for a moving, tilted vehicle in wind; `aero_enabled=False` gives
  zero.
- Gravity can be switched off.

## Definition of done

- Tests pass; ruff clean. Progress tracker updated per `docs/AGENT_START.md`.
