# S14: Drake vehicle plant, mass properties, state adapters

**Depends on:** S12, S13. **Effort:** 2 days.

## Context

First Drake code: the rigid body and its variable mass. Read `docs/DRAKE_DESIGN.md` (S12 decisions on
body frame, mass updating and integrator), `src/hopper/dynamics/mass_model.py`,
`src/hopper/dynamics/rigid_body.py`, `docs/CONVENTIONS.md`. New code lives in `src/hopper/drake/`
(a new subpackage; `hopper.control` and `hopper.estimation` may not import it or `pydrake`).

## Deliverables

- `hopper/drake/plant.py`: `build_vehicle_plant(cfg) -> (plant, scene_graph?)` with one floating body
  carrying the dry mass properties, plus the geometry placeholders S17 and S21 attach to.
- `hopper/drake/mass.py`: the mass updater, built the way S12 decided. A plain `LeafSystem` cannot
  write another system's parameters, so this is not an ordinary diagram system: it is either a hook
  the sim loop calls between `AdvanceTo` steps or a publish callback with the root context. It takes
  propellant mass, builds the full `SpatialInertia` from `mass_properties` and sets it with
  `SetSpatialInertiaInBodyFrame` (never `SetMass` and `SetCenterOfMassInBodyFrame` separately; that
  breaks the inertia), so there is one mass model.
- `hopper/drake/state.py`: `to_legacy_state(plant, context) -> ndarray(19)` partial view (pos, vel,
  quat, omega) and `set_legacy_state(plant, context, x)`, handling the CoM-vs-body-frame offset and
  Drake's quaternion/velocity frame conventions as decided in S12. The propellant, throttle and gimbal
  slots are owned by S15 systems.
- `hopper/drake/sim.py` (minimal): `make_simulator(diagram, integrator, dt)` configured per S12.

## Tests (`tests/drake/test_plant.py`, `test_state.py`)

- State round trip: random state -> Drake -> state equals the input to 1e-12 for position, velocity,
  quaternion (up to sign) and body rate, for non-zero CoM offsets and rotated vehicles.
- Free fall matches `z = z0 - g t^2 / 2` and the S13 golden 1 to the S12-proposed tolerance.
- Torque-free spin: angular momentum conserved; a body spinning about a non-principal axis matches
  the legacy rigid-body derivative integrated at a fine step; kinetic energy conserved.
- A pure force applied at the pivot gives the legacy CoM acceleration and torque (`r x F`, sign
  pinned like the S03 tests).
- `MassUpdater`: for propellant 20..0 kg the plant's mass, CoM and inertia equal `mass_properties` to
  1e-12, including the early CoM dip noted in S03 Decisions.
- Import boundary: `hopper.control`, `hopper.estimation`, `hopper.types`, `hopper.math3d` never import
  `pydrake` (extend `tests/control/test_import_boundary.py` and the estimation one).

## Definition of done

- Tests pass; ruff clean; `uv sync` and CI green with the dependency.
- Parity tolerances used are written in the test file with a comment pointing to the S12 measurement.
- Progress tracker updated per `docs/AGENT_START.md`; update `docs/CONVENTIONS.md` with the Drake
  package layout and the body-frame/state-conversion rule.
