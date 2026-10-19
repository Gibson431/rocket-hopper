# S14: Drake vehicle plant, mass properties, state adapters

**Depends on:** S12. **Effort:** 1-2 days.

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

Smoke tests only; Drake is the source of truth for the physics (see PROGRESS Decisions).

- `build_vehicle_plant` builds and finalises; a diagram with it runs 1 s of free fall with finite state.
- The state adapters round-trip (a handful of states, including rotated vehicles and a non-zero CoM
  offset): the controller sees positions, velocities, quaternion and body rate in the CONVENTIONS frames.
  This one is a test of our own conversion code, not of Drake.
- `MassUpdater` writes the `mass_properties` mass, CoM and inertia for propellant 20 and 0 kg.
- Import boundary: `hopper.control`, `hopper.estimation`, `hopper.types`, `hopper.math3d` never import
  `pydrake` (extend `tests/control/test_import_boundary.py` and the estimation one).

## Definition of done

- Tests pass; ruff clean; `uv sync` and CI green with the dependency.
- Progress tracker updated per `docs/AGENT_START.md`; update `docs/CONVENTIONS.md` with the Drake
  package layout and the body-frame/state-conversion rule.
