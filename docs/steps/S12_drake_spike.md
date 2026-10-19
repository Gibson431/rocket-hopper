# S12: Drake feasibility spike and design decisions

**Depends on:** S11. **Effort:** 1-2 days. **This is a gate:** S14-S21 assume its answers. If a
finding makes the port unattractive, mark the step `blocked` with the evidence and stop.

## Context

The plan ports the *simulation side* (dynamics, environment, contact, integration, sim loop, sensors
plumbing, faults, visualisation) from the hand-rolled code to Drake (`pydrake`). Flight software
(`hopper.control`, `hopper.estimation`, `hopper.types`, `hopper.math3d`) stays hand-written and must
never import `pydrake`. Read `docs/PLAN.md` ("Drake port"), `docs/CONVENTIONS.md`,
`src/hopper/sim/engine.py`, `src/hopper/dynamics/vehicle.py`, `src/hopper/dynamics/contact.py`.

Already checked while planning (pydrake 1.58, Python 3.12): there is no RK4 integrator in the bindings
(`RungeKutta2Integrator`, `RungeKutta3Integrator` and the error-controlled ones exist);
`RigidBody.SetMass/SetCenterOfMassInBodyFrame/SetSpatialInertiaInBodyFrame(context, ...)` exist, so
mass properties are context parameters; the plant has a body-spatial-acceleration output port.
Re-verify what you rely on.

## Deliverables

Throwaway spike code goes in `scripts/spikes/` (ruff-excluded already). The lasting output is
`docs/DRAKE_DESIGN.md`: one short section per question, each ending in **Decision:** and the evidence
(numbers). Add the Drake dependency with `uv add drake` and confirm `uv sync`, `uv run pytest` and CI
still work on the interpreter uv resolves for this project.

Questions to answer:

1. **Variable mass.** Can a single floating `MultibodyPlant` body follow the changing mass, CoM and
   inertia by writing the context parameters from a propellant-driven system each step (discrete
   update) while the plant integrates? Measure error against the legacy `mass_properties` over a full
   burn. Fallback options: a second body for the propellant, or a custom `LeafSystem` rigid body.
   Known from the planning spike (re-verify):
   - A `LeafSystem` event can only modify its own state, so a `MassUpdater` cannot write the plant's
     parameters through normal diagram wiring. Compare (a) an outer Python loop that calls
     `AdvanceTo(t + dt)` and sets the parameters between steps (worked, about 10k steps/s with a
     per-step update); (b) a periodic publish callback holding the root context (worked, but bends
     Drake's const-publish contract); (c) mass as state in a custom body. Pick one and say why.
   - Use `SetSpatialInertiaInBodyFrame` with a full `SpatialInertia.MakeFromCentralInertia`. Calling
     `SetMass` and `SetCenterOfMassInBodyFrame` separately keeps the unit inertia about the body origin
     fixed, so moving the CoM produced a non-positive-definite mass matrix.
   - Parameters change between steps, so they are constant inside each integrator step; compare the
     error with the legacy continuous `mass_properties` (which also neglects dI/dt).
   - Isolate plant error from forcing error: drive both sims from the same continuous engine states
     (do not replay logged forces held constant over a step; that alone gave about 1 cm / 1 cm/s at 1 s).
2. **Where the body frame sits.** Legacy state is the CoM with the body frame at the CoM; Drake's body
   frame is fixed on the body. Choose the Drake body frame (suggest: engine gimbal pivot at the base)
   and define the exact conversion to and from the 19-element legacy state, including the quaternion
   and angular-velocity frame conventions of `QuaternionFloatingJoint` (verify, do not assume).
3. **Continuous vs discrete plant.** Compare `MultibodyPlant(time_step=0)` with an explicit integrator
   against a discrete plant (`time_step=1e-3`, SAP/TAMSI). Record accuracy against legacy free flight,
   whether a fixed-step explicit integrator is stable with the leg contact, and the cost.
4. **Leg contact.** Reproduce the legacy leg behaviour (4 legs, spring 30 kN/m, damper 1.5 kN s/m,
   Coulomb friction) with Drake: (a) point contact on foot spheres with tuned stiffness/dissipation,
   (b) prismatic leg joints with spring-damper and a foot sphere. Drop the vehicle from 0.5 m and
   compare settle height, peak force and rebound. Pick one. Also evaluate (c): keep the legacy
   `contact_loads` law as a Python applied-force element (exact parity; Drake only integrates).
   Known from the planning spike (re-verify):
   - Drake point contact stiffness combines **in series** across the two geometries (30 kN/m on both
     gives 15 kN/m): set foot and ground values so the series stiffness is the config value.
   - Hunt-Crossley dissipation scales with penetration, so it cannot reproduce the legacy linear damper
     (1500 N s/m, damping ratio about 0.87 per leg). A 0.5 m drop with 30 kN/m on both geometries
     and default-ish dissipation (0.05 s/m) bounced to 0.38 m with 0.136 m peak penetration; legacy is
     heavily damped. Expect tuning to match settle height and rebound but not the force history, and
     record which metrics are matched.
   - Friction regularisation is Stribeck (`stiction_tolerance`), not `tanh(v / 0.05)`; re-check the S05
     stability finding (V_EPS) with the chosen integrator.
   If (a)/(b) cannot meet the S17 tolerances, (c) is the default.
5. **Integrator.** Choose the integrator and step (the project rule "fixed-step RK4" cannot hold
   literally). Prefer a fixed step so runs stay reproducible and logs stay on the legacy 1 ms grid.
   Show convergence on a free-flight case.
6. **Accelerometer source.** Which Drake quantity gives specific force (all non-gravity forces / mass)
   for the IMU model, in continuous and in discrete mode.
7. **Randomness and determinism.** Wind gust and sensor noise must still come from explicit
   `numpy.random.Generator`s seeded per run. Decide how: pre-generated sequences handed to systems,
   or generator state held outside the context. Show two identical runs are bit-identical and that the
   result does not depend on how many other simulations ran in the process.
8. **Throughput.** Python `LeafSystem` callbacks are slow. Build the smallest diagram with a
   propulsion-like Python system on the continuous path and measure steps/s against legacy
   (~1300 steps/s, ~1 EKF hop/min/worker). Say what has to be in C++ or kept off the hot path.
   Planning spike (RK3, 1 ms, free flight): plant plus 1, 3, 6 Python force systems and a combiner ran
   at about 12k, 7.5k and 4.3k steps/s; legacy open loop is 4-6k steps/s and about 70% of a legacy
   hop is physics. So expect **parity, not a speedup**; the port does not fix Monte Carlo throughput.
   Set the S18 budget as "no slower than 0.7x legacy" unless you measure otherwise.
9a. **Controller timing.** Simultaneous discrete updates in Drake all read the pre-update state. Show
   how a periodic controller update, the command delay and the physics step order, so that delay 0
   applies the new command from the same step (legacy semantics). Expected answer: controller and
   delay live in one system, so a separate delay system would add one step of latency.
9. **Multiprocessing.** Confirm a Drake diagram is built inside each worker from the scenario (Drake
   objects are not picklable) and that `ProcessPoolExecutor` runs are independent.

## Tests

None required for spike code. If a decision can be pinned by a small test that belongs in the final
design (for example the state conversion round trip), write it for S14 instead and list it in the doc.

## Definition of done

- `docs/DRAKE_DESIGN.md` answers all questions (1-9 and 9a), and records the Drake wheel footprint
  (about 150 MB pydrake plus a transitive 46 MB `mosek`; Linux and macOS wheels only, to the best of
  our knowledge) and the interpreter used with a Decision and measurements.
- Tolerances for the parity tests in S14-S20 are proposed from measured legacy self-convergence
  (legacy at `dt` vs `dt/2`), not guessed.
- `uv run pytest` and `uv run ruff check .` pass with the dependency added.
- Go/no-go recorded in Decisions. Progress tracker updated per `docs/AGENT_START.md`.
