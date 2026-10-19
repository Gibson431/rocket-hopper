# S15: Propulsion and actuators as Drake systems

**Depends on:** S14. **Effort:** 2 days. Can run in parallel with S16 and S17.

## Context

Port `src/hopper/dynamics/propulsion.py`: throttled, gimballed thrust with first-order throttle lag,
second-order gimbal with rate and angle limits, RCS roll torque, and propellant consumption by the
rocket equation. Read it and its tests (`tests/dynamics/test_propulsion.py`), the geometry section of
`docs/CONVENTIONS.md`, and `docs/DRAKE_DESIGN.md`.

## Deliverables

- `hopper/drake/propulsion.py`: `Engine(LeafSystem)`.
  - Input: `Command` as a vector port (throttle, gx, gy, roll_torque); input: plant state port only if
    needed for the force direction.
  - Continuous state: propellant, thrust fraction, gimbal angles and rates (the legacy slots 13..18),
    with the same limit logic (rate clamp, zero outward acceleration at the angle limit).
  - Outputs: `ExternallyAppliedSpatialForce` list for the plant (thrust at the gimbal pivot, plus RCS
    torque), propellant mass for `MassUpdater`, actual thrust (N) for the log.
  - Engine-off and empty-tank behaviour identical to legacy (thrust zero when propellant is gone).
  - Keep the hot path in plain floats (S12 throughput finding); no per-call allocation of Drake
    objects beyond what the port requires.
- Reuse `propulsion()` as the single source of truth if S12 shows the Python cost is acceptable;
  otherwise reimplement and keep the legacy function as a test oracle until S22.

## Tests (`tests/drake/test_propulsion.py`)

- For 200 random (state, command) pairs the engine's force, torque and state derivative equal
  `propulsion()` to 1e-10.
- Rocket equation: constant throttle burn in zero gravity and no aero gives delta-v within the
  S04-era tolerance of `isp * g0 * ln(m0/m1)`.
- Throttle lag: step response time constant equals `throttle_tau`; gimbal step stays within the angle
  and rate limits and settles per `gimbal_wn`/`gimbal_zeta` (S03 tests re-expressed against the
  diagram).
- Sign checks: positive `gy` thrust direction and torque about -y_B exactly as pinned in S03.
- Empty tank: thrust and mass flow are zero and the state does not go negative.
- S13 goldens 2-4 (burns, throttle and gimbal steps) match within the S12 tolerances.

## Definition of done

- Tests pass; ruff clean; throughput of the engine system recorded in the progress notes.
- Progress tracker updated per `docs/AGENT_START.md`.
