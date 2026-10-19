# S17: Legs, ground contact and outcome events on Drake

**Depends on:** S14. **Effort:** 2-3 days. The riskiest physics step; can run in parallel with S15
and S16.

## Context

Replace `src/hopper/dynamics/contact.py` (four spring-damper legs, regularised Coulomb friction,
base-strike detection) and `src/hopper/sim/events.py` (outcome classification). Read both, their
tests, the S05 Decisions in `docs/PROGRESS.md` (the V_EPS = 0.05 stability finding), and
`docs/DRAKE_DESIGN.md` (S12 contact and integrator decisions).

## Deliverables

- `hopper/drake/legs.py`: adds ground geometry, leg and base collision geometry and the contact model
  chosen in S12 to the plant from `LegConfig` (count, radius, length, stiffness, damping, friction).
  Parameters map one-to-one from the existing config; document how stiffness and damping translate
  to Drake's contact parameters if they do not map directly. Series stiffness: set the foot and
  ground stiffness so the combined value equals `legs.stiffness`. Tune Hunt-Crossley
  dissipation until the vehicle lands without bouncing or jitter. There is no need to match the
  legacy leg law; the S12 fallback (`contact_loads` as an applied force) is only for stability.
- Contact observation system: per-foot normal force, feet-in-contact count, total contact force and
  base-strike flag as outputs (they feed the log and the events).
- `hopper/drake/events.py`: the same `Outcome` classification and thresholds from `SimConfig`
  (`v_crash`, `crash_tilt`, `tipover_tilt`, `land_*`, `max_range`, `max_alt`, timeout, the `has_flown`
  guard and the pre-contact vertical speed rule). Implemented so it can be a `Simulator` monitor
  that terminates the run. Reuse `hopper.sim.events.check_events` if its inputs can be filled from
  Drake outputs; otherwise port and keep the same tests.

## Tests (`tests/drake/test_legs.py`, `test_events.py`)

Smoke tests for the contact model; the events are flight-logic and keep their tests.

- A vehicle dropped from 0.5 m settles upright on the pad (finite, no energy gain, rest height
  sensible) and does not jump or jitter; a hard 5 m/s landing stays numerically stable.
- Events: each `Outcome` is produced by a constructed state exactly as in `tests/sim/test_events.py`
  (port those tests to the Drake outputs), with the same precedence order. They define when a run
  counts as landed or crashed, which the controller scenarios depend on.

## Definition of done

- Tests pass; ruff clean. The contact parameters chosen are recorded in Decisions.
- Progress tracker updated per `docs/AGENT_START.md`.
