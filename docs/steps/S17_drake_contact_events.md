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
  ground stiffness so the combined value equals `legs.stiffness`. Hunt-Crossley dissipation is
  depth-proportional and cannot equal the legacy linear damper; tune it, and if the tolerances
  below cannot be met, use the S12 fallback (legacy `contact_loads` as a Python applied-force
  element, with `ContactOutput` feeding the observers).
- Contact observation system: per-foot normal force, feet-in-contact count, total contact force and
  base-strike flag as outputs (they feed the log and the events).
- `hopper/drake/events.py`: the same `Outcome` classification and thresholds from `SimConfig`
  (`v_crash`, `crash_tilt`, `tipover_tilt`, `land_*`, `max_range`, `max_alt`, timeout, the `has_flown`
  guard and the pre-contact vertical speed rule). Implemented so it can be a `Simulator` monitor
  that terminates the run. Reuse `hopper.sim.events.check_events` if its inputs can be filled from
  Drake outputs; otherwise port and keep the same tests.

## Tests (`tests/drake/test_legs.py`, `test_events.py`)

- Static rest: the vehicle settles at the legacy equilibrium height and total contact force equals
  weight within 1%; each foot carries a quarter on level ground.
- Drop from 0.5 m (S13 golden 5): settle height and rebound count within the S12 tolerances; tilted
  drop ends upright or tips as in legacy. Peak force and force history are compared only if S12
  showed the chosen contact model can match them; otherwise assert a physical bound (peak below the
  crash limit, energy never increases) and record the measured difference in Decisions.
- Friction: a sliding vehicle stops at the Coulomb distance `v^2 / (2 mu g)` within 5%; no yaw limit
  cycle after settling (yaw rate below 1e-3 rad/s after 3 s, the S05 regression).
- No contact above the reach of the legs; base strike flags when the base touches.
- Events: each `Outcome` is produced by a constructed state exactly as in `tests/sim/test_events.py`
  (port those tests to the Drake outputs), with the same precedence order.
- Stability: the chosen integrator and step stay stable (no energy gain) for a 10 s resting run and a
  hard 5 m/s landing.

## Definition of done

- Tests pass; ruff clean. Deviations from the legacy contact (for example a different friction
  regularisation) are recorded in Decisions with the measured effect.
- Progress tracker updated per `docs/AGENT_START.md`.
