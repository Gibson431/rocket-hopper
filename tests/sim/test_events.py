import numpy as np
import pytest

from hopper.dynamics.contact import ContactOutput
from hopper.math3d import quat_from_axis_angle
from hopper.sim.events import EventMonitor, EventState, Outcome, check_events
from hopper.types import make_state

DT = 0.01


def contact(touching=True, base_strike=False, n=4):
    forces = np.full(n, 100.0 if touching else 0.0)
    return ContactOutput(
        force_I=np.array([0, 0, forces.sum()]),
        torque_B=np.zeros(3),
        foot_forces=forces,
        any_contact=touching,
        base_strike=base_strike,
    )


def state(pos=(0, 0, 1.0), vel=(0, 0, 0), tilt=0.0, omega=(0, 0, 0)):
    q = quat_from_axis_angle([0, 1, 0], tilt)
    return make_state(pos, vel, q, omega, 10.0)


def run(cfg, steps, t0=0.0):
    """steps: iterable of (x, contact). Returns the list of outcomes."""
    mon = EventMonitor(cfg)
    return [mon.update(t0 + i * DT, x, c) for i, (x, c) in enumerate(steps)]


def test_nothing_happens_in_flight(cfg):
    out = run(cfg, [(state(pos=(0, 0, 5), vel=(0, 0, -2)), contact(False))] * 50)
    assert all(o is None for o in out)


def test_landed_requires_one_second_of_calm_contact(cfg):
    calm = (state(pos=(0, 0, 1.46)), contact())
    out = run(cfg, [calm] * 150)
    assert out.index(Outcome.LANDED) == pytest.approx(round(cfg.sim.land_hold / DT), abs=1)
    assert all(o is None for o in out[: out.index(Outcome.LANDED)])


def test_landed_hold_timer_resets_when_a_condition_is_violated(cfg):
    calm = (state(), contact())
    shaky = (state(omega=(0.5, 0, 0)), contact())
    out = run(cfg, [calm] * 80 + [shaky] + [calm] * 80)
    assert Outcome.LANDED not in out
    out = run(cfg, [calm] * 80 + [shaky] + [calm] * 110)
    assert out.index(Outcome.LANDED) > 80 + 100


@pytest.mark.parametrize(
    "bad",
    [
        state(vel=(0.3, 0, 0)),
        state(vel=(0, 0, -0.3)),
        state(tilt=np.deg2rad(12)),
        state(omega=(0, 0.3, 0)),
    ],
)
def test_landed_needs_slow_level_and_still_vehicle(cfg, bad):
    assert Outcome.LANDED not in run(cfg, [(bad, contact())] * 300)


def test_landed_needs_all_feet_in_contact(cfg):
    c = contact()
    c.foot_forces[0] = 0.0
    assert Outcome.LANDED not in run(cfg, [(state(), c)] * 300)


def test_fast_touchdown_crashes_but_a_slow_one_does_not(cfg):
    fast = [(state(vel=(0, 0, -5.0)), contact(False))] * 3 + [(state(vel=(0, 0, -4.0)), contact())]
    assert run(cfg, fast)[-1] is Outcome.CRASHED
    slow = [(state(vel=(0, 0, -1.0)), contact(False))] * 3 + [(state(vel=(0, 0, -0.9)), contact())]
    assert run(cfg, slow)[-1] is None


def test_crash_uses_the_approach_speed_not_the_cushioned_post_contact_speed(cfg):
    steps = [(state(vel=(0, 0, -4.0)), contact(False)), (state(vel=(0, 0, -0.5)), contact())]
    assert run(cfg, steps)[-1] is Outcome.CRASHED


def test_touchdown_with_large_tilt_crashes(cfg):
    steps = [(state(tilt=np.deg2rad(35)), contact(False)), (state(tilt=np.deg2rad(35)), contact())]
    assert run(cfg, steps)[-1] is Outcome.CRASHED
    steps = [(state(tilt=np.deg2rad(20)), contact(False)), (state(tilt=np.deg2rad(20)), contact())]
    assert run(cfg, steps)[-1] is None


def test_base_strike_is_a_crash(cfg):
    assert run(cfg, [(state(), contact(False, base_strike=True))])[-1] is Outcome.CRASHED


def test_tilt_past_sixty_degrees_in_contact_is_tipped_over(cfg):
    upright = (state(), contact())
    assert run(cfg, [upright, (state(tilt=np.deg2rad(65)), contact())])[-1] is Outcome.TIPPED_OVER
    assert run(cfg, [upright, (state(tilt=np.deg2rad(50)), contact())])[-1] is None
    # Large tilt in free flight is not a tip-over.
    assert run(cfg, [(state(tilt=np.deg2rad(65)), contact(False))])[-1] is None


def test_out_of_bounds_by_range_or_altitude(cfg):
    far = state(pos=(cfg.sim.max_range + 1, 0, 5))
    high = state(pos=(0, 0, cfg.sim.max_alt + 1))
    near = state(pos=(cfg.sim.max_range - 1, 0, cfg.sim.max_alt - 1))
    assert run(cfg, [(far, contact(False))])[-1] is Outcome.OUT_OF_BOUNDS
    assert run(cfg, [(high, contact(False))])[-1] is Outcome.OUT_OF_BOUNDS
    assert run(cfg, [(near, contact(False))])[-1] is None


def test_timeout_at_max_time(cfg):
    mon = EventMonitor(cfg)
    x = state(pos=(0, 0, 5))
    assert mon.update(cfg.sim.max_time - 0.01, x, contact(False)) is None
    assert mon.update(cfg.sim.max_time, x, contact(False)) is Outcome.TIMEOUT


def test_check_events_carries_touchdown_history_in_the_prev_state_it_is_given(cfg):
    prev = EventState()
    assert check_events(0.0, state(vel=(0, 0, -5.0)), contact(False), cfg, prev) is None
    assert check_events(0.01, state(vel=(0, 0, -1.0)), contact(), cfg, prev) is Outcome.CRASHED
