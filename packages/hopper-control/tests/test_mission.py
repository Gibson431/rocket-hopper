import numpy as np
import pytest
from hopper_control.mission import (
    MissionInputs,
    MissionParams,
    MissionState,
    MissionStateMachine,
)

S = MissionState
DT = 0.01


def params(**kw):
    base = dict(
        target_alt=10.0,
        hover_time=3.0,
        target_xy=(3.0, 0.0),
        ascent_rate=2.0,
        descent_rate=2.0,
        final_descent_rate=0.5,
        touchdown_alt=3.0,
        cutoff_alt=0.05,
    )
    base.update(kw)
    return MissionParams(**base)


class Driver:
    """Feeds scripted inputs to a state machine and records the state history."""

    def __init__(self, mp=None):
        self.mp = mp or params()
        self.sm = MissionStateMachine(self.mp)
        self.t = 0.0
        self.alt, self.vz, self.xy, self.vxy, self.tilt = 0.0, 0.0, np.zeros(2), np.zeros(2), 0.0
        self.out = None

    def tick(self, n=1, **kw):
        for k, v in kw.items():
            setattr(self, k, v)
        for _ in range(n):
            self.out = self.sm.update(
                MissionInputs(
                    self.t, self.alt, self.vz, np.array(self.xy), np.array(self.vxy), self.tilt
                )
            )
            self.t += DT
        return self.out

    def run_for(self, seconds, **kw):
        return self.tick(int(round(seconds / DT)), **kw)


def drive_to(d, state):
    """Scripted inputs that carry the machine into `state`."""
    d.tick()
    if state == S.IDLE:
        return d
    for _ in range(20000):
        if d.sm.state == state:
            return d
        s = d.sm.state
        if s in (S.IDLE, S.IGNITION):
            d.tick()
        elif s == S.ASCEND:
            d.tick(alt=d.out.alt_ref, vz=0.0)  # a perfect tracker of the reference
        elif s == S.HOVER:
            d.tick()
        elif s == S.TRANSLATE:
            d.tick(xy=d.mp.target_xy, vxy=(0.0, 0.0))
        elif s == S.DESCEND:
            d.tick(alt=d.mp.touchdown_alt - 0.1, vz=-d.mp.descent_rate)
        elif s == S.TOUCHDOWN:
            d.tick(alt=0.0, vz=-0.5)
        elif s == S.CUTOFF:
            d.tick(alt=0.0, vz=0.0)
        elif s == S.LANDED:
            break
    raise AssertionError(f"never reached {state}, stuck in {d.sm.state}")


def ramp_done(d):
    """Follow the altitude reference (still climbing) until it reaches the target."""
    for _ in range(20000):
        if d.out.alt_ref >= d.mp.target_alt - 1e-6:
            return d
        d.tick(alt=d.out.alt_ref, vz=1.0)
    raise AssertionError("altitude reference never reached the target")


def test_starts_in_idle_with_engine_off():
    d = Driver()
    out = d.tick()
    assert out.state == S.IDLE and out.engine_cut


def test_idle_moves_to_ignition_at_start_time():
    d = Driver(params(start_time=1.0))
    d.run_for(0.9)
    assert d.sm.state == S.IDLE
    d.run_for(0.2)
    assert d.sm.state in (S.IGNITION, S.ASCEND)


def test_ignition_runs_engine_then_ascends_after_ignition_time():
    d = drive_to(Driver(), S.IGNITION)
    assert not d.out.engine_cut
    d.run_for(d.mp.ignition_time + 0.1)
    assert d.sm.state == S.ASCEND


def test_ascend_commands_the_ascent_rate_and_a_rising_altitude_reference():
    d = drive_to(Driver(), S.ASCEND)
    d.run_for(2.0, alt=0.0, vz=0.0)
    first = d.out
    d.run_for(2.0, alt=2.0, vz=2.0)
    assert d.out.alt_ref > first.alt_ref
    assert 0.0 < d.out.vz_ref <= d.mp.ascent_rate + 1e-9


def test_vertical_reference_acceleration_is_limited():
    d = drive_to(Driver(), S.ASCEND)
    prev = d.out.vz_ref
    for _ in range(200):
        d.tick(alt=0.0, vz=0.0)
        assert abs(d.out.vz_ref - prev) <= d.mp.ref_accel * DT + 1e-12
        prev = d.out.vz_ref


def test_ascend_needs_altitude_and_low_speed_to_enter_hover():
    d = ramp_done(drive_to(Driver(), S.ASCEND))
    d.run_for(1.0, alt=10.0, vz=1.0)  # at altitude but still climbing fast
    assert d.sm.state == S.ASCEND
    d.run_for(1.0, alt=9.0, vz=0.0)  # slow but not there yet
    assert d.sm.state == S.ASCEND
    d.run_for(1.0, alt=10.0, vz=0.0)
    assert d.sm.state == S.HOVER


def test_hover_transition_has_hysteresis_against_a_single_noisy_sample():
    d = ramp_done(drive_to(Driver(), S.ASCEND))
    d.tick(alt=10.0, vz=0.0)  # one good sample
    assert d.sm.state == S.ASCEND
    d.tick(alt=8.0, vz=0.0)  # condition lost: the timer restarts
    d.run_for(d.mp.settle_time * 0.6, alt=10.0, vz=0.0)
    assert d.sm.state == S.ASCEND


def test_hover_lasts_hover_time_then_translates():
    d = drive_to(Driver(), S.HOVER)
    d.run_for(d.mp.hover_time - 0.2, alt=10.0, vz=0.0)
    assert d.sm.state == S.HOVER
    assert np.allclose(d.out.xy_ref, 0.0)
    d.run_for(0.4, alt=10.0, vz=0.0)
    assert d.sm.state == S.TRANSLATE
    assert np.allclose(d.out.xy_ref, d.mp.target_xy)


def test_hover_skips_translate_when_the_site_is_overhead():
    d = Driver(params(target_xy=(0.0, 0.0)))
    drive_to(d, S.HOVER)
    d.run_for(d.mp.hover_time + 0.2, alt=10.0, vz=0.0)
    assert d.sm.state == S.DESCEND


def test_translate_waits_for_position_and_speed_then_descends():
    d = drive_to(Driver(), S.TRANSLATE)
    d.run_for(1.0, alt=10.0, xy=(1.0, 0.0), vxy=(0.0, 0.0))
    assert d.sm.state == S.TRANSLATE
    d.run_for(1.0, alt=10.0, xy=(3.0, 0.0), vxy=(1.5, 0.0))
    assert d.sm.state == S.TRANSLATE
    d.run_for(1.0, alt=10.0, xy=(3.0, 0.1), vxy=(0.0, 0.0))
    assert d.sm.state == S.DESCEND


def test_descend_commands_descent_rate_in_velocity_mode_and_keeps_the_site():
    d = drive_to(Driver(), S.DESCEND)
    d.run_for(3.0, alt=8.0, vz=-1.0, xy=(3.0, 0.0))
    assert d.out.alt_ref is None
    assert d.out.vz_ref < 0.0 and d.out.vz_ref >= -d.mp.descent_rate - 1e-9
    assert np.allclose(d.out.xy_ref, d.mp.target_xy)


def test_descend_reaches_descent_rate_before_touchdown_altitude():
    d = drive_to(Driver(), S.DESCEND)
    d.run_for(4.0, alt=8.0, vz=-1.0)
    assert d.out.vz_ref == pytest.approx(-d.mp.descent_rate)


def test_below_touchdown_altitude_switches_to_final_descent_rate():
    d = drive_to(Driver(), S.DESCEND)
    d.run_for(4.0, alt=8.0, vz=-2.0)
    d.run_for(0.1, alt=d.mp.touchdown_alt - 0.2, vz=-2.0)
    assert d.sm.state == S.TOUCHDOWN
    d.run_for(5.0, alt=2.0, vz=-1.0)
    assert d.out.vz_ref == pytest.approx(-d.mp.final_descent_rate)


def test_touchdown_cuts_the_engine_at_cutoff_altitude():
    d = drive_to(Driver(), S.TOUCHDOWN)
    d.run_for(0.5, alt=0.5, vz=-0.5)
    assert d.sm.state == S.TOUCHDOWN and not d.out.engine_cut
    d.tick(alt=d.mp.cutoff_alt - 0.01, vz=-0.5)
    assert d.sm.state == S.CUTOFF and d.out.engine_cut


def test_cutoff_goes_to_landed_once_at_rest():
    d = drive_to(Driver(), S.CUTOFF)
    d.run_for(0.3, alt=0.0, vz=-0.3)
    assert d.sm.state == S.CUTOFF
    d.run_for(d.mp.settle_time + 1.0, alt=0.0, vz=0.0)
    assert d.sm.state == S.LANDED and d.out.engine_cut


def test_landed_is_terminal():
    d = drive_to(Driver(), S.LANDED)
    d.run_for(5.0, alt=5.0, vz=3.0, tilt=1.0)
    assert d.sm.state == S.LANDED


# --- timeouts and abort -------------------------------------------------------------------


@pytest.mark.parametrize("state", [S.ASCEND, S.TRANSLATE, S.DESCEND, S.TOUCHDOWN])
def test_state_timeout_triggers_abort(state):
    d = drive_to(Driver(), state)
    stuck = dict(alt=5.0, vz=0.0, xy=(0.0, 0.0), vxy=(5.0, 0.0))
    # keep the guards unsatisfied for longer than any timeout
    d.run_for(d.mp.state_timeout + 1.0, **stuck)
    assert d.sm.state == S.ABORT


def test_excess_tilt_triggers_abort():
    d = drive_to(Driver(), S.HOVER)
    d.run_for(0.5, alt=10.0, vz=0.0, tilt=d.mp.abort_tilt + 0.05)
    assert d.sm.state == S.ABORT


def test_single_tilt_spike_below_the_limit_does_not_abort():
    d = drive_to(Driver(), S.HOVER)
    d.tick(alt=10.0, vz=0.0, tilt=d.mp.abort_tilt - 0.05)
    assert d.sm.state == S.HOVER


def test_altitude_error_beyond_bound_triggers_abort():
    d = drive_to(Driver(), S.HOVER)
    d.run_for(1.0, alt=10.0 - d.mp.abort_alt_error - 1.0, vz=0.0)
    assert d.sm.state == S.ABORT


def test_abort_high_up_levels_and_descends_then_cuts_at_contact():
    d = drive_to(Driver(), S.HOVER)
    d.run_for(0.5, alt=10.0, vz=0.0, tilt=d.mp.abort_tilt + 0.05)
    assert d.sm.state == S.ABORT
    d.tick(alt=8.0, vz=-1.0)
    assert not d.out.engine_cut
    assert d.out.level
    assert d.out.alt_ref is None and d.out.vz_ref < 0.0
    d.run_for(3.0, alt=1.0, vz=-1.0)
    assert d.out.vz_ref == pytest.approx(-d.mp.final_descent_rate)
    d.tick(alt=d.mp.cutoff_alt - 0.01, vz=-0.5)
    assert d.sm.state == S.CUTOFF and d.out.engine_cut


def test_abort_when_low_cuts_the_engine_immediately():
    d = drive_to(Driver(), S.ASCEND)
    d.run_for(0.5, alt=0.5, vz=1.0, tilt=d.mp.abort_tilt + 0.1)
    assert d.sm.state in (S.ABORT, S.CUTOFF)
    assert d.out.engine_cut


def test_abort_is_not_entered_from_idle():
    d = Driver()
    d.run_for(0.3, tilt=2.0)
    assert d.sm.state == S.IDLE


def test_reset_returns_to_idle():
    d = drive_to(Driver(), S.DESCEND)
    d.sm.reset()
    assert d.sm.state == S.IDLE


def test_same_inputs_give_the_same_transitions():
    a = drive_to(Driver(), S.LANDED)
    b = drive_to(Driver(), S.LANDED)
    assert a.t == b.t
