import numpy as np
import pytest
from pydrake.all import DiagramBuilder
from test_legs import state_with_lowest_foot

from hopper.drake.events import DrakeEventMonitor
from hopper.drake.legs import ContactObserver, add_legs
from hopper.drake.plant import build_vehicle_plant
from hopper.drake.sim import make_simulator
from hopper.drake.state import set_legacy_state
from hopper.sim.events import Outcome
from hopper.types import OMEGA, make_state

DT = 0.01


class Rig:
    """A Drake plant with legs and a monitor; states are set by hand and time stepped manually."""

    def __init__(self, cfg):
        self.cfg = cfg
        builder = DiagramBuilder()
        plant, sg = build_vehicle_plant(cfg, builder, finalize=False)
        legs = add_legs(plant, sg, cfg)
        plant.Finalize()
        obs = builder.AddSystem(ContactObserver(plant, legs))
        builder.Connect(plant.get_state_output_port(), obs.get_plant_state_input_port())
        builder.Connect(
            plant.get_contact_results_output_port(), obs.get_contact_results_input_port()
        )
        self.diagram = builder.Build()
        self.plant, self.obs = plant, obs
        self.monitor = DrakeEventMonitor(plant, obs, cfg)
        self.ctx = self.diagram.CreateDefaultContext()

    def run(self, steps, t0=0.0):
        out = []
        for i, x in enumerate(steps):
            set_legacy_state(
                self.plant, self.plant.GetMyMutableContextFromRoot(self.ctx), x, self.cfg
            )
            self.ctx.SetTime(t0 + i * DT)
            out.append(self.monitor.evaluate(self.ctx))
        return out


def run(cfg, steps):
    return Rig(cfg).run(steps)


def fly(cfg, **kw):
    return state_with_lowest_foot(cfg, -1.0, **kw)


def rest(cfg, omega=(0, 0, 0), **kw):
    x = state_with_lowest_foot(cfg, 0.008, **kw)
    x[OMEGA] = omega
    return x


def test_nothing_happens_in_flight(cfg):
    assert all(o is None for o in run(cfg, [fly(cfg, vel=(0, 0, -2))] * 50))


def test_landed_requires_the_hold_time_of_calm_contact(cfg):
    out = run(cfg, [fly(cfg)] + [rest(cfg)] * 150)
    assert out.index(Outcome.LANDED) == pytest.approx(1 + round(cfg.sim.land_hold / DT), abs=1)
    assert all(o is None for o in out[: out.index(Outcome.LANDED)])


def test_resting_on_the_pad_before_flight_is_not_a_landing(cfg):
    assert Outcome.LANDED not in run(cfg, [rest(cfg)] * 300)


def test_landed_hold_timer_resets_when_a_condition_is_violated(cfg):
    shaky = make_state(rest(cfg)[:3], [0, 0, 0], None, [0.5, 0, 0], 20.0)
    out = run(cfg, [fly(cfg)] + [rest(cfg)] * 80 + [shaky] + [rest(cfg)] * 80)
    assert Outcome.LANDED not in out


@pytest.mark.parametrize(
    "bad",
    [
        dict(vel=(0.3, 0, 0)),
        dict(vel=(0, 0, -0.3)),
        dict(tilt=np.deg2rad(12)),
        dict(omega=(0, 0.3, 0)),
    ],
)
def test_landed_needs_slow_and_level(cfg, bad):
    assert Outcome.LANDED not in run(cfg, [fly(cfg)] + [rest(cfg, **bad)] * 300)


def test_landed_needs_all_feet_in_contact(cfg):
    one_foot = state_with_lowest_foot(cfg, 0.004, tilt=np.deg2rad(5))
    assert Outcome.LANDED not in run(cfg, [fly(cfg)] + [one_foot] * 300)


def test_fast_touchdown_crashes_but_a_slow_one_does_not(cfg):
    fast = [fly(cfg, vel=(0, 0, -5.0))] * 3 + [rest(cfg, vel=(0, 0, -4.0))]
    assert run(cfg, fast)[-1] is Outcome.CRASHED
    slow = [fly(cfg, vel=(0, 0, -1.0))] * 3 + [rest(cfg, vel=(0, 0, -0.9))]
    assert run(cfg, slow)[-1] is None


def test_crash_uses_the_approach_speed_not_the_cushioned_post_contact_speed(cfg):
    steps = [fly(cfg, vel=(0, 0, -4.0)), rest(cfg, vel=(0, 0, -0.5))]
    assert run(cfg, steps)[-1] is Outcome.CRASHED


def test_touchdown_with_large_tilt_crashes(cfg):
    for deg, expected in ((35, Outcome.CRASHED), (20, None)):
        tilt = np.deg2rad(deg)
        steps = [fly(cfg, tilt=tilt), state_with_lowest_foot(cfg, 0.004, tilt=tilt)]
        assert run(cfg, steps)[-1] is expected


def test_base_strike_is_a_crash(cfg):
    x = make_state([0, 0, 0.8], [0, 0, 0], None, [0, 0, 0], 20.0)  # pivot about 0.05 m below ground
    assert run(cfg, [fly(cfg), x])[-1] is Outcome.CRASHED


def test_tilt_past_sixty_degrees_in_contact_is_tipped_over(cfg):
    big = state_with_lowest_foot(cfg, 0.004, tilt=np.deg2rad(65))
    mid = state_with_lowest_foot(cfg, 0.004, tilt=np.deg2rad(50))
    assert run(cfg, [rest(cfg), big])[-1] is Outcome.TIPPED_OVER
    assert run(cfg, [rest(cfg), mid])[-1] is None
    assert run(cfg, [fly(cfg, tilt=np.deg2rad(65))])[-1] is None  # in flight it is not a tip-over


def test_out_of_bounds_by_range_or_altitude(cfg):
    def at(pos):
        return make_state(pos, [0, 0, 0], None, [0, 0, 0], 20.0)

    far = at([cfg.sim.max_range + 1, 0, 5])
    high = at([0, 0, cfg.sim.max_alt + 1])
    near = at([cfg.sim.max_range - 1, 0, cfg.sim.max_alt - 1])
    assert run(cfg, [far])[-1] is Outcome.OUT_OF_BOUNDS
    assert run(cfg, [high])[-1] is Outcome.OUT_OF_BOUNDS
    assert run(cfg, [near])[-1] is None


def test_timeout_at_max_time(cfg):
    rig = Rig(cfg)
    assert rig.run([fly(cfg)], t0=cfg.sim.max_time - 0.01)[-1] is None
    assert rig.run([fly(cfg)], t0=cfg.sim.max_time)[-1] is Outcome.TIMEOUT


def test_crash_outranks_out_of_bounds(cfg):
    x = make_state([cfg.sim.max_range + 5, 0, 0.8], [0, 0, 0], None, [0, 0, 0], 20.0)
    assert run(cfg, [x])[-1] is Outcome.CRASHED


def test_repeated_evaluation_at_the_same_time_does_not_change_the_touchdown_history(cfg):
    rig = Rig(cfg)
    rig.run([fly(cfg, vel=(0, 0, -5.0))])
    set_legacy_state(
        rig.plant,
        rig.plant.GetMyMutableContextFromRoot(rig.ctx),
        rest(cfg, vel=(0, 0, -4.0)),
        cfg,
    )
    rig.ctx.SetTime(DT)
    assert rig.monitor.evaluate(rig.ctx) is Outcome.CRASHED
    assert rig.monitor.evaluate(rig.ctx) is Outcome.CRASHED


def _drop_with_monitor(cfg, height, vz=0.0, t_end=6.0):
    rig = Rig(cfg)
    sim = make_simulator(rig.diagram)
    ctx = sim.get_mutable_context()
    x = state_with_lowest_foot(cfg, -height, vel=(0, 0, vz))
    set_legacy_state(rig.plant, rig.plant.GetMyMutableContextFromRoot(ctx), x, cfg)
    sim.set_monitor(rig.monitor.as_simulator_monitor(rig.diagram))
    sim.Initialize()
    sim.AdvanceTo(t_end)
    return rig.monitor.outcome, ctx.get_time()


def test_a_gentle_drop_ends_the_run_as_landed(cfg):
    outcome, t = _drop_with_monitor(cfg, 0.05)
    assert outcome is Outcome.LANDED
    assert 1.0 < t < 2.0  # touchdown near 0.1 s, then the 1 s hold


def test_a_hard_landing_ends_the_run_as_crashed_immediately(cfg):
    outcome, t = _drop_with_monitor(cfg, 0.02, vz=-5.0)
    assert outcome is Outcome.CRASHED
    assert t < 0.1
