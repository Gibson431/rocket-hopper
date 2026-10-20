"""Closed loop on Drake: multi-rate controller, delay, determinism, the perfect-state hop."""

import time

import numpy as np
import pytest

from hopper.analysis.logger import COLUMNS
from hopper.config import load_vehicle
from hopper.drake.simulation import DrakeSimulation
from hopper.dynamics.environment import Environment, Wind
from hopper.dynamics.mass_model import mass_properties
from hopper.scenario_glue import run_hop
from hopper.sim.engine import SimResult
from hopper.sim.events import Outcome
from hopper.types import Command, make_state

HOVER_Z = 20.0  # CoM height: far from the ground for every short run below


class CountingController:
    """Calls recorded; returns `late` from time `t_switch` on, else engine off."""

    def __init__(self, late: Command | None = None, t_switch: float = 0.0):
        self.times: list[float] = []
        self.resets = 0
        self._late, self._t_switch = late, t_switch

    def reset(self):
        self.resets += 1

    def step_state(self, est, t):
        self.times.append(t)
        if self._late is not None and t >= self._t_switch:
            return self._late
        return Command(0.0, np.zeros(2), 0.0)


def make_sim(cfg, controller, seed=0, wind=None, gravity=False, aero=False, **kw):
    rng = np.random.default_rng(seed)
    wind = wind or Wind.none()
    env = Environment(wind=wind, gravity_enabled=gravity, aero_enabled=aero)
    x0 = make_state([0, 0, HOVER_Z], [0, 0, 0], None, [0, 0, 0], cfg.tank.prop_mass_initial)
    return DrakeSimulation(cfg, controller, env, x0, rng, **kw)


def test_run_returns_a_sim_result_with_the_s06_columns_and_the_event_on_the_terminal_row(cfg):
    ctrl = CountingController()
    sim = make_sim(cfg, ctrl, gravity=True)  # free fall from 20 m: crashes at about 2 s
    res = sim.run(10.0)
    assert isinstance(res, SimResult)
    assert list(res.log.columns) == COLUMNS
    assert res.outcome is Outcome.CRASHED
    assert res.log["event"].iloc[-1] == "CRASHED"
    assert (res.log["event"].iloc[:-1] == "").all()
    assert res.log.attrs["seed"] == 0
    assert len(res.log.attrs["config_hash"]) == 16
    assert res.t_end == res.log["t"].iloc[-1]


def test_run_to_t_end_without_an_event_has_running_outcome_and_no_event_name(cfg):
    res = make_sim(cfg, CountingController()).run(0.5)
    assert res.outcome is Outcome.RUNNING
    assert res.t_end == pytest.approx(0.5)
    assert res.log["event"].iloc[-1] == ""
    assert len(res.log) == 501  # t = 0 .. 0.5 on the 1 ms grid


def test_ten_seconds_at_100_hz_makes_exactly_1000_controller_calls_on_the_grid(cfg):
    ctrl = CountingController()
    res = make_sim(cfg, ctrl).run(10.0)
    assert len(ctrl.times) == 1000
    assert ctrl.times[0] == 0.0
    # the terminal row (t = 10 s) is not a controller call
    assert max(ctrl.times) == pytest.approx(9.99)
    assert np.allclose(np.diff(ctrl.times), 0.01, atol=1e-9)
    assert res.log["t"].iloc[-1] == pytest.approx(10.0)


def test_controller_is_reset_once_per_run(cfg):
    ctrl = CountingController()
    make_sim(cfg, ctrl).run(0.1)
    assert ctrl.resets == 1


def test_controller_is_not_called_on_the_terminal_row(cfg):
    ctrl = CountingController()
    res = make_sim(cfg, ctrl, gravity=True).run(10.0)
    assert res.outcome is Outcome.CRASHED
    assert max(ctrl.times) < res.t_end - 1e-9


def first_row_with_throttle(res, value):
    return int(np.flatnonzero(res.log["cmd_throttle"].to_numpy() == value)[0])


@pytest.mark.parametrize("delay_steps", [0, 1, 7, 25])
def test_a_delay_of_k_steps_applies_the_command_k_steps_later(cfg, delay_steps):
    late = Command(0.5, np.zeros(2), 0.0)
    ctrl = CountingController(late, t_switch=0.05)  # first nonzero command computed at step 50
    res = make_sim(cfg, ctrl, command_delay=delay_steps * cfg.sim.dt).run(0.2)
    assert first_row_with_throttle(res, 0.5) == 50 + delay_steps


def test_delay_zero_acts_on_the_step_the_command_was_computed(cfg):
    late = Command(0.5, np.zeros(2), 0.0)
    res = make_sim(cfg, CountingController(late, t_switch=0.05)).run(0.2)
    log = res.log
    # the thrust fraction (lagged state) is still 0 at step 50 and rises right after it
    assert log["thr"].iloc[50] == 0.0
    assert log["thr"].iloc[51] > 0.0
    assert log["cmd_throttle"].iloc[49] == 0.0
    assert log["cmd_throttle"].iloc[50] == 0.5


def test_commands_are_held_between_controller_calls(cfg):
    late = Command(0.5, np.array([0.01, -0.02]), 0.3)
    res = make_sim(cfg, CountingController(late, t_switch=0.05)).run(0.2)
    held = res.log.iloc[50:]
    assert (held["cmd_throttle"] == 0.5).all()
    assert (held["cmd_gim_x"] == 0.01).all()
    assert (held["cmd_gim_y"] == -0.02).all()
    assert (held["cmd_roll"] == 0.3).all()


def test_negative_delay_is_rejected(cfg):
    with pytest.raises(ValueError):
        make_sim(cfg, CountingController(), command_delay=-0.001)


def test_log_decimation_keeps_every_nth_row_and_the_terminal_row(cfg):
    res = make_sim(cfg, CountingController(), gravity=True, log_decimation=10).run(10.0)
    t = res.log["t"].to_numpy()
    assert res.outcome is Outcome.CRASHED
    assert np.allclose(t[:-1], np.arange(len(t) - 1) * 0.01)
    assert t[-1] == pytest.approx(res.t_end)
    assert res.log["event"].iloc[-1] == "CRASHED"


def gusty_run(cfg, seed):
    late = Command(0.5, np.zeros(2), 0.0)
    wind = Wind(np.array([2.0, 0.0, 0.0]), 1.5, 1.0, np.random.default_rng(seed))
    sim = make_sim(cfg, CountingController(late), seed=seed, wind=wind, aero=True)
    return sim.run(1.0).log


def test_same_seed_gives_a_bit_identical_log_and_different_seeds_differ_with_wind(cfg):
    a, b, c = gusty_run(cfg, 3), gusty_run(cfg, 3), gusty_run(cfg, 4)
    assert a.equals(b)
    assert not a.equals(c)
    assert not np.array_equal(a["vel_x"].to_numpy(), c["vel_x"].to_numpy())


def test_a_second_run_in_the_same_process_does_not_depend_on_the_first(cfg):
    first = gusty_run(cfg, 3)
    gusty_run(cfg, 9)
    assert gusty_run(cfg, 3).equals(first)


def test_logged_state_starts_at_the_initial_state_and_mass_follows_the_propellant(cfg):
    late = Command(1.0, np.zeros(2), 0.0)
    log = make_sim(cfg, CountingController(late, t_switch=0.0)).run(0.5).log
    first = log.iloc[0]
    assert first["pos_z"] == pytest.approx(HOVER_Z, abs=1e-9)
    assert first["prop"] == pytest.approx(cfg.tank.prop_mass_initial)
    assert first["quat_w"] == pytest.approx(1.0)
    last = log.iloc[-1]
    assert last["prop"] < first["prop"]
    assert last["mass"] == pytest.approx(mass_properties(last["prop"], cfg).mass, abs=1e-9)
    assert last["thrust"] == pytest.approx(cfg.engine.thrust_max * last["thr"])


def test_controller_log_hook_columns_are_appended_before_the_event_column(cfg):
    class Logging(CountingController):
        log_columns = ("probe_a", "probe_b")

        def log_values(self):
            return (float(len(self.times)), 2.0)

    res = make_sim(cfg, Logging()).run(0.05)
    assert list(res.log.columns) == [*COLUMNS[:-1], "probe_a", "probe_b", "event"]
    assert (res.log["probe_b"] == 2.0).all()
    assert res.log["probe_a"].iloc[0] == 1.0  # the controller had been called once at row 0


def test_sensor_mode_and_faults_are_not_available_until_s19(cfg):
    with pytest.raises(NotImplementedError):
        make_sim(cfg, CountingController(), perfect_state=False)


# --- the perfect-state hop (S09 acceptance thresholds) ---------------------------------------


@pytest.fixture(scope="module")
def hop():
    cfg = load_vehicle("configs/vehicle_default.yaml")
    t0 = time.perf_counter()
    res, s = run_hop(cfg, seed=0, backend="drake")
    wall = time.perf_counter() - t0
    print(f"\nDRAKE HOP wall {wall:.1f} s, {res.t_end / cfg.sim.dt / wall:.0f} steps/s")
    return cfg, res, s


def test_drake_hop_lands_on_the_site_softly_and_upright(hop):
    _, res, s = hop
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 0.5
    assert s.touchdown_speed < 1.0
    assert np.degrees(s.touchdown_tilt) < 5.0


def test_drake_hop_with_20ms_delay_still_lands(hop):
    cfg, _, _ = hop
    res, s = run_hop(cfg, seed=0, command_delay=0.020, backend="drake")
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 0.5
    assert s.touchdown_speed < 1.0


def test_drake_hop_in_3_mps_wind_still_lands(hop):
    cfg, _, _ = hop
    res, s = run_hop(cfg, seed=0, wind_mean=(3.0, 0.0, 0.0), backend="drake")
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 1.5
    assert s.touchdown_speed < 1.0
