"""Full hop with perfect-state feedback: 10 m up, hover 3 s, translate 3 m, descend, land."""

import numpy as np
import pytest

from hopper.config import load_vehicle
from hopper.control.mission import MissionParams
from hopper.dynamics.mass_model import mass_properties
from hopper.scenario_glue import run_hop
from hopper.sim.events import Outcome


@pytest.fixture(scope="module")
def nominal():
    cfg = load_vehicle("configs/vehicle_default.yaml")
    return cfg, run_hop(cfg, seed=0)


def test_full_hop_lands_on_the_site_softly_and_upright(nominal):
    _, (res, s) = nominal
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 0.5
    assert s.touchdown_speed < 1.0
    assert np.degrees(s.touchdown_tilt) < 5.0


def test_full_hop_reaches_the_target_altitude_and_spends_propellant(nominal):
    cfg, (res, s) = nominal
    assert s.apogee == pytest.approx(MissionParams().target_alt, abs=0.75)
    assert 0.0 < s.fuel_used < cfg.tank.prop_mass_initial
    assert s.flight_time > MissionParams().hover_time


def test_full_hop_hovers_for_the_hover_time_at_altitude(nominal):
    cfg, (res, _) = nominal
    com = np.array([mass_properties(p, cfg).com_z for p in res.log["prop"]])
    foot = res.log["pos_z"].to_numpy() - com - cfg.legs.length
    near = np.abs(foot - 10.0) < 0.5
    assert near.sum() * res.log["t"].diff().median() >= 3.0


def test_full_hop_is_reproducible_for_a_fixed_seed(nominal):
    cfg, (res, _) = nominal
    again, _ = run_hop(cfg, seed=0)
    assert again.log["pos_x"].iloc[-1] == res.log["pos_x"].iloc[-1]
    assert again.t_end == res.t_end


def test_hop_with_20ms_command_delay_still_lands(nominal):
    cfg, _ = nominal
    res, s = run_hop(cfg, seed=0, command_delay=0.020)
    assert res.outcome is Outcome.LANDED
    assert s.landing_error < 0.5
    assert s.touchdown_speed < 1.0


def test_hop_in_3_mps_steady_wind_lands_within_1p5_m(nominal):
    cfg, _ = nominal
    for wind in ((3.0, 0.0, 0.0), (0.0, -3.0, 0.0)):
        res, s = run_hop(cfg, seed=0, wind_mean=wind)
        assert res.outcome is Outcome.LANDED, wind
        assert s.landing_error < 1.5
        assert s.touchdown_speed < 1.0


def test_timeout_abort_mid_climb_levels_and_lands_softly(nominal):
    cfg, _ = nominal
    # ASCEND needs ~7 s for 10 m, so a 4 s state timeout aborts the climb in the air.
    res, s = run_hop(cfg, mission=MissionParams(state_timeout=4.0), seed=0)
    assert res.outcome is Outcome.LANDED
    assert s.apogee < 9.5  # the climb really was cut short
    assert s.touchdown_speed < 1.0
    assert np.degrees(s.touchdown_tilt) < 5.0
