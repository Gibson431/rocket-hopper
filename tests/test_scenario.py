from pathlib import Path

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from hopper.control.mission import MissionParams
from hopper.drake.simulation import DrakeSimulation
from hopper.scenario import (
    Scenario,
    build_simulation,
    load_scenario,
    run_scenario,
    sample_dispersions,
)
from hopper.scenario_glue import run_hop
from hopper.sim.events import Outcome
from hopper.sim.faults import GimbalSlow, GimbalStuck, GpsOutage

SCENARIOS = Path(__file__).parent.parent / "configs" / "scenarios"
PROVIDED = [
    "hop_nominal",
    "hop_wind",
    "hop_gps_outage",
    "hop_gimbal_fault",
    "hop_gimbal_stuck",
    "hop_dispersed",
]
QUICK = {"target_alt": 3.0, "hover_time": 0.5, "target_xy": [1.0, 0.0]}


def _write(tmp_path, **overrides):
    d = {
        "name": "t",
        "vehicle": str(SCENARIOS.parent / "vehicles" / "vehicle_default.yaml"),
        "mission": QUICK,
    }
    d.update(overrides)
    p = tmp_path / "s.yaml"
    p.write_text(yaml.safe_dump(d))
    return p


def test_minimal_scenario_loads_with_defaults(tmp_path):
    s = load_scenario(_write(tmp_path))
    assert s.estimation in ("perfect", "ekf")
    assert s.mission.target_alt == 3.0
    assert s.faults == () and s.dispersions.is_empty()


@pytest.mark.parametrize(
    "bad",
    [
        {"bogus": 1},
        {"mission": {"target_alt": 3.0, "bogus": 1}},
        {"wind": {"mean": [1, 0, 0], "gust_sigma": 1.0, "gust_tau": 3.0, "bogus": 1}},
        {"initial": {"pos": [0, 0, 0], "bogus": 1}},
        {"faults": [{"type": "no_such_fault", "at": 1.0}]},
        {"faults": [{"type": "engine_out", "at": 1.0, "bogus": 1}]},
        {"dispersions": {"bogus": {"dist": "normal", "sigma": 1.0}}},
        {"dispersions": {"com_z": {"dist": "normal", "sigma": 0.02, "bogus": 1}}},
        {"estimation": "magic"},
    ],
)
def test_unknown_or_invalid_keys_are_rejected(tmp_path, bad):
    with pytest.raises(ValidationError):
        load_scenario(_write(tmp_path, **bad))


@pytest.mark.parametrize("name", PROVIDED)
def test_provided_scenarios_load(name):
    s = load_scenario(SCENARIOS / f"{name}.yaml")
    assert isinstance(s, Scenario) and s.name == name


def test_provided_scenarios_cover_the_listed_features():
    by = {n: load_scenario(SCENARIOS / f"{n}.yaml") for n in PROVIDED}
    assert any(isinstance(f, GpsOutage) for f in by["hop_gps_outage"].faults)
    assert any(isinstance(f, GimbalSlow) for f in by["hop_gimbal_fault"].faults)
    assert any(isinstance(f, GimbalStuck) for f in by["hop_gimbal_stuck"].faults)
    assert not by["hop_dispersed"].dispersions.is_empty()
    assert by["hop_wind"].wind.mean != (0.0, 0.0, 0.0)
    assert by["hop_nominal"].faults == () and by["hop_nominal"].dispersions.is_empty()


def test_scenario_mission_defaults_match_mission_params(tmp_path):
    p = _write(tmp_path)
    d = yaml.safe_load(p.read_text())
    del d["mission"]
    p.write_text(yaml.safe_dump(d))
    assert load_scenario(p).mission_params == MissionParams()


def test_build_simulation_returns_a_drake_simulation_with_the_faults(tmp_path):
    faults = [{"type": "engine_out", "at": 5.0}, {"type": "gps_outage", "start": 1, "duration": 2}]
    sim = build_simulation(load_scenario(_write(tmp_path, faults=faults)), seed=3)
    assert isinstance(sim, DrakeSimulation)
    assert [f.type for f in sim.faults] == ["engine_out", "gps_outage"]
    assert sim.seed == 3


@pytest.mark.parametrize("name", PROVIDED)
def test_provided_scenarios_run_for_one_second(name):
    scenario = load_scenario(SCENARIOS / f"{name}.yaml")
    got = run_scenario(scenario.model_copy(update={"t_max": 1.0}), dispersed=True)
    assert got.result.outcome is Outcome.RUNNING
    # the estimate columns are NaN until the first controller call
    truth = got.result.log.drop(columns="event").filter(regex=r"^(?!est_|sig_)")
    assert np.isfinite(truth.to_numpy(dtype=float)).all()


def test_seed_argument_overrides_the_scenario_seed(tmp_path):
    s = load_scenario(_write(tmp_path, seed=7))
    assert build_simulation(s).seed == 7
    assert build_simulation(s, seed=9).seed == 9


def test_initial_condition_sets_the_start_state(tmp_path):
    init = {"pos": [2.0, -1.0, 0.0], "vel": [0, 0, 0], "tilt_deg": 0.0, "yaw_deg": 90.0}
    sim = build_simulation(load_scenario(_write(tmp_path, initial=init)))
    assert sim.x0[0:2] == pytest.approx([2.0, -1.0])
    assert sim.x0[6:10] == pytest.approx([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])


def test_nominal_scenario_run_is_bit_identical_to_a_direct_run(tmp_path, cfg):
    scenario = load_scenario(_write(tmp_path, seed=2, estimation="perfect"))
    got = run_scenario(scenario)
    res, _ = run_hop(cfg, MissionParams(**{**QUICK, "target_xy": (1.0, 0.0)}), seed=2)
    assert got.result.outcome is res.outcome
    assert got.result.log.equals(res.log)


def test_quick_hop_scenario_lands(tmp_path):
    got = run_scenario(load_scenario(_write(tmp_path)))
    assert got.result.outcome is Outcome.LANDED
    assert got.summary.landing_error < 0.5


def test_landing_error_is_measured_from_the_commanded_site_when_starting_off_origin(tmp_path):
    init = {"pos": [4.0, 0.0, 0.0]}
    got = run_scenario(load_scenario(_write(tmp_path, initial=init)))
    assert got.result.outcome is Outcome.LANDED
    assert got.summary.landing_error < 0.5
    assert got.result.log.pos_x.iloc[-1] == pytest.approx(5.0, abs=0.5)


DISP = {
    "dry_mass": {"dist": "normal", "sigma_frac": 0.03},
    "com_z": {"dist": "normal", "sigma": 0.02},
    "thrust_scale": {"dist": "normal", "sigma_frac": 0.03},
    "wind_mean": {"dist": "uniform", "low": [-4, -4, 0], "high": [4, 4, 0]},
    "gyro_bias": {"dist": "normal", "sigma_dps": 0.1},
    "initial_pos": {"dist": "normal", "sigma": [0.2, 0.2, 0]},
}


def test_dispersion_samples_have_the_requested_moments(tmp_path):
    s = load_scenario(_write(tmp_path, dispersions=DISP))
    draws = [sample_dispersions(s, seed) for seed in range(200)]
    col = lambda k: np.array([d[k] for d in draws])  # noqa: E731
    assert col("dry_mass").mean() == pytest.approx(80.0, abs=0.5)
    assert col("dry_mass").std() == pytest.approx(0.03 * 80.0, rel=0.2)
    assert col("com_z").mean() == pytest.approx(0.0, abs=0.01)
    assert col("com_z").std() == pytest.approx(0.02, rel=0.2)
    assert col("thrust_scale").mean() == pytest.approx(1.0, abs=0.01)
    assert col("thrust_scale").std() == pytest.approx(0.03, rel=0.2)
    assert col("wind_e").min() >= -4 and col("wind_e").max() <= 4
    assert col("wind_e").std() == pytest.approx(8 / np.sqrt(12), rel=0.2)
    assert np.all(col("wind_u") == 0.0)
    assert col("gyro_bias_x").std() == pytest.approx(0.1, rel=0.25)
    assert col("pos_x").std() == pytest.approx(0.2, rel=0.2)
    assert np.all(col("pos_z") == 0.0)


def test_dispersions_are_a_function_of_the_seed_alone(tmp_path):
    s = load_scenario(_write(tmp_path, dispersions=DISP))
    assert sample_dispersions(s, 5) == sample_dispersions(s, 5)
    assert sample_dispersions(s, 5) != sample_dispersions(s, 6)


def test_scenario_without_dispersions_samples_nothing(tmp_path):
    assert sample_dispersions(load_scenario(_write(tmp_path)), 0) == {}


def test_dispersed_build_changes_the_truth_but_not_what_the_controller_knows(tmp_path, cfg):
    s = load_scenario(_write(tmp_path, dispersions=DISP))
    d = sample_dispersions(s, 4)
    sim = build_simulation(s, seed=4, dispersions=d)
    assert sim.cfg.mass.dry_mass == pytest.approx(d["dry_mass"])
    assert sim.cfg.mass.dry_mass != cfg.mass.dry_mass
    assert sim.cfg.engine.thrust_max == pytest.approx(d["thrust_scale"] * cfg.engine.thrust_max)
    assert sim.cfg.mass.dry_com_z == pytest.approx(cfg.mass.dry_com_z + d["com_z"])
    assert sim.controller.position.params.dry_mass == cfg.mass.dry_mass  # nominal knowledge
    assert sim.x0[0] == pytest.approx(d["pos_x"])
    (bias,) = [f for f in sim.faults if f.type == "gyro_bias"]
    expected = np.radians([d["gyro_bias_x"], d["gyro_bias_y"], d["gyro_bias_z"]])
    assert bias.bias == pytest.approx(expected)
    assert sim.env.wind.mean_i[0] == pytest.approx(d["wind_e"])


def test_run_scenario_applies_dispersions_only_when_asked(tmp_path):
    s = load_scenario(_write(tmp_path, dispersions=DISP))
    nominal = run_scenario(s, seed=1, dispersed=False)
    assert nominal.dispersions == {}
    # build only (no run) keeps the test fast
    assert build_simulation(s, seed=1).cfg.mass.dry_mass == 80.0
