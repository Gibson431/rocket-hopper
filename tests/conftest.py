from pathlib import Path

import pytest

from hopper.config import VehicleConfig, load_vehicle

DEFAULT = Path(__file__).parent.parent / "configs" / "vehicles" / "vehicle_default.yaml"


# A regression test is any test that runs a simulation end to end (a scenario, a closed loop, a
# full vehicle/filter trajectory, a statistical batch). A test that only brings up a plant or
# subsystem and steps it once or a few times to see how it behaves is a unit test, even if it
# builds Drake systems. The default run is the unit set; `uv run pytest -m ""` runs everything.
REGRESSION_FILES = {
    "test_hop_scenario.py",
    "test_closed_loop.py",
    "test_monte_carlo.py",
    "test_scenario.py",
    "test_consistency.py",
    "test_engine.py",
    "test_faults.py",
}
REGRESSION_TESTS = {
    # closed-loop controller runs
    "test_integrator_does_not_wind_up_while_saturated",
    "test_integrator_removes_a_constant_acceleration_offset",
    "test_altitude_step_of_5m_settles_within_1_percent_with_small_overshoot",
    "test_horizontal_step_respects_tilt_clamp_and_keeps_altitude",
    "test_hover_with_20ms_delay_holds_position",
    # vehicle/contact trajectories
    "test_drop_settles_with_foot_forces_supporting_weight_and_no_lateral_drift",
    "test_resting_normal_force_is_steady_at_default_dt",
    "test_sideways_push_while_standing_decelerates_and_stops",
    "test_spinning_in_place_on_the_ground_decays_without_a_numerical_limit_cycle",
    "test_tilting_past_footprint_angle_tips_over_but_a_small_tilt_recovers",
    "test_landing_from_a_small_drop_never_gains_mechanical_energy",
    "test_thrust_above_weight_lifts_off_and_contact_goes_to_zero",
    "test_rocket_equation_delta_v_with_no_gravity",
    "test_terminal_velocity_matches_drag_balance",
    # filter runs against simulated trajectories
    "test_drop_from_half_a_metre_settles_upright_without_bouncing",
    "test_hard_five_metre_per_second_landing_stays_finite_without_rebound",
    "test_a_gentle_drop_ends_the_run_as_landed",
    "test_a_hard_landing_ends_the_run_as_crashed_immediately",
    "test_buffered_gps_latency_compensation_beats_naive_application",
    "test_returns_none_until_aligned_then_levels_the_attitude_and_zeroes_velocity",
    "test_constant_gyro_bias_is_estimated_within_10_percent_after_30_s_of_motion",
    "test_estimate_tracks_the_manoeuvre_with_nominal_noise",
    "test_covariance_stays_symmetric_and_positive_definite_through_a_long_run",
    "test_gps_outage_inflates_the_covariance_and_the_estimate_stays_bounded",
    "test_persistent_gps_disagreement_is_accepted_after_the_reject_timeout",
    "test_gps_spike_of_50_m_is_rejected_and_the_next_good_fix_accepted",
    # long statistical sensor batches
    "test_baro_noise_and_bias_walk",
    "test_gyro_bias_random_walk_variance_grows_linearly",
    "test_accel_bias_random_walk_variance_grows_linearly",
    "test_lidar_noise_has_absolute_and_relative_parts",
    "test_gps_noise_statistics",
    "test_gps_dropout_probability",
}
REGRESSION_PREFIXES = ("test_closed_loop_",)


def pytest_collection_modifyitems(items):
    for item in items:
        name = item.originalname or item.name
        if (
            item.path.name in REGRESSION_FILES
            or name in REGRESSION_TESTS
            or name.startswith(REGRESSION_PREFIXES)
        ):
            item.add_marker(pytest.mark.regression)


@pytest.fixture
def cfg() -> VehicleConfig:
    return load_vehicle(DEFAULT)
