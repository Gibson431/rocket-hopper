from pathlib import Path

import pytest

from hopper.config import VehicleConfig, load_vehicle

DEFAULT = Path(__file__).parent.parent / "configs" / "vehicle_default.yaml"


# Files that fly full closed-loop hops or Monte Carlo batches (about 7 min together).
REGRESSION_FILES = {
    "test_hop_scenario.py",
    "test_closed_loop.py",
    "test_monte_carlo.py",
    "test_scenario.py",
    "test_consistency.py",
}
# Long statistical sensor tests (7-12 s each).
REGRESSION_TESTS = {
    "test_baro_noise_and_bias_walk",
    "test_gyro_bias_random_walk_variance_grows_linearly",
    "test_accel_bias_random_walk_variance_grows_linearly",
}


def pytest_collection_modifyitems(items):
    for item in items:
        if item.path.name in REGRESSION_FILES or item.originalname in REGRESSION_TESTS:
            item.add_marker(pytest.mark.regression)


@pytest.fixture
def cfg() -> VehicleConfig:
    return load_vehicle(DEFAULT)
