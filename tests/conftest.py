from pathlib import Path

import pytest

from hopper.config import VehicleConfig, load_vehicle

DEFAULT = Path(__file__).parent.parent / "configs" / "vehicle_default.yaml"


@pytest.fixture
def cfg() -> VehicleConfig:
    return load_vehicle(DEFAULT)
