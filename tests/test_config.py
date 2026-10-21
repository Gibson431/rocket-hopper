from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from hopper import config as cfg

DEFAULT = Path(__file__).parent.parent / "configs" / "vehicles" / "vehicle_default.yaml"


def raw() -> dict:
    return yaml.safe_load(DEFAULT.read_text())


def write(tmp_path, data) -> Path:
    p = tmp_path / "v.yaml"
    p.write_text(yaml.safe_dump(data))
    return p


def test_default_yaml_loads_into_a_vehicle_config():
    v = cfg.load_vehicle(DEFAULT)
    assert isinstance(v, cfg.VehicleConfig)
    assert v.mass.dry_mass + v.tank.prop_mass_initial == pytest.approx(100.0)
    assert v.engine.thrust_max == 2500.0
    assert v.engine.thrust_min_frac == 0.3
    assert v.legs.count == 4
    assert v.legs.radius == 0.6
    assert v.sim.dt == 0.001


def test_default_vehicle_geometry_is_self_consistent():
    v = cfg.load_vehicle(DEFAULT)
    # Feet must sit below the engine pivot (z=0) with room for the nozzle.
    assert v.legs.length > 0.3
    vehicle_top = 2.0
    assert v.tank.bottom_z + v.tank.height <= vehicle_top
    assert v.tank.bottom_z - 0.0 > 0.0
    assert 0.0 < v.mass.dry_com_z < vehicle_top
    assert v.tank.radius < v.legs.radius
    assert v.tank.bottom_z > 0.0
    assert 0.0 < v.mass.dry_com_z < v.tank.bottom_z + v.tank.height + 1.0
    assert v.engine.gimbal_max == pytest.approx(8 * 3.141592653589793 / 180, abs=1e-3)


def test_default_yaml_sensors_match_model_defaults():
    got = cfg.load_vehicle(DEFAULT).sensors
    want = cfg.SensorsConfig()
    for name in ("imu", "baro", "gps", "lidar"):
        a, b = getattr(got, name).model_dump(), getattr(want, name).model_dump()
        assert a == pytest.approx(b, rel=1e-6), name


def test_loaded_config_round_trips_through_dump():
    v = cfg.load_vehicle(DEFAULT)
    assert cfg.VehicleConfig.model_validate(v.model_dump()) == v


@pytest.mark.parametrize(
    ("group", "field", "value"),
    [
        ("mass", "dry_mass", -1.0),
        ("mass", "dry_mass", 0.0),
        ("mass", "dry_inertia", [1.0, 2.0]),
        ("mass", "dry_inertia", [1.0, -2.0, 3.0]),
        ("tank", "prop_mass_initial", -1.0),
        ("tank", "radius", 0.0),
        ("engine", "thrust_max", 0.0),
        ("engine", "thrust_min_frac", 0.0),
        ("engine", "thrust_min_frac", 1.5),
        ("engine", "gimbal_max", 0.8),
        ("engine", "gimbal_max", 0.0),
        ("engine", "isp", -5.0),
        ("engine", "throttle_tau", 0.0),
        ("legs", "count", 0),
        ("legs", "friction", -0.1),
        ("aero", "cd_axial", -0.1),
        ("sim", "dt", 0.0),
        ("sim", "controller_rate", 0.0),
    ],
)
def test_invalid_values_raise_validation_error(tmp_path, group, field, value):
    data = raw()
    data[group][field] = value
    with pytest.raises(ValidationError):
        cfg.load_vehicle(write(tmp_path, data))


def test_missing_group_raises_validation_error(tmp_path):
    data = raw()
    del data["engine"]
    with pytest.raises(ValidationError):
        cfg.load_vehicle(write(tmp_path, data))


def test_unknown_field_raises_validation_error(tmp_path):
    data = raw()
    data["engine"]["thrust_maxx"] = 1.0
    with pytest.raises(ValidationError):
        cfg.load_vehicle(write(tmp_path, data))


def test_controller_rate_must_not_exceed_physics_rate(tmp_path):
    data = raw()
    data["sim"]["controller_rate"] = 5000.0
    with pytest.raises(ValidationError):
        cfg.load_vehicle(write(tmp_path, data))


def test_rcs_may_be_omitted_and_means_no_roll_authority(tmp_path):
    data = yaml.safe_load(DEFAULT.read_text())
    del data["rcs"]
    assert cfg.load_vehicle(write(tmp_path, data)).rcs is None


def test_rcs_max_roll_torque_must_be_positive(tmp_path):
    data = yaml.safe_load(DEFAULT.read_text())
    data["rcs"] = {"max_roll_torque": 0.0}
    with pytest.raises(ValidationError):
        cfg.load_vehicle(write(tmp_path, data))
