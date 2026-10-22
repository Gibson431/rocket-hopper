import importlib

import pytest

SUBPACKAGES = ["dynamics", "sim", "sensors", "analysis"]
FLIGHT_PACKAGES = ["hopper_core", "hopper_control", "hopper_estimation"]


def test_package_imports():
    assert importlib.import_module("hopper")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    assert importlib.import_module(f"hopper.{name}")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_has_real_init(name):
    # A deleted __init__.py still imports as a namespace package (__file__ is None).
    assert importlib.import_module(f"hopper.{name}").__file__ is not None


@pytest.mark.parametrize("name", FLIGHT_PACKAGES)
def test_flight_package_imports(name):
    assert importlib.import_module(name).__file__ is not None
