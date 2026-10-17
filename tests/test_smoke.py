import importlib

import pytest

SUBPACKAGES = ["dynamics", "sim", "sensors", "estimation", "control", "analysis"]


def test_package_imports():
    assert importlib.import_module("hopper")


@pytest.mark.parametrize("name", SUBPACKAGES)
def test_subpackage_imports(name):
    assert importlib.import_module(f"hopper.{name}")
