"""Flight software packages (hopper_core, hopper_control, hopper_estimation) import only the
standard library, numpy/scipy as listed, hopper_core and themselves: no pydrake, no simulator
module, and the controller and estimator never import each other. Allowlist form, so a new
simulator module cannot slip in unnoticed."""

import ast
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
PACKAGES = REPO / "packages"
# (package dir, import name, allowed third-party roots, allowed hopper packages)
UNITS = [
    ("hopper-core", "hopper_core", {"numpy"}, {"hopper_core"}),
    ("hopper-control", "hopper_control", {"numpy"}, {"hopper_core", "hopper_control"}),
    (
        "hopper-estimation",
        "hopper_estimation",
        {"numpy", "scipy"},
        {"hopper_core", "hopper_estimation"},
    ),
]


def imported_modules(path: Path) -> set[str]:
    """Absolute dotted names of everything `path` imports (`from a import b` gives a and a.b)."""
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, f"{path.name}: use absolute imports"
            base = node.module or ""
            found.add(base)
            found |= {f"{base}.{a.name}" for a in node.names}
    return found


def violations(path: Path, third_party: set[str], own: set[str]) -> set[str]:
    bad = set()
    for name in imported_modules(path):
        root = name.split(".")[0]
        if root in sys.stdlib_module_names or root in third_party or root in own:
            continue
        bad.add(name)
    return bad


def sources(package_dir: str, name: str) -> list[Path]:
    return sorted((PACKAGES / package_dir / "src" / name).rglob("*.py"))


@pytest.mark.parametrize(("package_dir", "name", "third_party", "own"), UNITS)
def test_flight_software_imports_only_the_allowlist(package_dir, name, third_party, own):
    files = sources(package_dir, name)
    assert files
    for f in files:
        bad = violations(f, third_party, own)
        assert not bad, f"{f.relative_to(REPO)} imports {sorted(bad)}"


@pytest.mark.parametrize("package_dir", [u[0] for u in UNITS])
def test_declared_dependencies_exclude_simulator_and_siblings(package_dir):
    text = (PACKAGES / package_dir / "pyproject.toml").read_text()
    banned = {"drake", "pydrake", "pandas", "hopper-control", "hopper-estimation"}
    banned -= {package_dir}
    deps = text.split("dependencies = [")[1].split("]")[0]
    assert not [b for b in banned if f'"{b}' in deps]


def test_boundary_check_detects_simulator_and_drake_imports(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from hopper.dynamics.mass_model import mass_properties\n"
        "import hopper.sim\n"
        "from pydrake.all import Simulator\n"
        "from hopper import drake\n"
        "import pandas\n"
    )
    assert violations(bad, {"numpy"}, {"hopper_core", "hopper_control"}) == {
        "hopper",
        "hopper.dynamics.mass_model",
        "hopper.dynamics.mass_model.mass_properties",
        "hopper.sim",
        "pydrake.all",
        "pydrake.all.Simulator",
        "hopper.drake",
        "pandas",
    }
