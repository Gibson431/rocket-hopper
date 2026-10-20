"""Flight software (control, estimation) and the shared `types` / `math3d` modules import only the
standard library, numpy, scipy and each other: no pydrake and no simulator module. Allowlist form,
so a new simulator module cannot slip in unnoticed."""

import ast
import sys
from pathlib import Path

import pytest

import hopper

ROOT = Path(hopper.__file__).parent
THIRD_PARTY = {"numpy", "scipy"}
SHARED = {"hopper.types", "hopper.math3d"}
# (path under src/hopper, the package it may also import from)
UNITS = [
    ("control", "hopper.control"),
    ("estimation", "hopper.estimation"),
    ("types.py", None),
    ("math3d.py", None),
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


def violations(path: Path, own: str | None) -> set[str]:
    allowed_hopper = SHARED | ({own} if own else set())
    bad = set()
    for name in imported_modules(path):
        root = name.split(".")[0]
        if root in sys.stdlib_module_names or root in THIRD_PARTY:
            continue
        if root == "hopper" and (
            name == "hopper" or any(name == a or name.startswith(a + ".") for a in allowed_hopper)
        ):
            continue
        bad.add(name)
    return bad


def sources(unit: str) -> list[Path]:
    path = ROOT / unit
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


@pytest.mark.parametrize(("unit", "own"), UNITS)
def test_flight_software_imports_only_the_allowlist(unit, own):
    files = sources(unit)
    assert files and all(f.exists() for f in files)
    for f in files:
        assert not violations(f, own), f"{f.relative_to(ROOT)} imports {sorted(violations(f, own))}"


def test_boundary_check_detects_simulator_and_drake_imports(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text(
        "from hopper.dynamics.mass_model import mass_properties\n"
        "import hopper.sim\n"
        "from pydrake.all import Simulator\n"
        "from hopper import drake\n"
        "import pandas\n"
    )
    assert violations(bad, "hopper.control") == {
        "hopper.dynamics.mass_model",
        "hopper.dynamics.mass_model.mass_properties",
        "hopper.sim",
        "pydrake.all",
        "pydrake.all.Simulator",
        "hopper.drake",
        "pandas",
    }
