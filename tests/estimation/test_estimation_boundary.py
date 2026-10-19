"""hopper.estimation is flight software: it may import only hopper.types and hopper.math3d."""

import ast
from pathlib import Path

import hopper.estimation

ALLOWED = {"hopper.types", "hopper.math3d"}


def hopper_imports(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found |= {a.name for a in node.names if a.name.split(".")[0] == "hopper"}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative import: resolve it as a hopper.estimation sibling
                found.add(f"hopper.estimation.{base}".rstrip("."))
            elif base.split(".")[0] == "hopper":
                found |= {base} if base != "hopper" else {f"hopper.{a.name}" for a in node.names}
    return found


def own_modules(files):
    return {f"hopper.estimation.{f.stem}" for f in files} | {"hopper.estimation"}


def test_estimation_package_imports_only_types_and_math3d():
    files = sorted(Path(hopper.estimation.__file__).parent.glob("*.py"))
    assert len(files) >= 3
    mine = own_modules(files)
    for f in files:
        bad = hopper_imports(f) - ALLOWED - mine
        assert not bad, f"{f.name} imports {sorted(bad)}"


def test_estimation_package_does_not_import_drake():
    for f in Path(hopper.estimation.__file__).parent.glob("*.py"):
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
            if isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            assert not any(n.split(".")[0] == "pydrake" for n in names), f"{f.name} imports pydrake"


def test_boundary_check_detects_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from hopper.control.mission import MissionController\nimport hopper.sim\n")
    assert hopper_imports(bad) - ALLOWED == {"hopper.control.mission", "hopper.sim"}
