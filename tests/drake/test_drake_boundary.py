"""types and math3d must stay free of Drake (control/estimation have their own tests)."""

import ast
from pathlib import Path

import hopper


def _imports_pydrake(path: Path) -> bool:
    for node in ast.walk(ast.parse(path.read_text())):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
        if isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        if any(n.split(".")[0] == "pydrake" or n.startswith("hopper.drake") for n in names):
            return True
    return False


def test_types_and_math3d_do_not_import_drake():
    root = Path(hopper.__file__).parent
    for name in ("types.py", "math3d.py"):
        assert not _imports_pydrake(root / name), name


def test_detector_flags_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from pydrake.all import Simulator\n")
    assert _imports_pydrake(bad)
