import ast
from pathlib import Path

import hopper.control

FORBIDDEN = ("hopper.dynamics", "hopper.sim", "hopper.sensors", "hopper.drake", "pydrake")


def _imports(path: Path) -> list[str]:
    names = []
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            names.append(base)
            names += [f"{base}.{a.name}" for a in node.names]
    return names


def test_control_package_does_not_import_simulator_modules():
    files = sorted(Path(hopper.control.__file__).parent.glob("*.py"))
    assert len(files) >= 4  # guards against the glob silently matching nothing
    for f in files:
        for name in _imports(f):
            assert not name.startswith(FORBIDDEN), f"{f.name} imports {name}"


def test_boundary_check_detects_a_violation(tmp_path):
    bad = tmp_path / "bad.py"
    bad.write_text("from hopper.dynamics.propulsion import propulsion\n")
    assert any(n.startswith(FORBIDDEN) for n in _imports(bad))
