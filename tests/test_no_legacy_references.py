"""The hand-rolled simulator is gone (S22): its modules do not exist and nothing refers to them."""

import re
from pathlib import Path

import pytest

REPO = Path(__file__).parent.parent
SRC = REPO / "src" / "hopper"
DELETED = [
    "dynamics/rigid_body.py",
    "dynamics/propulsion.py",
    "dynamics/aero.py",
    "dynamics/contact.py",
    "dynamics/vehicle.py",
    "sim/integrator.py",
    "sim/engine.py",
]
# Built from parts so this file does not match itself.
STALE = re.compile(
    "|".join(
        [
            r"hopper\.sim\.(?:engine|integrator)",
            r"hopper\.dynamics\.(?:rigid_body|propulsion|aero|contact|vehicle)",
            "rk4" + "_step",
            "vehicle" + "_derivative",
            "contact" + "_loads",
            "aero" + "_loads",
            "rigid_body" + "_derivative",
            "fixed-step " + "RK4",
            "to_" + "legacy_state",
            "set_" + "legacy_state",
        ]
    )
)
SCANNED = [REPO / d for d in ("src", "tests", "scripts", ".github")] + [
    REPO / "README.md",
    REPO / "docs" / "CONVENTIONS.md",
    REPO / "docs" / "PLAN.md",
    REPO / "docs" / "AGENT_START.md",
]


@pytest.mark.parametrize("rel", DELETED)
def test_legacy_module_is_deleted(rel):
    assert not (SRC / rel).exists()


def test_nothing_refers_to_the_deleted_modules():
    hits = []
    for root in SCANNED:
        files = [root] if root.is_file() else [p for p in root.rglob("*") if p.is_file()]
        for f in files:
            if f.suffix not in {".py", ".md", ".yml", ".yaml", ".toml"} or f == Path(__file__):
                continue
            hits += [f"{f.relative_to(REPO)}: {m.group()}" for m in STALE.finditer(f.read_text())]
    assert not hits, hits
