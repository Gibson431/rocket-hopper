#!/usr/bin/env python3
"""Print the next step to work on, based on docs/PROGRESS.md. Stdlib only."""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PROGRESS = ROOT / "docs" / "PROGRESS.md"
STEPS = ROOT / "docs" / "steps"


def parse_rows(text):
    rows = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 6 and re.fullmatch(r"S\d+", cells[0]):
            deps = [] if cells[3] in ("-", "") else [d.strip() for d in cells[3].split(",")]
            rows.append(
                {"id": cells[0], "title": cells[1], "status": cells[2], "deps": deps, "notes": cells[5]}
            )
    return rows


def step_file(step_id):
    matches = sorted(STEPS.glob(f"{step_id}_*.md"))
    return matches[0].relative_to(ROOT) if matches else None


def main():
    rows = parse_rows(PROGRESS.read_text())
    if not rows:
        sys.exit("No step rows found in docs/PROGRESS.md")
    status = {r["id"]: r["status"] for r in rows}

    for r in rows:
        if r["status"] == "in_progress":
            print(f"IN PROGRESS: {r['id']} {r['title']}  ->  {step_file(r['id'])}")
            if r["notes"]:
                print(f"  notes: {r['notes']}")
    for r in rows:
        if r["status"] == "blocked":
            print(f"BLOCKED: {r['id']} {r['title']}  notes: {r['notes']}")

    ready = [
        r for r in rows
        if r["status"] == "todo" and all(status.get(d) == "done" for d in r["deps"])
    ]
    if ready:
        nxt = ready[0]
        print(f"NEXT: {nxt['id']} {nxt['title']}  ->  {step_file(nxt['id'])}")
        others = ", ".join(r["id"] for r in ready[1:])
        if others:
            print(f"Also ready (can run in parallel): {others}")
    elif all(r["status"] == "done" for r in rows):
        print("All steps are done.")
    else:
        print("No step is ready: remaining steps are in progress, blocked, or waiting on dependencies.")


if __name__ == "__main__":
    main()
