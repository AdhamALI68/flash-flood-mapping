#!/usr/bin/env python3
"""
The flood-trace pipeline plan, in order, with each step's status. STATUS.md
reads it every 20 s.

    ./.venv/bin/python phase7/plan.py init
    ./.venv/bin/python phase7/plan.py set s40 running "re-run with rule S5"
    ./.venv/bin/python phase7/plan.py show
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PLAN = ROOT / "data/trace_v3/plan.json"

STEPS = [
    ("s20", "Mark 20 new floods, look at the mistakes, add rules"),
    ("all", "Download and mark every flood that has training tiles, save everything"),
    ("rules", "Find and apply the rules on all saved floods, check samples by eye"),
    ("tiles", "Build training tiles from the clean floods"),
    ("train", "Train the model on the new labels"),
    ("score", "Score recall and precision on the held-out test regions"),
]


def load():
    return json.loads(PLAN.read_text())


def main() -> int:
    cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
    if cmd == "init":
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(json.dumps([{"id": i, "what": w, "status": "queued", "note": "", "at": ""}
                                    for i, w in STEPS], indent=1))
    elif cmd == "set":
        plan = load()
        for s in plan:
            if s["id"] == sys.argv[2]:
                s["status"] = sys.argv[3]
                s["note"] = sys.argv[4] if len(sys.argv) > 4 else s["note"]
                s["at"] = time.strftime("%Y-%m-%d %H:%M")
        PLAN.write_text(json.dumps(plan, indent=1))
    for s in load():
        print(f"{s['id']:6s} {s['status']:22s} {s['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
