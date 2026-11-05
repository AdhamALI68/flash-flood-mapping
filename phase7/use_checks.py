#!/usr/bin/env python3
"""
Turn the human check answers into a label set: only floods marked "right" on
the check page are kept. The answers are exported from the page database
(collection checks_v1) to a JSON file first.

    ./.venv/bin/python -u phase7/use_checks.py --answers data/trace_v3/checks_v1.json \
        --from-labels data/trace_v3/labels/candidates_v1 --name checked_v1

Then:  phase7/run_final.sh checked_v1       (photo labels only)
       phase7/run_finetune.sh checked_v1    (headline, then checked labels)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--answers", required=True, help="JSON: {event: verdict} or a list of {id, verdict}")
    ap.add_argument("--from-labels", required=True)
    ap.add_argument("--name", required=True)
    args = ap.parse_args()

    raw = json.loads(Path(args.answers).read_text())
    if isinstance(raw, list):
        raw = {str(d.get("id") or d.get("doc_id") or d.get("event")): (d.get("verdict") or d.get("data", {}).get("verdict")) for d in raw}
    verdicts = {str(k): v for k, v in raw.items()}
    src, dst = ROOT / args.from_labels, ROOT / "data/trace_v3/labels" / args.name
    dst.mkdir(parents=True, exist_ok=True)
    for f in dst.glob("*.tif"):
        f.unlink()

    right = [e for e, v in verdicts.items() if v == "right" and (src / f"{e}.tif").exists()]
    for e in right:
        shutil.copy(src / f"{e}.tif", dst / f"{e}.tif")
    tally = pd.Series(list(verdicts.values())).value_counts().to_dict()
    idx = pd.read_csv(ROOT / "data/tiles_v4/index.csv", usecols=["event"])
    summary = {"rules": args.name, "from": args.from_labels, "answers": tally, "kept": len(right),
               "tiles_of_kept_floods_before_masking": int(idx.event.astype(str).isin(right).sum())}
    (dst / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
