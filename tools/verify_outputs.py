#!/usr/bin/env python3
"""
Spot-check saved Phase 3 output for real content, and look for failure patterns.

Read-only. Safe to run while the pipeline is downloading.

Check 1 - are the rasters real?
    A raster can be present, correctly sized, and still useless: all zeros, all
    NaN, or a constant fill. Coherence must sit in (0,1] with genuine spread;
    amplitude must be positive with a wide dynamic range. Checks the actual
    pixel statistics rather than trusting file size.

Check 2 - do failures cluster?
    Scattered failures are random infrastructure noise. Failures concentrated in
    one basin or region mean something systematic - a track with bad geometry, a
    basin whose bursts do not overlap - and that would bias the dataset by
    quietly removing particular terrain.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
GROUPS = ROOT / "data/phase3/groups"


def stats(path: Path) -> dict:
    with rasterio.open(path) as s:
        a = s.read(1).astype("float64")
        nod = s.nodata
    total = a.size
    if nod is not None:
        a = np.where(a == nod, np.nan, a)
    a = np.where(np.isfinite(a), a, np.nan)
    valid = a[np.isfinite(a)]
    nz = valid[valid != 0]
    return {
        "px": total,
        "valid_pct": 100 * len(valid) / total if total else 0,
        "nonzero_pct": 100 * len(nz) / total if total else 0,
        "min": float(np.nanmin(valid)) if len(valid) else None,
        "max": float(np.nanmax(valid)) if len(valid) else None,
        "mean": float(np.nanmean(valid)) if len(valid) else None,
        "std": float(np.nanstd(valid)) if len(valid) else None,
        "uniq": int(len(np.unique(np.round(valid[:200000], 4)))) if len(valid) else 0,
    }


def check_rasters(n: int = 3) -> int:
    dirs = sorted([d for d in GROUPS.iterdir() if d.is_dir()])[:n] if GROUPS.exists() else []
    if not dirs:
        print("no completed groups on disk yet")
        return 1
    problems = 0
    for d in dirs:
        print(f"\n=== {d.name} ===")
        for tag in ("baseline", "coevent"):
            sub = d / tag
            if not sub.exists():
                print(f"  {tag}: MISSING"); problems += 1; continue
            for kind in ("corr", "amp"):
                hits = sorted(sub.glob(f"*_{kind}.tif"))
                if not hits:
                    print(f"  {tag}/{kind}: MISSING"); problems += 1; continue
                st = stats(hits[0])
                flags = []
                if st["nonzero_pct"] < 1:
                    flags.append("ALL ZERO/EMPTY")
                if st["uniq"] < 10:
                    flags.append("CONSTANT FILL")
                if kind == "corr":
                    if st["max"] is None or st["max"] > 1.001 or st["min"] < -0.001:
                        flags.append("OUT OF RANGE for coherence")
                    if st["std"] is not None and st["std"] < 0.01:
                        flags.append("NO SPREAD")
                if kind == "amp" and st["max"] is not None and st["max"] <= 0:
                    flags.append("NON-POSITIVE AMPLITUDE")
                problems += len(flags)
                verdict = "  <-- " + ", ".join(flags) if flags else "ok"
                print(f"  {tag}/{kind:<4}: {st['px']:>10,} px | valid {st['valid_pct']:5.1f}% | "
                      f"nonzero {st['nonzero_pct']:5.1f}% | "
                      f"range {st['min']:.3f}-{st['max']:.3f} | "
                      f"mean {st['mean']:.3f} sd {st['std']:.3f} | "
                      f"{st['uniq']} distinct  {verdict}")
    return problems


def check_failures() -> None:
    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    groups = st.get("groups", {})
    events = st.get("events", {})

    gfail = {k: v for k, v in groups.items() if str(v.get("stage", "")).endswith("failed")}
    efail = {k: v for k, v in events.items() if v.get("stage") == "failed"}

    print("\n" + "=" * 70)
    print("FAILURE PATTERN")
    print("=" * 70)
    print(f"groups: {dict(Counter(v.get('stage') for v in groups.values()))}")
    print(f"failed groups: {len(gfail)}   failed events: {len(efail)}")

    if not gfail and not efail:
        print("\nNo failures. Nothing to cluster.")
        # Still show what has completed, per region, so under-representation
        # would be visible even before any outright failure occurs.
        recs = [json.loads(l) for l in (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines() if l.strip()]
        by_event = {str(r["EVENT_ID"]): r for r in recs}
        done_regions = Counter()
        for eid, v in events.items():
            if v.get("stage") == "done" and eid in by_event:
                done_regions[by_event[eid]["huc4"]] += 1
        total_regions = Counter(r["huc4"] for r in recs)
        print("\nprogress by region so far (checks for silent under-coverage):")
        rows = [{"huc4": k, "done": done_regions.get(k, 0), "total": v,
                 "pct": round(100 * done_regions.get(k, 0) / v, 1)}
                for k, v in sorted(total_regions.items())]
        print(pd.DataFrame(rows).to_string(index=False))
        return

    recs = [json.loads(l) for l in (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines() if l.strip()]
    by_event = {str(r["EVENT_ID"]): r for r in recs}
    reg = Counter(); bas = Counter(); reasons = Counter()
    for eid, v in efail.items():
        r = by_event.get(eid)
        if r:
            reg[r["huc4"]] += 1
            bas[r["basin_name"]] += 1
        reasons[str(v.get("error", ""))[:60]] += 1
    for k, v in gfail.items():
        reasons[str(v.get("error", ""))[:60]] += 1

    if reg:
        print("\nfailed events by region:")
        for k, v in reg.most_common():
            print(f"  {k}: {v}")
        print("\nfailed events by basin:")
        for k, v in bas.most_common():
            print(f"  {k}: {v}")
    print("\nreasons:")
    for k, v in reasons.most_common(10):
        print(f"  {v:>3}x  {k}")


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print(f"Spot-checking {n} completed groups (read-only)")
    problems = check_rasters(n)
    check_failures()
    print("\n" + "=" * 70)
    print(f"VERDICT: {'PASS - rasters contain real data' if problems == 0 else f'{problems} PROBLEM(S) FOUND'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
