#!/usr/bin/env python3
"""
Freeze the Phase 3 input set.

Everything Phase 1-2 established, applied as one explicit filter chain so the
dataset definition lives in one place and every exclusion is counted. Phase 3
reads this file and nothing else.

Filters, in order:

  1. basin is arid          PRISM mean annual precipitation < 250 mm
  2. basin has lidar        >= 90% 3DEP 1 m coverage
  3. basin has volume       >= 20 SAR-usable events
  4. event is SAR-usable    2 pre + 1 post on one relative orbit
  5. intervals match        |baseline span - co-event span| <= 3 days
  6. no DST ambiguity       no scene within 1 h of the event time

Filter 5 is the one the plan did not anticipate. Coherence decays with elapsed
time on its own, so if the baseline pair spans 24 days and the co-event pair 6,
the difference between them measures the calendar, not the flood. Median
mismatch is 0.1 days, so this costs little.

Filter 6 exists because NOAA records local standard time while California
observes daylight saving. A one-hour error is irrelevant against multi-day gaps
except when a scene sits within an hour of the event, where it could flip a
scene from pre to post.

Output: data/dataset_phase3.csv + data/dataset_manifest.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

ARID_MM = 250
MIN_LIDAR = 0.90
MIN_EVENTS = 20
MAX_INTERVAL_MISMATCH_D = 3.0
DST_GUARD_D = 1 / 24


def main() -> int:
    ar = pd.read_csv(ROOT / "data/basin_aridity.csv")
    av = pd.read_csv(ROOT / "data/sar_availability.csv", low_memory=False)
    ev = pd.read_csv(ROOT / "data/events.csv", low_memory=False)
    for d in (ar, av, ev):
        d["huc8"] = d["huc8"].astype(str).str.zfill(8)

    steps = []

    def note(label, df):
        steps.append({"step": label, "events": int(len(df)),
                      "basins": int(df["huc8"].nunique()) if "huc8" in df else None})
        print(f"{label:<42} {len(df):>6} events  "
              f"{df['huc8'].nunique() if 'huc8' in df else '-':>3} basins")

    basins = ar[(ar["ppt_mm_mean"] < ARID_MM)
                & (ar["lidar_coverage_frac"] >= MIN_LIDAR)
                & (ar["sar_usable_events"] >= MIN_EVENTS)]
    print(f"qualifying basins: {len(basins)}  "
          f"({basins['huc4'].nunique()} HUC4 regions)\n")

    df = av.copy()
    note("all events checked for SAR", df)

    df = df[df["usable"] == True]
    note("SAR-usable (2 pre + 1 post, one orbit)", df)

    df = df[df["huc8"].isin(basins["huc8"])]
    note("in arid basins with lidar", df)

    for c in ("pre_gap_days", "post_gap_days", "baseline_gap_days"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["coevent_span_days"] = df["pre_gap_days"] + df["post_gap_days"]
    df["interval_mismatch_days"] = (df["baseline_gap_days"] - df["coevent_span_days"]).abs()

    df = df[df["interval_mismatch_days"] <= MAX_INTERVAL_MISMATCH_D]
    note(f"baseline/co-event intervals within {MAX_INTERVAL_MISMATCH_D:g} d", df)

    df = df[(df["pre_gap_days"] >= DST_GUARD_D) & (df["post_gap_days"] >= DST_GUARD_D)]
    note("no scene within 1 h of the event (DST guard)", df)

    keep = ["EVENT_ID", "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
            "BEGIN_UTC", "END_UTC", "basin_area_km2", "DAMAGE_PROPERTY",
            "DEATHS_DIRECT", "CZ_NAME", "STATE"]
    out = df.merge(ev[[c for c in keep if c in ev.columns]].drop_duplicates("EVENT_ID"),
                   on="EVENT_ID", how="left", suffixes=("", "_ev"))
    out = out.merge(ar[["huc8", "ppt_mm_mean", "aridity_band", "lidar_coverage_frac"]],
                    on="huc8", how="left")
    out.to_csv(ROOT / "data/dataset_phase3.csv", index=False)

    manifest = {
        "generated_from": ["data/basin_aridity.csv", "data/sar_availability.csv",
                           "data/events.csv"],
        "filters": {
            "arid_mm": ARID_MM, "min_lidar_frac": MIN_LIDAR,
            "min_events_per_basin": MIN_EVENTS,
            "max_interval_mismatch_days": MAX_INTERVAL_MISMATCH_D,
            "dst_guard_days": DST_GUARD_D,
        },
        "steps": steps,
        "final": {
            "events": int(len(out)),
            "basins": int(out["huc8"].nunique()),
            "huc4_regions": int(out["huc4"].nunique()),
            "date_range": [str(pd.to_datetime(out["BEGIN_UTC"]).min()),
                           str(pd.to_datetime(out["BEGIN_UTC"]).max())],
            "hyp3_jobs_needed": int(len(out) * 2),
            "hyp3_credits_needed": int(len(out) * 2),
        },
    }
    (ROOT / "data/dataset_manifest.json").write_text(json.dumps(manifest, indent=2))

    print()
    print("=" * 62)
    print(f"FINAL: {len(out)} events | {out['huc8'].nunique()} basins | "
          f"{out['huc4'].nunique()} HUC4 regions")
    print(f"HyP3 jobs needed: {2*len(out)} (2 per event) = {2*len(out)} credits")
    print("=" * 62)
    print()
    print("events per basin:")
    per = out.groupby(["huc8", "basin_name"]).size().sort_values(ascending=False)
    print(per.to_string())
    print()
    print("events per HUC4 region (the cross-region split unit):")
    print(out.groupby("huc4").size().sort_values(ascending=False).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
