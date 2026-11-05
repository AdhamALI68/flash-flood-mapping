#!/usr/bin/env python3
"""
Reconcile the event count. Writes NEW files; overwrites nothing.

dataset_phase3.csv lists 623 events, but 6 of them never resolved to a burst
covering all three dates, so no SAR product exists for them. Three derived files
(lidar_vintage, event_dem_cleanliness, gauge_piles) were computed over all 623,
so every percentage from them uses a denominator 6 too large.

One percent is small, but it is the same class of fault as trial 1's 6,666 vs
5,427: a number that looks right, reconciles with nothing, and ends up in a
table. Fixed by defining one canonical list and recording the exclusions
explicitly rather than by quietly dropping rows.

Writes:
  data/dataset_final.csv      the 617 events that have data - use this from now on
  data/excluded_events.csv    the 6, with the reason
  data/headline_numbers.json  every corrected figure, in one place
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"


def main() -> int:
    ds = pd.read_csv(DATA / "dataset_phase3.csv", low_memory=False)
    st = json.loads((DATA / "phase3_state.json").read_text())
    stored = {int(k) for k, v in st["events"].items() if v.get("stage") == "done"}

    final = ds[ds["EVENT_ID"].isin(stored)].copy()
    excl = ds[~ds["EVENT_ID"].isin(stored)].copy()
    excl["exclusion_reason"] = "no single burst covers the point on all three dates"

    for p, d in ((DATA / "dataset_final.csv", final),
                 (DATA / "excluded_events.csv", excl)):
        if p.exists():
            raise SystemExit(f"{p} already exists - refusing to overwrite")
        d.to_csv(p, index=False)
        print(f"wrote {p.name}  ({len(d)} rows)")

    # ---- recompute every headline figure on the canonical 617 ----------
    vin = pd.read_csv(DATA / "lidar_vintage.csv")
    clean = pd.read_csv(DATA / "event_dem_cleanliness.csv")
    piles = pd.read_csv(DATA / "gauge_piles.csv")

    vin_f = vin[vin["EVENT_ID"].isin(stored)]
    clean_f = clean[clean["EVENT_ID"].isin(stored)]
    piles_f = piles[piles["EVENT_ID"].isin(stored)]

    n = len(final)
    pos = int((piles_f["pile"] == "POSITIVE").sum())
    neg = int((piles_f["pile"] == "NEGATIVE").sum())
    dem_clean = int(clean_f["all_clean"].sum())
    lidar_pre = int(vin_f["has_pre_event_lidar"].sum())

    groups = sum(1 for v in st["groups"].values() if v.get("stage") == "done")

    out = {
        "generated": "reconcile_dataset.py",
        "canonical_events": n,
        "excluded_events": len(excl),
        "independent_observations_groups": groups,
        "basins": int(final["huc8"].nunique()),
        "huc4_regions": int(final["huc4"].nunique()),
        "gauge": {
            "positive": pos, "negative": neg,
            "usable": pos + neg,
            "usable_pct": round(100 * (pos + neg) / n, 1),
            "no_gauge_record": int((piles_f["pile"] == "no_data").sum()),
        },
        "dem_vintage": {
            "fully_clean_events": dem_clean,
            "fully_clean_pct": round(100 * dem_clean / n, 1),
            "any_pre_event_lidar": lidar_pre,
            "any_pre_event_pct": round(100 * lidar_pre / n, 1),
        },
        "note": ("Percentages previously quoted used a denominator of 623. "
                 "These use 617, the events that actually have SAR data."),
    }
    (DATA / "headline_numbers.json").write_text(json.dumps(out, indent=2))

    print("\n" + "=" * 62)
    print("CORRECTED HEADLINE NUMBERS (denominator 617, not 623)")
    print("=" * 62)
    print(f"  events                      {n}")
    print(f"  independent observations    {groups}")
    print(f"  basins / regions            {out['basins']} / {out['huc4_regions']}")
    print()
    print(f"  gauge positive              {pos}")
    print(f"  gauge negative              {neg}")
    print(f"  gauge usable                {pos+neg}  ({out['gauge']['usable_pct']}%)")
    print()
    print(f"  DEM fully clean             {dem_clean}  ({out['dem_vintage']['fully_clean_pct']}%)")
    print(f"  any pre-event lidar         {lidar_pre}  ({out['dem_vintage']['any_pre_event_pct']}%)")
    print()
    print("  wrote data/headline_numbers.json")

    # Did the 6 exclusions change anything material?
    drop_clean = int(clean["all_clean"].sum()) - dem_clean
    drop_pos = int((piles["pile"] == "POSITIVE").sum()) - pos
    drop_neg = int((piles["pile"] == "NEGATIVE").sum()) - neg
    print()
    print("impact of the 6 excluded events:")
    print(f"  fully-clean DEM events lost : {drop_clean}")
    print(f"  gauge positives lost        : {drop_pos}")
    print(f"  gauge negatives lost        : {drop_neg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
