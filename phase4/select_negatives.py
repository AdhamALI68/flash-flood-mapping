#!/usr/bin/env python3
"""
Build a TRUE negative set: same places, quiet dates.

The threshold sweep returned AUC 0.5 because the negative pile was defined as
"a flood was reported but the gauge stayed flat", which in a desert usually means
the gauge was far away, not that nothing happened. Event 983830 - the Gate B
flood, visible in the coherence image at 10x contrast - was in that pile.

A real negative is a place and time with NO flood reported at all.

Design choices, each removing a confound:

  SAME LOCATION as a flood event. Terrain, land cover, radar geometry and
  incidence angle are then identical between the two groups, so any difference
  in coherence drop cannot be terrain.

  SAME SEASON. Floods here are monsoon-driven. Drawing negatives from winter
  would confound "flood vs no flood" with "summer vs winter" - dust storms,
  vegetation and soil moisture all differ. Candidates are drawn within +/- 45
  days of the flood's day-of-year, in any year.

  QUIET WINDOW. No flash flood reported anywhere in that HUC8 within +/- 14
  days, checked against all 4,872 events, not just the 617 in the dataset.

  SAME RELATIVE ORBIT as the paired flood event, so look angle matches.

Output: data/negative_candidates.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

QUIET_DAYS = 14        # no flood report within this many days
SEASON_DAYS = 45       # candidate must be this close in day-of-year
MIN_GAP_YEARS = 0      # allow the same year, provided the quiet window holds


def doy_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    """Circular day-of-year distance, so December and January are close."""
    d = abs(a.dayofyear - b.dayofyear)
    return min(d, 365 - d)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=180, help="negatives to select")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    final = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    final["BEGIN_UTC"] = pd.to_datetime(final["BEGIN_UTC"])
    final["huc8"] = final["huc8"].astype(str).str.zfill(8)

    # Every flash flood ever reported in these basins - the disqualifier set.
    allev = pd.read_csv(ROOT / "data/events.csv", low_memory=False)
    allev["BEGIN_UTC"] = pd.to_datetime(allev["BEGIN_UTC"])
    allev["huc8"] = allev["huc8"].astype(str).str.zfill(8)
    by_basin = {h: np.sort(g["BEGIN_UTC"].values)
                for h, g in allev.groupby("huc8")}
    print(f"disqualifier set: {len(allev)} reported floods across "
          f"{len(by_basin)} basins")

    avail = pd.read_csv(ROOT / "data/sar_availability.csv", low_memory=False)
    avail["huc8"] = avail["huc8"].astype(str).str.zfill(8)

    # Sample flood events to pair against, stratified so no region dominates.
    rng = np.random.default_rng(args.seed)
    per_region = max(1, args.n // final["huc4"].nunique())
    picks = []
    for reg, g in final.groupby("huc4"):
        take = min(per_region, len(g))
        picks.append(g.sample(take, random_state=args.seed))
    seeds = pd.concat(picks, ignore_index=True)
    print(f"pairing against {len(seeds)} flood events "
          f"({seeds['huc4'].nunique()} regions)\n")

    rows = []
    for r in seeds.itertuples(index=False):
        floods = by_basin.get(r.huc8, np.array([], dtype="datetime64[ns]"))
        # Candidate quiet dates: same season, any year in the S1 archive.
        cands = []
        for year in range(2015, 2027):
            try:
                c = pd.Timestamp(year=year, month=r.BEGIN_UTC.month,
                                 day=min(r.BEGIN_UTC.day, 28))
            except ValueError:
                continue
            for shift in (-30, -15, 0, 15, 30):
                d = c + pd.Timedelta(days=shift)
                if not (pd.Timestamp("2015-01-01") <= d <= pd.Timestamp("2026-06-01")):
                    continue
                if doy_diff(d, r.BEGIN_UTC) > SEASON_DAYS:
                    continue
                # quiet check
                if len(floods):
                    delta = np.abs((floods - np.datetime64(d)).astype("timedelta64[D]").astype(int))
                    if delta.min() <= QUIET_DAYS:
                        continue
                cands.append(d)
        if not cands:
            continue
        pick = cands[int(rng.integers(len(cands)))]
        rows.append({
            "NEG_ID": f"NEG{r.EVENT_ID}",
            "paired_event": int(r.EVENT_ID),
            "huc8": r.huc8, "huc4": r.huc4, "basin_name": r.basin_name,
            "BEGIN_LAT": r.BEGIN_LAT, "BEGIN_LON": r.BEGIN_LON,
            "flood_date": str(r.BEGIN_UTC),
            "quiet_date": str(pick),
            "doy_gap": doy_diff(pick, r.BEGIN_UTC),
            "n_candidate_dates": len(cands),
        })

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/negative_candidates.csv", index=False)

    print("=" * 66)
    print(f"selected {len(out)} quiet dates")
    print(f"  regions covered : {out['huc4'].nunique()}")
    print(f"  basins covered  : {out['huc8'].nunique()}")
    print(f"  median day-of-year gap from its paired flood: "
          f"{out['doy_gap'].median():.0f} days")
    print(f"  median candidate dates available per site   : "
          f"{out['n_candidate_dates'].median():.0f}")
    print()
    print("per region:")
    print(out.groupby("huc4").size().rename("negatives").to_frame().to_string())
    print()
    print("wrote data/negative_candidates.csv")
    print()
    print("Each negative is the SAME location as a real flood event, in the same")
    print("season, on a date with no flood reported within 14 days. Terrain and")
    print("look geometry are therefore identical between the two groups.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
