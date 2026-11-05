#!/usr/bin/env python3
"""
NEXT_STEPS 3b - does pre-event lidar exist?

The blocking question for the resolution ablation. A lidar survey flown AFTER a
flood contains that flood's own scour and deposition, so using it as a model
input leaks the answer through the terrain channel. For each event this asks:
is there a 1 m survey covering its neighbourhood that predates it?

Per event, over tiles intersecting a 5 km buffer (the Gate B radius):
  - latest survey that PREDATES the event   -> the clean choice
  - earliest survey that POSTDATES it       -> the fallback, flagged
  - signed gap in years (negative = clean)

Output: data/lidar_vintage.csv
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import geopandas as gpd
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EQUAL_AREA = "EPSG:5070"
BUFFER_M = 5000


def survey_year(project: str) -> tuple[int | None, str]:
    """Year for a 3DEP project, plus how it was obtained.

    Two naming conventions coexist, and matching only the first drops two
    thirds of the index:

      explicit  AZ_PimaCounty_2021_B21   -> 2021
      code      CA_FEMAR9Southeast_D24   -> D24 -> 2024
                NV_ClarkCounty_B22       -> B22 -> 2022

    The letter-code form is a delivery/fiscal year, which is at or AFTER
    acquisition. Using it therefore makes surveys look newer than they were, so
    an event can be marked "post-event lidar" when a pre-event survey actually
    exists. That errs toward excluding events rather than leaking them, which is
    the safe direction - but the method is returned so the two can be separated.
    """
    p = str(project)
    m = re.search(r"(20\d{2})", p)
    if m:
        return int(m.group(1)), "explicit"
    m = re.search(r"_[A-Z](\d{2})(?:_|$)", p) or re.search(r"[A-Z](\d{2})$", p)
    if m:
        return 2000 + int(m.group(1)), "code"
    return None, "none"


def main() -> int:
    files = glob.glob(str(ROOT / "data/raw/tnm_tile_cache/*.csv"))
    tiles = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    parsed = tiles["project"].apply(survey_year)
    tiles["survey_year"] = [x[0] for x in parsed]
    tiles["year_method"] = [x[1] for x in parsed]
    n_before = len(tiles)
    tiles = tiles[tiles["survey_year"].notna() & tiles["geometry_wkt"].notna()].copy()
    tiles["survey_year"] = tiles["survey_year"].astype(int)
    print(f"tile records: {n_before:,} -> {len(tiles):,} with a year "
          f"({(tiles['year_method'] == 'code').sum():,} from delivery code)")
    g = gpd.GeoDataFrame(tiles, geometry=gpd.GeoSeries.from_wkt(tiles["geometry_wkt"]),
                         crs="EPSG:4326").to_crs(EQUAL_AREA)
    print(f"tile records with a parseable survey year: {len(g):,}")
    print(f"survey years present: {sorted(g['survey_year'].unique())}")

    ev = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    ev["flood_year"] = pd.to_datetime(ev["BEGIN_UTC"]).dt.year
    ep = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.BEGIN_LON, ev.BEGIN_LAT),
                          crs="EPSG:4326").to_crs(EQUAL_AREA)
    ep["geometry"] = ep.buffer(BUFFER_M)

    hit = gpd.sjoin(ep[["EVENT_ID", "huc8", "flood_year", "geometry"]],
                    g[["survey_year", "geometry"]], how="left", predicate="intersects")

    rows = []
    for eid, grp in hit.groupby("EVENT_ID"):
        fy = int(grp["flood_year"].iloc[0])
        yrs = grp["survey_year"].dropna().astype(int)
        pre = yrs[yrs < fy]
        post = yrs[yrs >= fy]
        rows.append({
            "EVENT_ID": eid,
            "huc8": str(grp["huc8"].iloc[0]).zfill(8),
            "flood_year": fy,
            "n_surveys": len(yrs),
            "best_pre_year": int(pre.max()) if len(pre) else None,
            "earliest_post_year": int(post.min()) if len(post) else None,
            "has_pre_event_lidar": bool(len(pre)),
            "gap_years": int(pre.max() - fy) if len(pre) else
                         (int(post.min() - fy) if len(post) else None),
        })

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/lidar_vintage.csv", index=False)

    n = len(out)
    clean = out["has_pre_event_lidar"].sum()
    nocov = (out["n_surveys"] == 0).sum()
    print()
    print("=" * 64)
    print(f"events                        : {n}")
    print(f"  with PRE-event 1 m lidar    : {clean} ({100*clean/n:.0f}%)   <- clean")
    print(f"  only POST-event lidar       : {n - clean - nocov} "
          f"({100*(n-clean-nocov)/n:.0f}%)   <- leaks")
    print(f"  no 1 m lidar within 5 km    : {nocov} ({100*nocov/n:.0f}%)")
    print()
    print("by flood year:")
    by = out.groupby("flood_year").agg(
        events=("EVENT_ID", "size"),
        clean=("has_pre_event_lidar", "sum")).reset_index()
    by["pct_clean"] = (100 * by["clean"] / by["events"]).round(0).astype(int)
    print(by.to_string(index=False))
    print()
    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    nm = dict(zip(sets["huc8"], sets["basin_name"]))
    bb = out.groupby("huc8").agg(events=("EVENT_ID", "size"),
                                 clean=("has_pre_event_lidar", "sum")).reset_index()
    bb["basin"] = [nm.get(h, h) for h in bb["huc8"]]
    bb["pct_clean"] = (100 * bb["clean"] / bb["events"]).round(0).astype(int)
    print("by basin:")
    print(bb[["basin", "events", "clean", "pct_clean"]]
          .sort_values("pct_clean", ascending=False).to_string(index=False))
    print()
    print("gap in years for the clean subset (negative = DEM predates flood):")
    cg = out[out["has_pre_event_lidar"]]["gap_years"]
    print(cg.describe().round(1).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
