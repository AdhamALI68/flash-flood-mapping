#!/usr/bin/env python3
"""
Choose WHICH 1 m tile to download for each location, per NEXT_STEPS 3b.

Each ground location is covered by up to a dozen lidar projects flown in
different years, so a rule is required. Two constraints pull against each other:

  - a survey flown AFTER a flood contains that flood's own scour, so using it as
    a model input leaks the answer through the terrain channel
  - one location often serves events spanning many years, and no single survey
    can predate all of them

Rule applied identically everywhere:

  1. for each candidate survey year at a location, count how many of the events
     it serves would be CLEAN (flood year strictly after the survey year)
  2. take the year maximising clean events
  3. tie-break on the more recent year (newer lidar is better quality)
  4. tie-break on larger file size - a 4 MB "1 m tile" is a partial edge product
     covering almost no ground, against ~100 MB for a full one

Records, per tile and per event, which project was chosen and the SIGNED year
gap, so the leaking subset can be separated in analysis rather than discovered
later.

Output: data/dem_download_manifest.csv, data/dem_event_vintage.csv
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
BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/1m/Projects"

# Below this, a "tile" is an edge sliver with almost no real ground in it.
MIN_USEFUL_MB = 20.0


def survey_year(project: str) -> int | None:
    """Explicit year, else the delivery code suffix (_D24 -> 2024)."""
    p = str(project)
    m = re.search(r"(20\d{2})", p)
    if m:
        return int(m.group(1))
    m = re.search(r"_[A-Z](\d{2})(?:_|$)", p) or re.search(r"[A-Z](\d{2})$", p)
    return 2000 + int(m.group(1)) if m else None


def tile_url(project: str, zone: str, gx: str, gy: str) -> str:
    return f"{BASE}/{project}/TIFF/USGS_1M_{zone}_x{gx}y{gy}_{project}.tif"


def main() -> int:
    files = glob.glob(str(ROOT / "data/raw/tnm_tile_cache/*.csv"))
    t = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    t["size_mb"] = pd.to_numeric(t["size_mb"], errors="coerce").fillna(0)
    t["survey_year"] = t["project"].apply(survey_year)
    t = t[t["survey_year"].notna() & t["gx"].notna() & t["geometry_wkt"].notna()].copy()
    t["survey_year"] = t["survey_year"].astype(int)
    t["grid_key"] = t["zone"] + "_" + t["gx"] + "_" + t["gy"]
    t = t.drop_duplicates(["grid_key", "project"])
    print(f"tile records: {len(t):,} over {t['grid_key'].nunique():,} locations, "
          f"{t['project'].nunique()} projects")

    g = gpd.GeoDataFrame(t, geometry=gpd.GeoSeries.from_wkt(t["geometry_wkt"]),
                         crs="EPSG:4326").to_crs(EQUAL_AREA)

    ev = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    ev["flood_year"] = pd.to_datetime(ev["BEGIN_UTC"]).dt.year
    ep = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.BEGIN_LON, ev.BEGIN_LAT),
                          crs="EPSG:4326").to_crs(EQUAL_AREA)
    ep["geometry"] = ep.buffer(BUFFER_M)

    # One location can serve many events and vice versa.
    hit = gpd.sjoin(g[["grid_key", "geometry"]].drop_duplicates("grid_key"),
                    ep[["EVENT_ID", "flood_year", "geometry"]],
                    how="inner", predicate="intersects")
    served = hit.groupby("grid_key")["flood_year"].apply(list).to_dict()
    print(f"locations intersecting a 5 km event buffer: {len(served):,}")

    rows, ev_rows = [], []
    for key, years in served.items():
        cand = t[t["grid_key"] == key]
        if cand.empty:
            continue
        best = None
        for y in sorted(cand["survey_year"].unique()):
            sub = cand[cand["survey_year"] == y]
            big = sub[sub["size_mb"] >= MIN_USEFUL_MB]
            use = big if not big.empty else sub
            clean = sum(1 for fy in years if fy > y)
            score = (clean, y, float(use["size_mb"].max()))
            if best is None or score > best[0]:
                best = (score, use.sort_values("size_mb", ascending=False).iloc[0], y, clean)
        if best is None:
            continue
        (_, row, year, clean) = best
        rows.append({
            "grid_key": key, "project": row["project"], "survey_year": year,
            "zone": row["zone"], "gx": row["gx"], "gy": row["gy"],
            "size_mb": round(float(row["size_mb"]), 1),
            "events_served": len(years), "events_clean": clean,
            "n_candidate_projects": int(cand["project"].nunique()),
            "url": tile_url(row["project"], row["zone"], row["gx"], row["gy"]),
        })
        for fy in years:
            ev_rows.append({"grid_key": key, "flood_year": fy, "survey_year": year,
                            "gap_years": year - fy, "clean": fy > year})

    man = pd.DataFrame(rows).sort_values("grid_key")
    man.to_csv(ROOT / "data/dem_download_manifest.csv", index=False)
    evd = pd.DataFrame(ev_rows)
    evd.to_csv(ROOT / "data/dem_event_vintage.csv", index=False)

    total_gb = man["size_mb"].sum() / 1024
    print()
    print("=" * 66)
    print(f"tiles to download : {len(man):,}")
    print(f"total size        : {total_gb:.1f} GB")
    print(f"survey years      : {sorted(man['survey_year'].unique())}")
    print(f"tiles under {MIN_USEFUL_MB:g} MB kept (no better option): "
          f"{(man['size_mb'] < MIN_USEFUL_MB).sum()}")
    print()
    print("event-tile pairs by cleanliness:")
    print(f"  clean (DEM predates flood) : {int(evd['clean'].sum()):,} "
          f"({100*evd['clean'].mean():.0f}%)")
    print(f"  leaking (DEM postdates)    : {int((~evd['clean']).sum()):,}")
    print()
    print("chosen survey year distribution:")
    print(man["survey_year"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
