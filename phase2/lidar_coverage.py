#!/usr/bin/env python3
"""
Phase 2b - 3DEP 1 m lidar coverage per basin.

Gate A proved the SAR scenes exist. This asks the other half: which of those
basins actually have 1 m lidar, since the resolution ablation is meaningless
without it. A basin needs BOTH to be a candidate.

Coverage is computed as (union of 1 m tile footprints) intersected with the
basin, over basin area, in an equal-area projection. The National Map returns
tile bounding boxes in lat/lon rather than true footprints, so treat the result
as a close approximation for ranking basins, not as a final number.

Output: data/lidar_coverage.csv
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "Digital Elevation Model (DEM) 1 meter"

# CONUS Albers equal area - areas in m^2 are meaningful here, unlike EPSG:4326.
EQUAL_AREA = "EPSG:5070"

REGIONS = ["13", "14", "15", "16", "18"]


def fetch_tiles(session: requests.Session, bounds: tuple) -> gpd.GeoDataFrame:
    """Every 1 m DEM tile footprint intersecting a bounding box, paginated."""
    minx, miny, maxx, maxy = bounds
    geoms, titles, offset = [], [], 0
    while True:
        r = session.get(
            TNM,
            params={
                "datasets": DATASET,
                "bbox": f"{minx},{miny},{maxx},{maxy}",
                "max": 1000,
                "offset": offset,
                "outputFormat": "JSON",
            },
            timeout=180,
        )
        r.raise_for_status()
        d = r.json()
        items = d.get("items", [])
        for it in items:
            bb = it.get("boundingBox") or {}
            if not bb:
                continue
            geoms.append(box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]))
            titles.append(it.get("title", ""))
        total = d.get("total", 0)
        offset += len(items)
        if not items or offset >= total:
            break
        time.sleep(0.3)  # be polite to the API
    if not geoms:
        return gpd.GeoDataFrame(columns=["title", "geometry"], geometry="geometry", crs="EPSG:4326")
    return gpd.GeoDataFrame({"title": titles, "geometry": geoms}, crs="EPSG:4326")


def load_huc8(work: Path) -> gpd.GeoDataFrame:
    frames = []
    for r in REGIONS:
        g = gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")
        frames.append(g[["huc8", "name", "geometry"]])
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default="data/sar_basin_summary.csv")
    ap.add_argument("--top", type=int, default=30, help="check the N basins with most usable events")
    ap.add_argument("--out", default="data/lidar_coverage.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    work = root / "data" / "raw" / "wbd" / "extracted"

    sar = pd.read_csv(root / args.summary)
    sar["huc8"] = sar["huc8"].astype(str).str.zfill(8)
    sar = sar.sort_values("usable", ascending=False).head(args.top)

    huc8 = load_huc8(work)
    huc8["huc8"] = huc8["huc8"].astype(str).str.zfill(8)
    huc8 = huc8.set_index("huc8")

    session = requests.Session()
    rows = []
    for i, r in enumerate(sar.itertuples(), 1):
        if r.huc8 not in huc8.index:
            continue
        poly = huc8.loc[r.huc8, "geometry"]
        if hasattr(poly, "iloc"):
            poly = poly.iloc[0]

        try:
            tiles = fetch_tiles(session, poly.bounds)
        except Exception as e:
            print(f"[{i}/{len(sar)}] {r.huc8} {r.basin_name}: TNM query failed - {e}")
            continue

        basin_ea = gpd.GeoSeries([poly], crs="EPSG:4326").to_crs(EQUAL_AREA).iloc[0]
        if tiles.empty:
            frac, n_proj = 0.0, 0
        else:
            tiles_ea = tiles.to_crs(EQUAL_AREA)
            covered = tiles_ea.union_all().intersection(basin_ea)
            frac = covered.area / basin_ea.area
            # Project name is the trailing token of the tile title.
            n_proj = tiles["title"].str.split().str[-1].nunique()

        rows.append({
            "huc8": r.huc8, "basin_name": r.basin_name, "huc4": r.huc4,
            "sar_usable_events": r.usable,
            "lidar_1m_tiles": len(tiles),
            "lidar_projects": n_proj,
            "lidar_coverage_frac": round(frac, 3),
            "basin_area_km2": round(basin_ea.area / 1e6, 1),
        })
        print(f"[{i}/{len(sar)}] {r.huc8} {r.basin_name:<34} "
              f"{r.usable:>4} SAR events | {len(tiles):>5} tiles | "
              f"{100*frac:>5.1f}% covered", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(root / args.out, index=False)

    print("\n" + "=" * 78)
    print(f"wrote {args.out}")
    print()
    print("CANDIDATE BASINS - need SAR events AND 1 m lidar")
    for thr in (0.9, 0.5, 0.2):
        sel = df[(df["lidar_coverage_frac"] >= thr) & (df["sar_usable_events"] >= 20)]
        print(f"  >= {int(thr*100):>2}% lidar cover and >= 20 usable events: "
              f"{len(sel)} basins, {sel['huc4'].nunique()} distinct HUC4 regions")
    print()
    best = df.sort_values(["lidar_coverage_frac", "sar_usable_events"], ascending=False)
    print(best.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
