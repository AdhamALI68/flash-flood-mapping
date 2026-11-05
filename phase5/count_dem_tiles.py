#!/usr/bin/env python3
"""
NEXT_STEPS section 2 - how many 1 m tiles are actually needed?

Full wall-to-wall coverage of the 11 STRICT basins is 755 tiles / 204 GB. But
the pipeline only needs 1 m near events, so the real requirement is the tiles
intersecting a buffer around each event point. This counts that, at several
buffer radii, so the tradeoff is visible rather than assumed.

Two things make the naive count wrong:

  1. Each basin is covered by 5-12 separate lidar projects, so the same ground
     is indexed many times. 3DEP tile titles carry a UTM grid reference -
     "USGS 1 Meter 12 x49y357 AZ_PimaCounty_2021_B21" - so (zone, x, y) is a
     stable key for "same patch of ground". Without deduplicating on it the
     index reports 3,373 tiles for what is ~755 distinct locations.

  2. Events cluster heavily, so buffers overlap. They are dissolved before
     intersecting.

Counted for both event sets:
  623 - the frozen Phase 3 dataset
  766 - before the interval-matching and DST filters, in case those are relaxed

Output: data/dem_tile_counts.csv, data/dem_tiles_needed.csv
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import box

ROOT = Path(__file__).resolve().parent.parent
TNM = "https://tnmaccess.nationalmap.gov/api/v1/products"
DATASET = "Digital Elevation Model (DEM) 1 meter"
EQUAL_AREA = "EPSG:5070"

# "USGS 1 Meter 12 x49y357 AZ_BrawleyRillito_2018_D19"
TITLE_RE = re.compile(r"USGS\s+1\s+Meter\s+(\d+)\s+x(\d+)y(\d+)\s+(\S+)", re.I)

MEDIAN_TILE_MB = 277


PAGE = 100          # larger pages reliably time the API out
RETRIES = 8
MAX_BBOX_DEG = 0.5  # split anything wider than this before querying
CACHE = ROOT / "data/raw/tnm_tile_cache"


def _get(session: requests.Session, params: dict) -> dict:
    """TNM returns 5xx under load; back off and retry rather than lose the run."""
    delay = 4
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(TNM, params=params, timeout=300)
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError) as e:
            if attempt == RETRIES:
                raise
            print(f"      retry {attempt}/{RETRIES} after {type(e).__name__}; "
                  f"waiting {delay}s", flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 120)
    return {}


def split_bbox(bounds, step=MAX_BBOX_DEG):
    """Break a basin bbox into API-sized pieces.

    Whole-basin bboxes span degrees and reliably 504. Many small requests
    succeed where one large one does not.
    """
    minx, miny, maxx, maxy = bounds
    xs, x = [], minx
    while x < maxx:
        xs.append((x, min(x + step, maxx))); x += step
    ys, y = [], miny
    while y < maxy:
        ys.append((y, min(y + step, maxy))); y += step
    return [(a, c, b, d) for a, b in xs for c, d in ys]


def fetch_tiles(session: requests.Session, bounds) -> pd.DataFrame:
    """All 1 m tile records intersecting a bbox, split and paginated."""
    rows = []
    for sub in split_bbox(bounds):
        rows.extend(_fetch_one(session, sub))
    return pd.DataFrame(rows)


def _fetch_one(session: requests.Session, bounds) -> list[dict]:
    minx, miny, maxx, maxy = bounds
    rows, offset = [], 0
    while True:
        d = _get(session, {
            "datasets": DATASET,
            "bbox": f"{minx},{miny},{maxx},{maxy}",
            "max": PAGE, "offset": offset, "outputFormat": "JSON",
        })
        items = d.get("items", [])
        for it in items:
            bb = it.get("boundingBox") or {}
            if not bb:
                continue
            title = it.get("title", "")
            m = TITLE_RE.match(title)
            rows.append({
                "title": title,
                "zone": m.group(1) if m else None,
                "gx": m.group(2) if m else None,
                "gy": m.group(3) if m else None,
                "project": m.group(4) if m else title.split()[-1],
                "size_mb": (it.get("sizeInBytes") or 0) / 1048576,
                "geometry": box(bb["minX"], bb["minY"], bb["maxX"], bb["maxY"]),
            })
        total = d.get("total", 0)
        offset += len(items)
        if not items or offset >= total:
            break
        time.sleep(0.3)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--radii", default="2,5,10", help="buffer radii in km")
    args = ap.parse_args()
    radii = [float(x) for x in args.radii.split(",")]

    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    strict = sets[sets["set"] == "STRICT"]

    frozen = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    frozen["huc8"] = frozen["huc8"].astype(str).str.zfill(8)
    av = pd.read_csv(ROOT / "data/sar_availability.csv", low_memory=False)
    av["huc8"] = av["huc8"].astype(str).str.zfill(8)
    prefilter = av[(av["usable"] == True) & (av["huc8"].isin(strict["huc8"]))]

    event_sets = {
        "frozen_623": frozen[["BEGIN_LON", "BEGIN_LAT"]].dropna(),
        "prefilter_766": prefilter[["BEGIN_LON", "BEGIN_LAT"]].dropna()
        if "BEGIN_LON" in prefilter.columns else None,
    }
    if event_sets["prefilter_766"] is None:
        ev = pd.read_csv(ROOT / "data/events.csv", low_memory=False)
        ev["huc8"] = ev["huc8"].astype(str).str.zfill(8)
        merged = prefilter[["EVENT_ID"]].merge(
            ev[["EVENT_ID", "BEGIN_LON", "BEGIN_LAT"]], on="EVENT_ID", how="left")
        event_sets["prefilter_766"] = merged[["BEGIN_LON", "BEGIN_LAT"]].dropna()

    # ---- tile index over the strict basins ------------------------------
    work = ROOT / "data/raw/wbd/extracted"
    frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "geometry"]]
              for r in ("13", "14", "15", "16", "18")]
    huc8 = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    huc8["huc8"] = huc8["huc8"].astype(str).str.zfill(8)
    basins = huc8[huc8["huc8"].isin(strict["huc8"])]

    CACHE.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    all_tiles = []
    for r in basins.itertuples():
        name = strict[strict.huc8 == r.huc8]["basin_name"].iloc[0]
        cached = CACHE / f"{r.huc8}.csv"
        if cached.exists():
            df = pd.read_csv(cached, dtype=str)
            df["size_mb"] = pd.to_numeric(df["size_mb"], errors="coerce").fillna(0)
            df["geometry"] = gpd.GeoSeries.from_wkt(df["geometry_wkt"])
            print(f"  {r.huc8} {name:<34} {len(df):>5} tile records (cached)", flush=True)
        else:
            df = fetch_tiles(session, r.geometry.bounds)
            out = df.copy()
            out["geometry_wkt"] = out["geometry"].apply(lambda g: g.wkt)
            out.drop(columns="geometry").to_csv(cached, index=False)
            print(f"  {r.huc8} {name:<34} {len(df):>5} tile records", flush=True)
        df["huc8"] = r.huc8
        all_tiles.append(df)
    tiles = pd.concat(all_tiles, ignore_index=True)
    tiles = gpd.GeoDataFrame(tiles, geometry="geometry", crs="EPSG:4326")
    print(f"\nraw tile records across strict basins : {len(tiles):,}")

    # Deduplicate to one tile per patch of ground.
    keyed = tiles[tiles["gx"].notna()].copy()
    keyed["grid_key"] = keyed["zone"] + "_" + keyed["gx"] + "_" + keyed["gy"]
    unkeyed = tiles[tiles["gx"].isna()]
    if len(unkeyed):
        print(f"  {len(unkeyed)} records had an unparseable title (counted separately)")
    # Prefer the largest file per location - a truncated tile is usually a
    # partial-coverage edge product.
    uniq = (keyed.sort_values("size_mb", ascending=False)
                 .drop_duplicates("grid_key", keep="first"))
    print(f"distinct ground locations (deduplicated) : {len(uniq):,}")
    print(f"  covered by {keyed['project'].nunique()} lidar projects")

    uniq_ea = uniq.to_crs(EQUAL_AREA)

    # ---- count tiles hit by event buffers -------------------------------
    rows = []
    for set_name, pts in event_sets.items():
        g = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(pts["BEGIN_LON"], pts["BEGIN_LAT"]),
            crs="EPSG:4326").to_crs(EQUAL_AREA)
        for km in radii:
            dissolved = g.buffer(km * 1000).union_all()
            hit = uniq_ea[uniq_ea.intersects(dissolved)]
            mb = hit["size_mb"].replace(0, pd.NA).fillna(MEDIAN_TILE_MB).sum()
            rows.append({
                "event_set": set_name, "n_events": len(pts),
                "buffer_km": km, "tiles": len(hit),
                "gb": round(mb / 1024, 1),
                "pct_of_full": round(100 * len(hit) / len(uniq), 1),
            })
            print(f"  {set_name:<15} {km:>4.0f} km buffer -> {len(hit):>4} tiles, "
                  f"{mb/1024:>6.1f} GB ({100*len(hit)/len(uniq):.0f}% of full coverage)",
                  flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/dem_tile_counts.csv", index=False)

    # Persist the actual tile list at 5 km on the frozen set - the download list.
    g = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(event_sets["frozen_623"]["BEGIN_LON"],
                                    event_sets["frozen_623"]["BEGIN_LAT"]),
        crs="EPSG:4326").to_crs(EQUAL_AREA)
    hit = uniq_ea[uniq_ea.intersects(g.buffer(5000).union_all())]
    hit.drop(columns="geometry").to_csv(ROOT / "data/dem_tiles_needed.csv", index=False)

    print()
    print("=" * 68)
    print(f"full wall-to-wall coverage : {len(uniq):,} tiles, "
          f"{uniq['size_mb'].replace(0, pd.NA).fillna(MEDIAN_TILE_MB).sum()/1024:.0f} GB")
    print(out.to_string(index=False))
    print()
    print("wrote data/dem_tile_counts.csv and data/dem_tiles_needed.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
