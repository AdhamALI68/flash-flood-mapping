#!/usr/bin/env python3
"""
Phase 2c - Aridity screen.

Gate A and the lidar check selected basins on data availability alone, which is
"arid by state line". A dataset whose headline claim is *arid* has to select
basins by a measured climate variable, or the first review question is why a
coastal Mediterranean basin is in it.

Screen: PRISM 30-year mean annual precipitation, 4 km, averaged over the pixels
falling inside each basin. Conventional bands - arid < 250 mm, semi-arid
250-500 mm, sub-humid > 500 mm.

Note this is an unweighted pixel mean, not area-weighted. PRISM is on a
geographic grid so pixel ground area shrinks with latitude; across 32-38 N the
spread is only a few percent, well inside the band widths, but the paper should
describe it accurately.

Mean annual precipitation is a proxy. The stricter definition is the UNEP
aridity index P/PET, which needs PET this does not compute; the bands below are
wide enough that the ranking is unlikely to change, but the paper should say
which one it used.

Output: data/basin_aridity.csv
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.mask import mask

# PRISM centralised its web services in 2025; the old /normals/4km/ppt/annual
# path now returns a migration notice rather than data.
PRISM = "https://services.nacse.org/prism/data/get/normals/us/4km/ppt/annual"
REGIONS = ["13", "14", "15", "16", "18"]


def _raster_in(dest: Path) -> Path | None:
    """PRISM ships GeoTIFF now; older archives were ESRI BIL."""
    for pat in ("*.tif", "*.bil"):
        found = next(dest.glob(pat), None)
        if found:
            return found
    return None


def fetch_prism(dest: Path) -> Path:
    """Download and unpack the PRISM annual precipitation normals raster."""
    existing = _raster_in(dest)
    if existing:
        return existing
    dest.mkdir(parents=True, exist_ok=True)
    r = requests.get(PRISM, timeout=600)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        z.extractall(dest)
    found = _raster_in(dest)
    if not found:
        raise SystemExit(f"no raster inside the PRISM archive: {sorted(p.name for p in dest.iterdir())}")
    return found


def load_huc8(work: Path) -> gpd.GeoDataFrame:
    frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "name", "geometry"]]
              for r in REGIONS]
    g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    g["huc8"] = g["huc8"].astype(str).str.zfill(8)
    return g


def band(mm: float) -> str:
    if mm < 250:
        return "arid"
    if mm < 500:
        return "semi-arid"
    return "sub-humid"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coverage", default="data/lidar_coverage.csv")
    ap.add_argument("--out", default="data/basin_aridity.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    work = root / "data" / "raw" / "wbd" / "extracted"

    cand = pd.read_csv(root / args.coverage)
    cand["huc8"] = cand["huc8"].astype(str).str.zfill(8)

    print("fetching PRISM annual precipitation normals...")
    bil = fetch_prism(root / "data" / "raw" / "prism_ppt")
    print(f"  raster: {bil.name}")

    huc8 = load_huc8(work).set_index("huc8")

    rows = []
    with rasterio.open(bil) as src:
        nodata = src.nodata
        for r in cand.itertuples():
            geom = huc8.loc[r.huc8, "geometry"]
            if hasattr(geom, "iloc"):
                geom = geom.iloc[0]
            gdf = gpd.GeoSeries([geom], crs=huc8.crs).to_crs(src.crs)
            arr, _ = mask(src, [gdf.iloc[0]], crop=True, filled=True,
                          nodata=nodata if nodata is not None else -9999)
            a = arr[0].astype("float64")
            a = a[np.isfinite(a)]
            if nodata is not None:
                a = a[a != nodata]
            a = a[a > -1000]
            if a.size == 0:
                continue
            rows.append({
                "huc8": r.huc8, "basin_name": r.basin_name, "huc4": r.huc4,
                "sar_usable_events": r.sar_usable_events,
                "lidar_coverage_frac": r.lidar_coverage_frac,
                "ppt_mm_mean": round(float(a.mean()), 1),
                "ppt_mm_min": round(float(a.min()), 1),
                "ppt_mm_max": round(float(a.max()), 1),
                "aridity_band": band(float(a.mean())),
            })

    df = pd.DataFrame(rows).sort_values("ppt_mm_mean")
    df.to_csv(root / args.out, index=False)
    print(f"\nwrote {args.out}\n")

    print("basins by aridity band:")
    print(df["aridity_band"].value_counts().to_string())
    print()

    strict = df[(df["aridity_band"] == "arid") &
                (df["sar_usable_events"] >= 20) &
                (df["lidar_coverage_frac"] >= 0.9)]
    incl = df[(df["aridity_band"].isin(["arid", "semi-arid"])) &
              (df["sar_usable_events"] >= 20) &
              (df["lidar_coverage_frac"] >= 0.9)]
    print(f"STRICT  (arid only, >=20 events, >=90% lidar): "
          f"{len(strict)} basins, {strict['huc4'].nunique()} HUC4 regions, "
          f"{int(strict['sar_usable_events'].sum())} events")
    print(f"INCLUSIVE (arid + semi-arid, same filters)  : "
          f"{len(incl)} basins, {incl['huc4'].nunique()} HUC4 regions, "
          f"{int(incl['sar_usable_events'].sum())} events")
    print()
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
