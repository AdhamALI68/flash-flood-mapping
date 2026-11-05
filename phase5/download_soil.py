#!/usr/bin/env python3
"""
POLARIS soil properties over the 11 basins.

NEXT_STEPS names SSURGO. POLARIS is SSURGO gridded to a continuous 30 m raster,
which is what a model can actually consume - raw SSURGO is a polygon database
keyed to map units and needs a join per county before it becomes a raster.
Same underlying survey, usable form.

Variables, chosen for runoff generation rather than completeness:

  ksat   saturated hydraulic conductivity - the infiltration rate. In arid
         terrain a flood happens when rainfall intensity exceeds this, so it is
         the single most relevant soil property here
  clay   fine texture, low infiltration, crusting
  sand   coarse texture, high infiltration, transmission losses down a wash
  bd     bulk density - compaction, another control on infiltration

Depth 0-5 cm only. Infiltration-excess runoff, which is what generates desert
flash floods, is controlled at the surface; deeper horizons matter for
saturation-excess, which is not the mechanism here.

Resumable. Output: data/soil/<var>/*.tif
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://hydrology.cee.duke.edu/POLARIS/PROPERTIES/v1.0"
DEST = ROOT / "data/soil"
VARS = ["ksat", "clay", "sand", "bd"]
DEPTH = "0_5"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    strict = set(sets[sets["set"] == "STRICT"]["huc8"])
    work = ROOT / "data/raw/wbd/extracted"
    frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "geometry"]]
              for r in ("13", "14", "15", "16", "18")]
    huc8 = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    huc8["huc8"] = huc8["huc8"].astype(str).str.zfill(8)
    minx, miny, maxx, maxy = huc8[huc8["huc8"].isin(strict)].total_bounds
    print(f"basin bounds: {minx:.2f},{miny:.2f} to {maxx:.2f},{maxy:.2f}")

    # POLARIS ships 1 x 1 degree tiles named lat{lo}{hi}_lon{lo}{hi}
    tiles = []
    for lat in range(math.floor(miny), math.ceil(maxy)):
        for lon in range(math.floor(minx), math.ceil(maxx)):
            tiles.append(f"lat{lat}{lat+1}_lon{lon}{lon+1}")
    if args.limit:
        tiles = tiles[: args.limit]
    print(f"{len(tiles)} tiles x {len(VARS)} variables = {len(tiles)*len(VARS)} files")
    print(f"~{len(tiles)*len(VARS)*33/1024:.1f} GB\n", flush=True)

    session = requests.Session()
    ok = missing = 0
    for v in VARS:
        d = DEST / v
        d.mkdir(parents=True, exist_ok=True)
        for t in tiles:
            out = d / f"{t}.tif"
            if out.exists() and out.stat().st_size > 100000:
                ok += 1
                continue
            url = f"{BASE}/{v}/mean/{DEPTH}/{t}.tif"
            try:
                r = session.get(url, timeout=300, stream=True)
                if r.status_code == 404:
                    missing += 1
                    continue
                r.raise_for_status()
                tmp = out.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                tmp.replace(out)
                ok += 1
                print(f"  {v}/{t}  {out.stat().st_size/1048576:.0f} MB", flush=True)
            except Exception as e:
                missing += 1
                print(f"  {v}/{t}  FAILED {type(e).__name__}", flush=True)

    total = sum(f.stat().st_size for f in DEST.rglob("*.tif"))
    print(f"\ndownloaded {ok}, unavailable {missing}")
    print(f"total {total/1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
