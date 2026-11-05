#!/usr/bin/env python3
"""
3DEP 10 m seamless DEM, basin-wide. Needed for FLOW ROUTING.

NEXT_STEPS section 3 splits the terrain two ways:
  1 m  local, around events - micro-relief, where fine detail matters
  10 m basin-wide           - flow routing and drainage-area context

The 1 m tiles cover only a 5 km buffer around each event, which is far too
small for flow accumulation: water arriving at a wash was collected over the
whole upstream basin, sometimes 100 km away. Routing on a clipped DEM invents a
watershed boundary at the clip edge and silently truncates drainage area.

Trial 1 measured this directly - routing over the complete basin rather than per
tile was worth +0.073 tile AUC, its single largest gain. Reproducing that needs
basin-wide coverage.

10 m (1/3 arcsec) rather than 30 m: 15.5 GB against 1.9 GB, and drainage
networks in low-relief desert basins are poorly resolved at 30 m.

This is a NATIVE product. It is for routing only and must never be used as an
ablation rung - those are degraded from the 1 m source (section 3c). Kept in a
separate directory so the two can never be confused.

Output: data/dem10_routing/*.tif
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://prd-tnm.s3.amazonaws.com/StagedProducts/Elevation/13/TIFF/current"
DEST = ROOT / "data/dem10_routing"
STATE = ROOT / "data/dem10_state.json"


def tile_name(lat: int, lon: int) -> str:
    """USGS names a tile by its NORTH-WEST corner."""
    return f"n{lat+1:02d}w{abs(lon):03d}"


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

    tiles = [tile_name(lat, lon)
             for lat in range(math.floor(miny), math.ceil(maxy))
             for lon in range(math.floor(minx), math.ceil(maxx))]
    if args.limit:
        tiles = tiles[: args.limit]
    print(f"{len(tiles)} tiles, ~{len(tiles)*432/1024:.1f} GB\n", flush=True)

    DEST.mkdir(parents=True, exist_ok=True)
    st = json.loads(STATE.read_text()) if STATE.exists() else {}
    session = requests.Session()
    ok = missing = 0
    for i, t in enumerate(tiles, 1):
        out = DEST / f"USGS_13_{t}.tif"
        if out.exists() and out.stat().st_size > 1_000_000:
            ok += 1
            continue
        url = f"{BASE}/{t}/USGS_13_{t}.tif"
        # These are ~430 MB each over WiFi, so a dropped connection partway is
        # normal. Retry with backoff rather than recording a permanent failure -
        # a single missing tile leaves a hole in the routed basin.
        done_this = False
        for attempt in range(1, 5):
            try:
                r = session.get(url, timeout=900, stream=True)
                if r.status_code == 404:
                    st[t] = "not published"   # ocean / outside CONUS, not an error
                    missing += 1
                    done_this = True
                    break
                r.raise_for_status()
                tmp = out.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                exp = int(r.headers.get("content-length", 0))
                if exp and tmp.stat().st_size != exp:
                    raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {exp}")
                tmp.replace(out)
                st[t] = "ok"
                ok += 1
                done_this = True
                print(f"  [{i}/{len(tiles)}] {t}  {out.stat().st_size/1048576:.0f} MB",
                      flush=True)
                break
            except Exception as e:
                if attempt == 4:
                    st[t] = f"failed: {type(e).__name__}"
                    missing += 1
                    print(f"  [{i}/{len(tiles)}] {t}  FAILED after 4 tries "
                          f"({type(e).__name__})", flush=True)
                else:
                    wait = 15 * attempt
                    print(f"  [{i}/{len(tiles)}] {t}  retry {attempt}/4 after "
                          f"{type(e).__name__}, waiting {wait}s", flush=True)
                    time.sleep(wait)
        if not done_this:
            pass
        STATE.write_text(json.dumps(st, indent=1))

    total = sum(f.stat().st_size for f in DEST.glob("*.tif"))
    print(f"\ndownloaded {ok}, unavailable {missing}")
    print(f"total {total/1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
