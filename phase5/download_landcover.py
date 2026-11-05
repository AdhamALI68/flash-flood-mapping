#!/usr/bin/env python3
"""
Step 1 - land cover, for the farmland mask.

Gate B showed irrigated fields decorrelate in BOTH the baseline and co-event
pairs regardless of flooding, because they are ploughed and watered on their own
schedule. Unmasked, they read as flood everywhere and would contaminate the
threshold work.

NEXT_STEPS specifies NLCD. MRLC has been returning 503 for two days and their S3
object is a 42-byte stub, so ESA WorldCover is used instead: 10 m (finer than
NLCD's 30 m), on AWS, no credentials. Class 40 is cropland.

The substitution is recorded rather than silent - NLCD has annual epochs that
match the 2015-2026 event span, while WorldCover has only 2020/2021. For a mask
that matters little (fields stay fields) but it is a real difference, and NLCD
should be cross-checked against this once MRLC is back.

Downloads only the 3x3 degree tiles covering the 11 basins. Resumable.

Output: data/landcover/worldcover/*.tif, data/landcover/manifest.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/landcover/worldcover"
BASE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
CROPLAND_CLASS = 40


def tile_name(lat: int, lon: int) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"ESA_WorldCover_10m_2021_v200_{ns}{abs(lat):02d}{ew}{abs(lon):03d}_Map.tif"


def main() -> int:
    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    strict = set(sets[sets["set"] == "STRICT"]["huc8"])

    work = ROOT / "data/raw/wbd/extracted"
    frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "geometry"]]
              for r in ("13", "14", "15", "16", "18")]
    huc8 = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    huc8["huc8"] = huc8["huc8"].astype(str).str.zfill(8)
    basins = huc8[huc8["huc8"].isin(strict)]
    minx, miny, maxx, maxy = basins.total_bounds
    print(f"basin bounds: {minx:.2f},{miny:.2f} to {maxx:.2f},{maxy:.2f}")

    # WorldCover ships 3x3 degree tiles on a grid anchored at multiples of 3.
    need = []
    for lat in range(int(math.floor(miny / 3) * 3), int(math.ceil(maxy / 3) * 3), 3):
        for lon in range(int(math.floor(minx / 3) * 3), int(math.ceil(maxx / 3) * 3), 3):
            need.append((lat, lon))
    print(f"tiles needed: {len(need)}\n")

    DEST.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    got, missing = [], []
    for lat, lon in need:
        name = tile_name(lat, lon)
        out = DEST / name
        url = f"{BASE}/{name}"
        if out.exists() and out.stat().st_size > 1_000_000:
            print(f"  {name}  already present")
            got.append(name)
            continue
        try:
            h = session.head(url, timeout=60)
            if h.status_code != 200:
                # Ocean-only tiles genuinely do not exist. Not an error.
                print(f"  {name}  not published (no land) — skipping")
                missing.append(name)
                continue
            size = int(h.headers.get("content-length", 0))
            with session.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = out.with_suffix(".part")
                with tmp.open("wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                if size and tmp.stat().st_size != size:
                    raise RuntimeError(f"size mismatch {tmp.stat().st_size} != {size}")
                tmp.replace(out)
            print(f"  {name}  {out.stat().st_size/1048576:.0f} MB", flush=True)
            got.append(name)
        except Exception as e:
            print(f"  {name}  FAILED: {type(e).__name__}: {e}", flush=True)
            missing.append(name)

    (ROOT / "data/landcover/manifest.json").write_text(json.dumps({
        "source": "ESA WorldCover v200 2021, 10 m",
        "url_base": BASE,
        "substitution_note": ("NLCD specified in NEXT_STEPS; MRLC returned 503 for "
                              "two days and its S3 object is a 42-byte stub. "
                              "Cross-check against NLCD when MRLC returns."),
        "cropland_class": CROPLAND_CLASS,
        "tiles_downloaded": got,
        "tiles_unavailable": missing,
    }, indent=2))

    total = sum((DEST / n).stat().st_size for n in got if (DEST / n).exists())
    print(f"\ndownloaded {len(got)} tiles, {total/1024**3:.2f} GB")
    print(f"wrote data/landcover/manifest.json")
    return 0 if got else 1


if __name__ == "__main__":
    raise SystemExit(main())
