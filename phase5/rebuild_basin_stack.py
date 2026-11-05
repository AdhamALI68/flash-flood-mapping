#!/usr/bin/env python3
"""
Phase 5, step 2c - swap the repaired HAND into each event's basin stack.

WHY
    basin_hand is the strongest single feature we have (within-tile AUC 0.5546),
    and phase5/fix_hand.py showed it was built against a stale stream raster in
    Death Valley, Mojave and Whitewater River. In Mojave only 9.4% of the basin
    had a HAND value at all.

    Because the valid mask is the AND of every channel, a hole in HAND deletes
    the pixel outright. That is why region 1810 produced zero tiles from 32
    events and region 1809 produced tiles from only 54 of 113.

WHAT THIS DOES
    Reads the event's existing basin.tif and writes basin_v2.tif, identical
    except that basin_hand is replaced by the rebuilt 100 m HAND.

    THE OLD 10 m HAND IS NOT CARRIED ALONGSIDE. It is tempting to keep both and
    let the baselines pick, but the valid mask is the AND of every channel, so
    an extra band that is 90% empty in Mojave would delete exactly the pixels
    this repair exists to recover. The comparison is still available and costs
    nothing: data/tiles_v2 was tiled with the 10 m HAND and scored 0.5546, so
    the new run measures the replacement head to head on the same evaluation.

    The rebuild agrees with the old HAND at rho +0.65 to +0.79 across the eight
    basins that were never broken, which is what "the same quantity, measured
    coarser" looks like.

Output: data/features*/<id>/basin_v2.tif
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    feat = ROOT / args.features
    import pandas as pd
    idx = pd.read_csv(feat / "_index.csv", dtype={"id": str, "huc8": str})
    if args.limit:
        idx = idx.head(args.limit)
    print(f"{len(idx)} events in {args.features}", flush=True)

    done = skipped = 0
    for n, r in enumerate(idx.itertuples(index=False), 1):
        d = feat / r.id
        src = d / "basin.tif"
        dest = d / "basin_v2.tif"
        h100 = BF / str(r.huc8).zfill(8) / "hand100.tif"
        if dest.exists():
            done += 1
            continue
        if not src.exists() or not h100.exists():
            skipped += 1
            continue
        try:
            with rasterio.open(src) as s:
                a = s.read().astype("float32")
                names = [s.descriptions[i] or f"b{i}" for i in range(s.count)]
                prof = s.profile
                grid = (s.crs, s.transform, (s.height, s.width))
            new = np.full(grid[2], np.nan, dtype="float32")
            with rasterio.open(h100) as s:
                reproject(source=rasterio.band(s, 1), destination=new,
                          dst_transform=grid[1], dst_crs=grid[0],
                          src_nodata=np.nan, dst_nodata=np.nan,
                          resampling=Resampling.bilinear)
            if "basin_hand" not in names:
                skipped += 1
                continue
            hi = names.index("basin_hand")
            a[hi] = new                       # repaired HAND replaces the old one
            out_names, out = list(names), a

            prof.update(count=len(out_names), dtype="float32", nodata=np.nan,
                        compress="deflate", tiled=True,
                        blockxsize=256, blockysize=256)
            with rasterio.open(dest, "w", **prof) as o:
                for i, nm in enumerate(out_names, 1):
                    o.write(out[i - 1], i)
                    o.set_band_description(i, nm)
            done += 1
        except Exception as e:
            print(f"  {r.id}: {type(e).__name__}: {e}", flush=True)
            skipped += 1
        if n % 100 == 0:
            print(f"  [{n}/{len(idx)}] written {done}, skipped {skipped}", flush=True)

    print(f"\nwritten {done}, skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
