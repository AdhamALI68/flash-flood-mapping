#!/usr/bin/env python3
"""
Phase 5, step 2b - sample the catchment-routed rainfall onto each event grid.

WHY THIS IS A SEPARATE STEP
    The routed bands live on a 100 m basin-wide grid; the model trains on the
    event's own 20 m window. Rebuilding all 617 feature stacks to add four bands
    would cost many hours and redo the 1 m terrain degradation, which has not
    changed. This only reads the event's existing grid definition and warps the
    routed raster onto it, so it runs in seconds per event.

    The event grid is taken from label.tif in the stack that already exists, so
    the new bands land pixel-for-pixel on the same grid as everything else. No
    re-registration, and no chance of the 2.5 km offset that bit us before.

UPSAMPLING 100 m -> 20 m IS BILINEAR ON PURPOSE
    These are catchment integrals: smooth, and already coarser than the 1 km
    radar they come from. Nearest would put visible 100 m blocks into a 20 m
    feature for no gain in information.

Output: data/features/<id>/rain_routed.tif, then re-run build_tiles.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--routed", default="data/rain_routed")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    feat, routed = ROOT / args.features, ROOT / args.routed
    ids = sorted(p.name for p in feat.iterdir() if p.is_dir())
    if args.limit:
        ids = ids[: args.limit]
    print(f"{len(ids)} event stacks in {args.features}", flush=True)

    done = missing = failed = 0
    for n, eid in enumerate(ids, 1):
        src = routed / f"{eid}.tif"
        dest = feat / eid / "rain_routed.tif"
        ref = feat / eid / "label.tif"
        if dest.exists():
            done += 1
            continue
        if not src.exists() or not ref.exists():
            missing += 1
            continue
        try:
            with rasterio.open(ref) as r:
                crs, transform, shape = r.crs, r.transform, (r.height, r.width)
            with rasterio.open(src) as s:
                names = [s.descriptions[i] or f"routed_{i}" for i in range(s.count)]
                out = np.full((s.count, *shape), np.nan, dtype="float32")
                for i in range(s.count):
                    reproject(source=rasterio.band(s, i + 1), destination=out[i],
                              dst_transform=transform, dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
            if not np.isfinite(out).any():
                failed += 1
                continue
            with rasterio.open(dest, "w", driver="GTiff", height=shape[0],
                               width=shape[1], count=len(names), dtype="float32",
                               crs=crs, transform=transform, nodata=np.nan,
                               compress="deflate", tiled=True,
                               blockxsize=256, blockysize=256) as o:
                for i, nm in enumerate(names, 1):
                    o.write(out[i - 1], i)
                    o.set_band_description(i, nm)
            done += 1
        except Exception as e:
            print(f"  {eid}: {type(e).__name__}: {e}", flush=True)
            failed += 1
        if n % 50 == 0:
            print(f"  [{n}/{len(ids)}] added {done}, "
                  f"no routed raster {missing}, failed {failed}", flush=True)

    print(f"\nadded {done}, no routed raster {missing}, failed {failed}")
    print("next: add 'rain_routed' to SHARED_STACKS in build_tiles.py and re-tile")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
