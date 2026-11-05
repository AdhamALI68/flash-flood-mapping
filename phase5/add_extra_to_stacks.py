#!/usr/bin/env python3
"""
Phase 5 - sample sink depth and channel distance onto each event grid.

Same pattern as add_routed_to_stacks: the features live on a 100 m basin grid,
the model trains on the event's 20 m window. The event grid is read from the
label file that already exists, so the new bands land pixel-for-pixel on the
same grid as everything else.

Output: data/features*/<id>/extra.tif  (sink_depth, dist_stream, dist_big_stream)
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"
BANDS = [("sink", "sink_depth"), ("dist_stream", "dist_stream"),
         ("dist_big_stream", "dist_big_stream")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    args = ap.parse_args()
    feat = ROOT / args.features
    idx = pd.read_csv(feat / "_index.csv", dtype={"id": str, "huc8": str})
    print(f"{len(idx)} events in {args.features}", flush=True)
    done = skip = 0
    for n, r in enumerate(idx.itertuples(index=False), 1):
        d = feat / r.id
        dest = d / "extra.tif"
        ref = d / "label.tif"
        if dest.exists():
            done += 1; continue
        if not ref.exists():
            skip += 1; continue
        srcs = [(BF / str(r.huc8).zfill(8) / f"{f}.tif", nm) for f, nm in BANDS]
        if not all(p.exists() for p, _ in srcs):
            skip += 1; continue
        with rasterio.open(ref) as s:
            crs, tf, shape = s.crs, s.transform, (s.height, s.width)
        out = np.full((len(srcs), *shape), np.nan, dtype="float32")
        try:
            for i, (p, _) in enumerate(srcs):
                with rasterio.open(p) as s:
                    reproject(source=rasterio.band(s, 1), destination=out[i],
                              dst_transform=tf, dst_crs=crs,
                              src_nodata=np.nan, dst_nodata=np.nan,
                              resampling=Resampling.bilinear)
            if not np.isfinite(out).any():
                skip += 1; continue
            prof = dict(driver="GTiff", height=shape[0], width=shape[1],
                        count=len(srcs), dtype="float32", crs=crs, transform=tf,
                        nodata=np.nan, compress="deflate", tiled=True,
                        blockxsize=256, blockysize=256)
            with rasterio.open(dest, "w", **prof) as o:
                for i, (_, nm) in enumerate(srcs, 1):
                    o.write(out[i - 1], i); o.set_band_description(i, nm)
            done += 1
        except Exception as e:
            print(f"  {r.id}: {type(e).__name__}: {e}", flush=True); skip += 1
        if n % 150 == 0:
            print(f"  [{n}/{len(idx)}] {done} written", flush=True)
    print(f"\nwritten {done}, skipped {skip}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
