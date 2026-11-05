#!/usr/bin/env python3
"""
Phase 5 - two features nothing in this project had: sink depth and distance to
the nearest channel.

WHY THESE TWO
    sink_depth = filled_DEM - DEM
        How deep the hollow is at this pixel. Everything we had describes how
        BUMPY the ground is (microrelief, roughness) - nothing said "this is a
        depression that holds water". In a desert full of playas and closed
        basins that is precisely where water sits.

    dist_to_stream
        HORIZONTAL metres to the nearest channel. We had basin_hand, which is
        the VERTICAL height above drainage, and no horizontal distance at all.
        Distance-to-river is routinely among the strongest predictors in the
        flood susceptibility literature and we simply did not have it.

    dist_to_big_stream
        The same, but only to channels draining a large area. A pixel 50 m from
        a minor gully is not in the same danger as one 50 m from a main wash.

BUILT AT 100 m
    Same grid as the repaired HAND, from the cached basin DEMs. Fine enough for
    a distance field, and it keeps the whole job to minutes.

Output: data/basin_features/<huc8>/{sink,dist_stream,dist_big_stream}.tif
"""
from __future__ import annotations
import argparse, shutil, tempfile
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"
DEM100 = ROOT / "data/basin_dem100"
RES = 100.0
# Thresholds in CELLS of 100 m (each 0.01 km2). The first attempt used 5 cells
# (0.05 km2) to match the 10 m routing, but at 100 m that qualifies 31% of the
# basin as "stream" and every pixel ends up within 500 m of one - a distance
# field with no information. A channel network needs a real contributing area.
STREAM_CELLS = 100      # 1 km2  - a defined wash
BIG_CELLS = 1000        # 10 km2 - a main channel


def build(huc8: str, wbt) -> bool:
    dem_p = DEM100 / f"{huc8}.tif"
    if not dem_p.exists():
        print(f"  {huc8}: no 100 m DEM"); return False
    out_sink = BF / huc8 / "sink.tif"
    out_d = BF / huc8 / "dist_stream.tif"
    out_db = BF / huc8 / "dist_big_stream.tif"
    if out_sink.exists() and out_d.exists() and out_db.exists():
        print(f"  {huc8}: already built"); return True

    with rasterio.open(dem_p) as s:
        dem = s.read(1).astype("float32")
        prof = s.profile
    valid = np.isfinite(dem)
    tmp = Path(tempfile.mkdtemp(prefix=f"xf_{huc8}_", dir=str(ROOT / "data")))
    try:
        # whitebox needs a real nodata value, not NaN
        src = tmp / "dem.tif"
        p2 = dict(prof); p2.update(dtype="float32", nodata=-32768.0,
                                   compress="deflate", tiled=True,
                                   blockxsize=512, blockysize=512)
        with rasterio.open(src, "w", **p2) as d:
            d.write(np.where(valid, dem, -32768.0).astype("float32"), 1)

        filled, facc = tmp / "filled.tif", tmp / "facc.tif"
        if wbt.fill_depressions(str(src), str(filled)) != 0 or not filled.exists():
            print(f"  {huc8}: fill_depressions failed"); return False
        if wbt.d8_flow_accumulation(str(filled), str(facc),
                                    out_type="cells") != 0:
            print(f"  {huc8}: flow accumulation failed"); return False

        with rasterio.open(filled) as s:
            fl = s.read(1).astype("float32")
            nd = s.nodata
        if nd is not None:
            fl = np.where(fl == nd, np.nan, fl)
        # depth of the hollow. Clipped at 0: filling can only raise ground.
        sink = np.where(valid, np.maximum(fl - dem, 0.0), np.nan).astype("float32")

        with rasterio.open(facc) as s:
            fa = s.read(1).astype("float32")
            nd = s.nodata
        if nd is not None:
            fa = np.where(fa == nd, np.nan, fa)

        def dist_to(mask):
            # distance_transform_edt measures distance to the nearest ZERO, so
            # the stream mask is inverted before the transform.
            if mask.sum() == 0:
                return np.full(dem.shape, np.nan, "float32")
            d = ndimage.distance_transform_edt(~mask) * RES
            return np.where(valid, d, np.nan).astype("float32")

        dist = dist_to(np.nan_to_num(fa, nan=0) >= STREAM_CELLS)
        dist_big = dist_to(np.nan_to_num(fa, nan=0) >= BIG_CELLS)

        po = dict(prof); po.update(dtype="float32", nodata=np.nan, count=1,
                                   compress="deflate", tiled=True,
                                   blockxsize=512, blockysize=512)
        for path, arr, nm in ((out_sink, sink, "sink_depth"),
                              (out_d, dist, "dist_stream"),
                              (out_db, dist_big, "dist_big_stream")):
            with rasterio.open(path, "w", **po) as o:
                o.write(arr, 1); o.set_band_description(1, nm)
        v = sink[np.isfinite(sink)]
        d1 = dist[np.isfinite(dist)]
        print(f"  {huc8}: sink p99 {np.percentile(v,99):.2f} m, "
              f"max {v.max():.1f} m | dist to stream median "
              f"{np.median(d1):.0f} m, max {d1.max():.0f} m", flush=True)
        return True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--only", default="")
    args = ap.parse_args()
    from whitebox import WhiteboxTools
    wbt = WhiteboxTools(); wbt.verbose = False
    basins = sorted(p.name for p in BF.iterdir() if p.is_dir())
    if args.only:
        basins = [b for b in basins if b == args.only]
    print(f"building sink depth and channel distance for {len(basins)} basins\n")
    ok = sum(build(b, wbt) for b in basins)
    print(f"\n{ok}/{len(basins)} basins done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
