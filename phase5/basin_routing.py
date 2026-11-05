#!/usr/bin/env python3
"""
Phase 5, step 1 - flow routing over COMPLETE basins.

Trial 1's single largest gain (+0.073 tile AUC) came from routing water over the
whole basin instead of per tile. Routing on a clipped DEM invents a watershed
boundary at the clip edge, so every cell near that edge gets a truncated
drainage area - and drainage area is most of the signal in flood prediction.

Per basin:
  mosaic 10 m tiles -> reproject to equal-area -> fill depressions ->
  flow direction -> flow accumulation -> stream network -> HAND, TWI, SPI

Outputs are basin-wide rasters. Phase 5 step 2 samples them onto each event's
20 m grid.

Uses the NATIVE 10 m product, never a degraded rung. Written to a directory
named for routing so it can never be mistaken for an ablation input
(NEXT_STEPS section 3c).

Depressions are FILLED rather than breached: desert basins contain genuine
closed depressions (playas), and breaching would carve false outlets through
them. Filling keeps them as sinks, which is what they physically are.

Output: data/basin_features/<huc8>/{facc,hand,twi,spi,slope,stream}.tif
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.warp import Resampling, calculate_default_transform, reproject

ROOT = Path(__file__).resolve().parent.parent
DEM10 = ROOT / "data/dem10_routing"
OUT = ROOT / "data/basin_features"
WORK = ROOT / "data/basin_work"
EQUAL_AREA = "EPSG:5070"       # metres, so slope and area are physical
RES = 10.0
STREAM_CELLS = 500             # ~0.05 km2 to initiate a channel


def basin_geoms() -> gpd.GeoDataFrame:
    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    strict = set(sets[sets["set"] == "STRICT"]["huc8"])
    work = ROOT / "data/raw/wbd/extracted"
    frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "name", "geometry"]]
              for r in ("13", "14", "15", "16", "18")]
    g = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
    g["huc8"] = g["huc8"].astype(str).str.zfill(8)
    return g[g["huc8"].isin(strict)].copy()


def mosaic_for(geom, dest: Path) -> Path | None:
    """Mosaic every 10 m tile intersecting a basin, in equal-area metres.

    A buffer is added because flow into the basin edge originates outside it;
    clipping exactly to the boundary would truncate contributing area on the
    rim, which is the fault this whole step exists to avoid.
    """
    if dest.exists():
        return dest
    b = geom.buffer(0.15).bounds
    tiles = []
    for t in sorted(DEM10.glob("*.tif")):
        with rasterio.open(t) as s:
            tb = s.bounds
            if tb.left < b[2] and tb.right > b[0] and tb.bottom < b[3] and tb.top > b[1]:
                tiles.append(t)
    if not tiles:
        return None
    srcs = [rasterio.open(t) for t in tiles]
    try:
        arr, tf = merge(srcs, bounds=b)
        prof = srcs[0].profile
    finally:
        for s in srcs:
            s.close()
    prof.update(height=arr.shape[1], width=arr.shape[2], transform=tf,
                count=1, compress="deflate")

    dst_crs = EQUAL_AREA
    dt, dw, dh = calculate_default_transform(
        prof["crs"], dst_crs, prof["width"], prof["height"],
        *rasterio.transform.array_bounds(prof["height"], prof["width"], tf),
        resolution=RES)
    out = np.full((dh, dw), np.nan, dtype="float32")
    reproject(source=arr[0], destination=out,
              src_transform=tf, src_crs=prof["crs"],
              dst_transform=dt, dst_crs=dst_crs,
              src_nodata=prof.get("nodata"), dst_nodata=np.nan,
              resampling=Resampling.bilinear)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(dest, "w", driver="GTiff", height=dh, width=dw, count=1,
                       dtype="float32", crs=dst_crs, transform=dt,
                       nodata=np.nan, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512) as d:
        d.write(out, 1)
    return dest


def route(dem_path: Path, outdir: Path, wbt) -> dict:
    """Fill, route, accumulate, then derive the wetness indices.

    The intermediates are deleted in a finally block. They were previously
    removed only on the success path, so when this crashed - which it did, on a
    full disk - it left ~3 GB per basin of dead weight behind, making the
    original problem worse.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    filled = outdir / "_filled.tif"
    fdir = outdir / "_fdir.tif"
    facc = outdir / "facc.tif"
    slope = outdir / "slope.tif"
    stream = outdir / "stream.tif"
    hand = outdir / "hand.tif"

    try:
        return _route_inner(dem_path, outdir, wbt, filled, fdir, facc,
                            slope, stream, hand)
    finally:
        for tmp in (filled, fdir):
            tmp.unlink(missing_ok=True)


# Compression is OFF.
#
# It caused two routing failures - RasterBlockError, from a profile inheriting a
# strip-organised source's blockysize while demanding tiled=True - and each one
# aborted the whole chain and took about an hour to trace. With 122 GB free
# (after deleting phase and look-vector layers no code reads) it saves disk we
# do not need and risks the one thing that has repeatedly broken the pipeline.
#
# Raw output: ~15 GB per basin, 3 basins left, against 122 GB free.
COMPRESS = False


def _shrink(path: Path, as_mask: bool = False) -> None:
    """Compress one raster in place. Disabled - see COMPRESS above.

    WhiteboxTools writes uncompressed, and HAND arrives as float64. Deferring
    this to the end of a basin meant six raw layers coexisted - ~15 GB for a
    900 M cell basin - which filled the disk twice. Doing it per layer caps the
    peak at one.
    """
    if not COMPRESS or not path.exists():
        return
    try:
        with rasterio.open(path) as s:
            if s.profile.get("compress"):
                return
            arr = s.read(1)
            prof = s.profile
    except Exception:
        return
    if as_mask:
        arr = (np.nan_to_num(arr, nan=0) > 0).astype("uint8")
        prof.update(dtype="uint8", nodata=None)
    else:
        arr = arr.astype("float32")
        prof.update(dtype="float32")
    prof.pop("blockysize", None)
    prof.pop("blockxsize", None)
    prof.update(compress="deflate", predictor=2, tiled=True,
                blockxsize=512, blockysize=512, zlevel=6)
    tmp = path.with_suffix(".tmp.tif")
    try:
        with rasterio.open(tmp, "w", **prof) as d:
            d.write(arr, 1)
        tmp.replace(path)
    except Exception:
        tmp.unlink(missing_ok=True)


def _route_inner(dem_path, outdir, wbt, filled, fdir, facc, slope, stream, hand):
    wbt.fill_depressions(str(dem_path), str(filled))
    wbt.d8_pointer(str(filled), str(fdir))
    wbt.d8_flow_accumulation(str(filled), str(facc), out_type="cells")
    _shrink(facc)
    wbt.slope(str(filled), str(slope), units="degrees")
    _shrink(slope)
    wbt.extract_streams(str(facc), str(stream), threshold=STREAM_CELLS)
    # HAND needs the raw stream mask, so compress it after HAND is derived.
    wbt.elevation_above_stream(str(filled), str(stream), str(hand))
    _shrink(hand)
    _shrink(stream, as_mask=True)
    # fdir is only needed by the calls above; release it early rather than
    # holding ~1 GB until the finally block.
    fdir.unlink(missing_ok=True)

    with rasterio.open(facc) as s:
        a = s.read(1).astype("float64")
        prof = s.profile
    with rasterio.open(slope) as s:
        sl = s.read(1).astype("float64")

    area = np.maximum(a, 1) * RES * RES          # m2 of contributing area
    tan_b = np.tan(np.radians(np.maximum(sl, 0.1)))
    twi = np.log(area / tan_b)
    spi = np.log(area * tan_b)

    # This profile came from facc. If facc is strip-organised its blockysize is
    # inherited, and GDAL rejects a tiled write whose block dims are not
    # multiples of 16 - which is exactly what aborted routing at 8/11 basins.
    # Uncompressed and untiled: nothing here needs to be clever, and tiling on
    # an inherited profile is exactly what broke routing twice.
    prof.pop("blockysize", None)
    prof.pop("blockxsize", None)
    prof.update(dtype="float32", nodata=np.nan, tiled=False)
    prof.pop("compress", None)
    for name, arr in (("twi", twi), ("spi", spi)):
        with rasterio.open(outdir / f"{name}.tif", "w", **prof) as d:
            d.write(arr.astype("float32"), 1)

    # Recompress whitebox's outputs. It writes uncompressed, and at 10 m over a
    # buffered basin that is ~750 MB per layer - HAND arrives as float64 (8 bytes
    # for a height in metres) and the stream network as float32 (4 bytes for a
    # 0/1 mask). Left raw, 11 basins came to ~60 GB and filled the disk.
    for name, dtype in ((("facc", "float32"), ("slope", "float32"),
                         ("hand", "float32"), ("stream", "uint8"))
                        if COMPRESS else ()):
        f = outdir / f"{name}.tif"
        if not f.exists():
            continue
        with rasterio.open(f) as src:
            arr = src.read(1)
            prof2 = src.profile
        if dtype == "uint8":
            arr = (np.nan_to_num(arr, nan=0) > 0).astype("uint8")
            prof2.update(dtype="uint8", nodata=None)
        else:
            arr = arr.astype("float32")
            prof2.update(dtype="float32")
        # Drop inherited strip keys before tiling - GDAL rejects a tiled write
        # whose block dims came from a stripped source.
        prof2.pop("blockysize", None); prof2.pop("blockxsize", None)
        prof2.update(compress="deflate", predictor=2, tiled=True,
                     blockxsize=512, blockysize=512, zlevel=6)
        tmp = f.with_suffix(".tmp.tif")
        with rasterio.open(tmp, "w", **prof2) as d:
            d.write(arr, 1)
        tmp.replace(f)

    with rasterio.open(hand) as s:
        h = s.read(1).astype("float64")
        h = h[np.isfinite(h)]
    return {"cells": int(a.size),
            "max_facc_cells": int(np.nanmax(a)),
            "max_drainage_km2": round(float(np.nanmax(a)) * RES * RES / 1e6, 1),
            "hand_median_m": round(float(np.median(h)), 2) if h.size else None,
            "stream_cells": int(np.nansum(rasterio.open(stream).read(1) > 0))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None, help="single huc8")
    args = ap.parse_args()

    from whitebox import WhiteboxTools
    wbt = WhiteboxTools()
    wbt.verbose = False

    basins = basin_geoms()
    if args.only:
        basins = basins[basins["huc8"] == args.only]
    if args.limit:
        basins = basins.head(args.limit)
    print(f"{len(basins)} basins\n", flush=True)

    state_path = ROOT / "data/basin_routing_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    total = len(basins)
    for bi, r in enumerate(basins.itertuples(index=False), 1):
        if state.get(r.huc8, {}).get("done"):
            print(f"  [{bi}/{total}] {r.huc8} {r.name}: already done", flush=True)
            continue
        print(f"  [{bi}/{total}] {r.huc8} {r.name}: mosaicking...", flush=True)
        WORK.mkdir(parents=True, exist_ok=True)
        dem = mosaic_for(r.geometry, WORK / f"{r.huc8}_dem10.tif")
        if dem is None:
            print(f"    no 10 m tiles cover this basin yet")
            continue
        with rasterio.open(dem) as s:
            print(f"    {s.width}x{s.height} px "
                  f"({s.width*s.height/1e6:.0f} M cells) — routing...", flush=True)
        try:
            stats = route(dem, OUT / r.huc8, wbt)
        except Exception as e:
            print(f"    FAILED: {type(e).__name__}: {e}", flush=True)
            state[r.huc8] = {"done": False, "error": str(e)[:200]}
            state_path.write_text(json.dumps(state, indent=1))
            continue
        state[r.huc8] = {"done": True, **stats}
        state_path.write_text(json.dumps(state, indent=1))
        print(f"    [{bi}/{total}] done — max drainage "
              f"{stats['max_drainage_km2']} km2, "
              f"HAND median {stats['hand_median_m']} m", flush=True)
        dem.unlink(missing_ok=True)     # mosaic is reproducible; the outputs are what matter

    done = sum(1 for v in state.values() if v.get("done"))
    print(f"\nrouted {done} basins")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
