#!/usr/bin/env python3
"""
Catchment-routed rainfall. The quantity that actually drives a flood.

THE PROBLEM THIS FIXES
    The rain features in the stacks are rainfall AT THE POINT where a person
    filed a storm report. A wash floods because of rain that fell tens of
    kilometres upstream and ran downhill. Point rainfall is the wrong quantity,
    and it showed: rain correlated NEGATIVELY with flooding (-0.14), physically
    backwards, while every terrain feature carried the correct sign.

WHAT THIS COMPUTES, per pixel, over everything draining INTO it:
    *_upstream_vol_m3     water arriving  = sum(depth x cell area)
    *_upstream_depth_mm   that volume / contributing area (catchment mean depth)
    rain_cover_frac       fraction of the catchment the radar actually saw
    runoff_1h_upstream_*  infiltration-excess runoff, routed

INFILTRATION EXCESS IS COMPUTED BEFORE ROUTING, NOT AFTER
    Runoff is generated where rain intensity beats what the ground can absorb:

        runoff_1h = max(0, rain_1h - ksat)      [mm in that hour]

    then that runoff is routed. Subtracting soil capacity AFTER routing would be
    meaningless - POLARIS ksat is ~6 cm/hr here, so a 24 h capacity is ~1400 mm,
    larger than any storm on record, and the "excess" would just be -ksat with
    the rain signal erased. Infiltration excess is the mechanism behind desert
    flash floods, so getting its order of operations right is the whole point.

WHY IT IS AFFORDABLE
    Routing at 100 m, not 10 m. Rain is a 1 km product, so 100 m already
    oversamples it, and a basin drops from 200 M cells to 2 M. Flow direction is
    built once per basin and cached.

Output: data/rain_routed/<id>.tif
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge
from rasterio.warp import reproject, transform_bounds

warnings.filterwarnings("ignore")

# pysheds 0.5 still calls np.in1d, which NumPy 2 removed. np.isin is the exact
# replacement (same semantics, ravels its first argument the same way). Patched
# here rather than in site-packages so the environment stays reproducible.
if not hasattr(np, "in1d"):
    np.in1d = np.isin

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"
DEM10 = ROOT / "data/dem10_routing"
OUT = ROOT / "data/rain_routed"
DEM100 = ROOT / "data/basin_dem100"
FDIR100 = ROOT / "data/basin_fdir100"
WORK_RES = 100.0
DECIM = 10                      # basin_features rasters are 10 m
ESRI_DIRMAP = (64, 128, 1, 2, 4, 8, 16, 32)


def log(m: str) -> None:
    print(m, flush=True)


def basin_grid(huc8: str) -> dict | None:
    """The 100 m grid for a basin: an exact 10x decimation of its facc raster.

    Deriving it from facc rather than from fresh bounds keeps this grid aligned
    with the basin features already computed, so nothing has to be re-registered.
    """
    src = BF / huc8 / "facc.tif"
    if not src.exists():
        return None
    with rasterio.open(src) as s:
        t = s.transform
        return {"crs": s.crs,
                "transform": rasterio.Affine(t.a * DECIM, t.b, t.c,
                                             t.d, t.e * DECIM, t.f),
                "shape": (s.height // DECIM, s.width // DECIM)}


def warp_onto(path: str, grid: dict, src_nodata=None,
              resampling=Resampling.bilinear) -> np.ndarray:
    out = np.full(grid["shape"], np.nan, dtype="float32")
    with rasterio.open(path) as s:
        reproject(source=rasterio.band(s, 1), destination=out,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=s.nodata if src_nodata is None else src_nodata,
                  dst_nodata=np.nan, resampling=resampling)
    return out


def grid_bounds(grid: dict, dst_crs) -> tuple:
    h, w = grid["shape"]
    b = rasterio.transform.array_bounds(h, w, grid["transform"])
    # array_bounds returns (left, bottom, right, top) in the grid's own CRS
    return transform_bounds(grid["crs"], dst_crs, *b, densify_pts=21)


def build_dem100(huc8: str, grid: dict) -> Path | None:
    """Mosaic only the 10 m tiles that actually overlap this basin's grid.

    The tile directory holds 36 tiles covering the whole study area. Warping all
    36 onto every basin would be 36x the work and would pull in terrain from
    other basins.
    """
    DEM100.mkdir(parents=True, exist_ok=True)
    dest = DEM100 / f"{huc8}.tif"
    if dest.exists():
        return dest
    tiles = []
    for t in sorted(DEM10.glob("*.tif")):
        with rasterio.open(t) as s:
            gb = grid_bounds(grid, s.crs)
            tb = s.bounds
            if tb.left < gb[2] and tb.right > gb[0] and tb.bottom < gb[3] and tb.top > gb[1]:
                tiles.append(t)
    if not tiles:
        return None
    srcs = [rasterio.open(t) for t in tiles]
    try:
        arr, tf = merge(srcs)
        src_crs, src_nodata = srcs[0].crs, srcs[0].nodata
    finally:
        for s in srcs:
            s.close()
    dem = np.full(grid["shape"], np.nan, dtype="float32")
    reproject(source=arr[0], destination=dem,
              src_transform=tf, src_crs=src_crs,
              dst_transform=grid["transform"], dst_crs=grid["crs"],
              src_nodata=src_nodata, dst_nodata=np.nan,
              resampling=Resampling.average)
    if not np.isfinite(dem).any():
        return None
    with rasterio.open(dest, "w", driver="GTiff", height=grid["shape"][0],
                       width=grid["shape"][1], count=1, dtype="float32",
                       crs=grid["crs"], transform=grid["transform"],
                       nodata=np.nan, compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512) as d:
        d.write(dem, 1)
    log(f"    mosaicked {len(tiles)} DEM tiles -> {dest.name} "
        f"({100*np.isfinite(dem).mean():.0f}% valid)")
    return dest


def build_fdir(huc8: str, grid: dict) -> Path | None:
    """D8 flow direction at 100 m, cached per basin.

    Depressions are FILLED, not breached, for the same reason as the 10 m
    routing: desert basins hold genuine closed depressions (playas), and
    breaching would carve false outlets through them.
    """
    FDIR100.mkdir(parents=True, exist_ok=True)
    dest = FDIR100 / f"{huc8}.tif"
    if dest.exists():
        return dest
    dem_path = build_dem100(huc8, grid)
    if dem_path is None:
        return None

    from pysheds.grid import Grid
    g = Grid.from_raster(str(dem_path))
    dem = g.read_raster(str(dem_path))
    filled = g.fill_depressions(dem)
    flats = g.resolve_flats(filled)
    fdir = g.flowdir(flats, dirmap=ESRI_DIRMAP)
    with rasterio.open(dest, "w", driver="GTiff", height=grid["shape"][0],
                       width=grid["shape"][1], count=1, dtype="int32",
                       crs=grid["crs"], transform=grid["transform"], nodata=0,
                       compress="deflate", tiled=True,
                       blockxsize=512, blockysize=512) as d:
        d.write(np.asarray(fdir).astype("int32"), 1)
    log(f"    flow direction built -> {dest.name}")
    return dest


def basin_ksat(huc8: str, grid: dict) -> np.ndarray | None:
    """POLARIS saturated conductivity on the basin grid, in mm/hr.

    Stored as log10(cm/hr) with -9999 for nodata. Every overlapping tile is
    merged, as in build_features.soil_bands - taking a single tile would give
    one basin's soil to all eleven.
    """
    d = ROOT / "data/soil/ksat"
    if not d.exists():
        return None
    acc = np.full(grid["shape"], np.nan, dtype="float32")
    for tif in sorted(d.glob("*.tif")):
        try:
            with rasterio.open(tif) as s:
                gb = grid_bounds(grid, s.crs)
                tb = s.bounds
                if not (tb.left < gb[2] and tb.right > gb[0]
                        and tb.bottom < gb[3] and tb.top > gb[1]):
                    continue
            a = warp_onto(str(tif), grid, src_nodata=-9999.0)
        except Exception:
            continue
        acc = np.where(np.isfinite(acc), acc, a)
    if not np.isfinite(acc).any():
        return None
    return (10.0 ** acc) * 10.0            # log10(cm/hr) -> mm/hr


def accumulate(g, fdir, field: np.ndarray, vf) -> np.ndarray:
    from pysheds.view import Raster
    w = Raster(field.astype("float64"), viewfinder=vf)
    return np.asarray(g.accumulation(fdir, weights=w, dirmap=ESRI_DIRMAP),
                      dtype="float64")


def route_event(grid: dict, fdir_path: Path, rain_files: dict,
                ksat: np.ndarray | None) -> dict | None:
    """Accumulate this event's rainfall, and its runoff, downstream."""
    from pysheds.grid import Grid
    from pysheds.view import Raster, ViewFinder

    with rasterio.open(fdir_path) as s:
        fdir_arr = s.read(1)
    vf = ViewFinder(affine=grid["transform"], shape=grid["shape"],
                    crs=grid["crs"], nodata=0)
    fdir = Raster(fdir_arr, viewfinder=vf)
    g = Grid.from_raster(fdir)

    cell_m2 = WORK_RES * WORK_RES
    area = accumulate(g, fdir, np.full(grid["shape"], cell_m2, "float64"), vf)
    ok_area = area > 0

    fields: dict[str, np.ndarray] = {}      # depths in mm -> volume + mean depth
    means: dict[str, np.ndarray] = {}       # dimensionless -> catchment mean only
    cover = None
    for name, path in rain_files.items():
        try:
            rain = warp_onto(path, grid)
        except Exception:
            continue
        # MRMS: -3 no radar coverage, -1 missing. Never rainfall.
        rain[rain < 0] = np.nan
        seen = np.isfinite(rain).astype("float64")
        rain = np.nan_to_num(rain, nan=0.0)
        fields[name] = rain
        if cover is None:
            cover = seen

    if not fields:
        return None

    out: dict[str, np.ndarray] = {}
    # Radar coverage of the catchment. Without it, an upstream area the radar
    # never saw is indistinguishable from one where no rain fell.
    cov = accumulate(g, fdir, cover * cell_m2, vf)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["rain_cover_frac"] = np.where(ok_area, cov / area, np.nan).astype("float32")
    out["catchment_area_km2"] = (area / 1e6).astype("float32")

    # Infiltration excess, generated locally, then routed.
    if ksat is not None and "rain_1h" in fields:
        # Gaps in POLARIS get the basin median, NOT zero. Zero would mean "this
        # ground absorbs nothing", turning missing soil data into maximum runoff -
        # the one direction a gap must never push a flood prediction.
        k = np.where(np.isfinite(ksat), ksat, np.nanmedian(ksat))
        k = np.maximum(k, 0.1)
        fields["runoff_1h"] = np.maximum(0.0, fields["rain_1h"] - k)
        # The hard threshold is nearly always unmet: POLARIS ksat here is ~50
        # mm/hr (sand) while hourly QPE peaks near 30, so runoff_1h is ~0 over
        # 99.9% of the basin. That is a property of the soil product, not a bug.
        # The ratio keeps the same physics without a hard cutoff, so the model
        # can find its own threshold instead of inheriting POLARIS's.
        means["rain_1h_over_ksat"] = fields["rain_1h"] / k

    for name, field in fields.items():
        vol = accumulate(g, fdir, field * cell_m2 / 1000.0, vf)     # m3
        with np.errstate(divide="ignore", invalid="ignore"):
            depth = np.where(ok_area, vol / area * 1000.0, np.nan)  # mm
        out[f"{name}_upstream_vol_m3"] = vol.astype("float32")
        out[f"{name}_upstream_depth_mm"] = depth.astype("float32")

    for name, field in means.items():
        tot = accumulate(g, fdir, field * cell_m2, vf)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"{name}_upstream_mean"] = np.where(
                ok_area, tot / area, np.nan).astype("float32")
    return out


def rain_paths(date: str) -> dict:
    d = pd.to_datetime(date).floor("h")
    day = ROOT / "data/rain" / d.strftime("%Y%m%d")
    want = {"rain_1h": f"RadarOnly_QPE_01H_00.00_{d:%Y%m%d-%H0000}.grib2.gz",
            "rain_24h": f"RadarOnly_QPE_24H_00.00_{d:%Y%m%d-%H0000}.grib2.gz"}
    return {n: f"/vsigzip/{day / fn}" for n, fn in want.items() if (day / fn).exists()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default="")
    ap.add_argument("--negatives", action="store_true")
    ap.add_argument("--out", default="data/rain_routed")
    args = ap.parse_args()

    if args.negatives:
        recs = [json.loads(l) for l in
                (ROOT / "data/negative_bursts.jsonl").read_text().splitlines() if l.strip()]
        items = [{"id": r["NEG_ID"], "huc8": str(r["huc8"]).zfill(8),
                  "date": r["quiet_date"]} for r in recs]
    else:
        ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
        items = [{"id": str(int(r.EVENT_ID)), "huc8": str(r.huc8).zfill(8),
                  "date": str(r.BEGIN_UTC)} for r in ev.itertuples(index=False)]
    if args.only:
        items = [i for i in items if i["id"] == args.only]
    if args.limit:
        items = items[: args.limit]

    outdir = ROOT / args.out
    outdir.mkdir(parents=True, exist_ok=True)
    log(f"{len(items)} items to route -> {outdir.relative_to(ROOT)}\n")

    grids: dict[str, dict | None] = {}
    fdirs: dict[str, Path | None] = {}
    ksats: dict[str, np.ndarray | None] = {}
    done = skipped = 0
    reasons: dict[str, int] = {}

    def skip(why: str) -> None:
        nonlocal skipped
        skipped += 1
        reasons[why] = reasons.get(why, 0) + 1

    for i, it in enumerate(items, 1):
        dest = outdir / f"{it['id']}.tif"
        if dest.exists():
            done += 1
            continue
        h = it["huc8"]
        if h not in grids:
            grids[h] = basin_grid(h)
            if grids[h] is None:
                fdirs[h], ksats[h] = None, None
                log(f"  basin {h}: no facc.tif, cannot route")
            else:
                log(f"  basin {h}: {grids[h]['shape'][0]}x{grids[h]['shape'][1]} "
                    f"@100 m, preparing...")
                fdirs[h] = build_fdir(h, grids[h])
                ksats[h] = basin_ksat(h, grids[h])
        grid, fdir = grids[h], fdirs.get(h)
        if grid is None or fdir is None:
            skip("no routing grid")
            continue

        rf = rain_paths(it["date"])
        if not rf:
            skip("no rain file for that hour")
            continue
        bands = route_event(grid, fdir, rf, ksats.get(h))
        if not bands:
            skip("rain would not read")
            continue

        keys = sorted(bands)
        with rasterio.open(dest, "w", driver="GTiff", height=grid["shape"][0],
                           width=grid["shape"][1], count=len(keys),
                           dtype="float32", crs=grid["crs"],
                           transform=grid["transform"], nodata=np.nan,
                           compress="deflate", tiled=True,
                           blockxsize=512, blockysize=512) as o:
            for bi, k in enumerate(keys, 1):
                o.write(bands[k], bi)
                o.set_band_description(bi, k)
        done += 1
        if done % 20 == 0:
            log(f"  [{i}/{len(items)}] routed {done}, skipped {skipped}")

    log(f"\nrouted {done}, skipped {skipped}")
    for k, v in sorted(reasons.items(), key=lambda x: -x[1]):
        log(f"  {v:>4}  {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
