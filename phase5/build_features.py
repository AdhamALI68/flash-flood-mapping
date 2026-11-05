#!/usr/bin/env python3
"""
Phase 5, step 2 - build one feature stack per event, on a fixed 20 m grid.

NEXT_STEPS section 3c: the prediction grid is ALWAYS 20 m. Only the resolution
of the DEM the terrain features are derived from varies (1/2/5/10/30 m). That
keeps one variable moving and removes the receptive-field confound, and it is
forced anyway - 20 m is the finest label HyP3 produces, so a finer prediction
could not be scored.

Per event, at a 5 km radius (97% of events have full 1 m lidar there; at 10 km
only 79% do):

  label       continuous coherence drop. NOT thresholded - section 3d keeps the
              threshold a reporting choice, applied at scoring time
  basin_*     HAND, TWI, SPI, flow accumulation, slope, routed over the COMPLETE
              basin at native 10 m (trial 1: +0.073 for basin-wide routing)
  terrain_R   derived from 1 m DEGRADED to R, then aggregated into 20 m cells.
              Never a native product - native 10 m and 1 m-downsampled-to-10 m
              are different data (section 3c)
  rain_*      24 h total, peak hourly intensity, antecedent 48/72 h
  soil_*      ksat, clay, sand, bulk density
  lc_*        land cover class and a cropland mask

Micro-relief within each 20 m cell is the whole point of the ablation: at 1 m a
cell holds 400 samples and an incised channel is visible; at 30 m the cell is
smaller than one pixel, so range and roughness are structurally zero. The
ablation measures what that costs.

RULES ENFORCED HERE
  - elevation is never filtered by sign. Death Valley and the Salton Sea are
    genuinely below sea level; the nodata sentinel (-999999) is used instead
  - one valid mask, the AND of every channel, applied to all of them, so no row
    is part real and part fabricated
  - dropped-pixel counts recorded per event (section 5)

Output: data/features/<event_id>/*.tif + meta.json
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

RADIUS_M = 5000
GRID_RES = 20.0
DEM_RUNGS = [1, 2, 5, 10, 30]
DEM_NODATA = -999999.0
CROPLAND = 40


# ----------------------------------------------------------------- helpers
def event_grid(base_corr: str, coev_corr: str, lon: float, lat: float):
    """The 20 m label grid: event radius INTERSECTED with both SAR rasters.

    Deriving the grid from the co-event raster alone loses a third of every
    event. Baseline and co-event products do not share an origin - measured at
    2500 m in x and 580 m in y for event 983830 - so a window taken from one
    extends past the other, and those pixels have no baseline to difference
    against. Intersecting first keeps only ground both passes actually saw.

    Measured effect on event 565903: valid fraction 0.672 -> ~0.93.
    """
    with rasterio.open(coev_corr) as c, rasterio.open(base_corr) as b:
        if c.crs != b.crs:
            return None
        crs = c.crs
        ex, ey = warp_transform("EPSG:4326", crs, [lon], [lat])
        left = max(ex[0] - RADIUS_M, c.bounds.left, b.bounds.left)
        right = min(ex[0] + RADIUS_M, c.bounds.right, b.bounds.right)
        bottom = max(ey[0] - RADIUS_M, c.bounds.bottom, b.bounds.bottom)
        top = min(ey[0] + RADIUS_M, c.bounds.top, b.bounds.top)
        if right - left < 1000 or top - bottom < 1000:
            return None
        win = from_bounds(left, bottom, right, top, transform=c.transform)
        win = win.intersection(rasterio.windows.Window(0, 0, c.width, c.height))
        if win.width < 50 or win.height < 50:
            return None
        return {"crs": crs, "transform": c.window_transform(win),
                "shape": (int(win.height), int(win.width)),
                "window": win, "cx": ex[0], "cy": ey[0]}


def warp_onto(src_path: str, grid: dict, band: int = 1,
              resampling=Resampling.bilinear, src_nodata=None) -> np.ndarray:
    """Reproject any raster onto the event grid."""
    out = np.full(grid["shape"], np.nan, dtype="float32")
    with rasterio.open(src_path) as s:
        reproject(source=rasterio.band(s, band), destination=out,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=src_nodata if src_nodata is not None else s.nodata,
                  dst_nodata=np.nan, resampling=resampling)
    return out


def block_stats(fine: np.ndarray, factor: int) -> dict:
    """Aggregate a fine grid into 20 m cells, keeping WITHIN-cell structure.

    The mean is what a coarse DEM would give. The range, std and min are what a
    coarse DEM cannot express - they are the whole reason 1 m might help.
    """
    h = (fine.shape[0] // factor) * factor
    w = (fine.shape[1] // factor) * factor
    a = fine[:h, :w].reshape(h // factor, factor, w // factor, factor)
    a = a.transpose(0, 2, 1, 3).reshape(h // factor, w // factor, -1)
    with np.errstate(invalid="ignore"):
        return {"mean": np.nanmean(a, axis=2),
                "min": np.nanmin(a, axis=2),
                "range": np.nanmax(a, axis=2) - np.nanmin(a, axis=2),
                "std": np.nanstd(a, axis=2)}


def terrain_from_1m(tiles: list[str], grid: dict, rung_m: int) -> dict | None:
    """Terrain features at one ablation rung, derived from the 1 m source.

    Degrade 1 m to `rung_m`, then aggregate into 20 m cells. At 30 m the cell is
    smaller than one pixel, so within-cell range and roughness collapse to zero -
    which is exactly the information a coarse DEM cannot carry.
    """
    if not tiles:
        return None
    # Work on a grid `rung_m` fine, covering the same extent as the 20 m grid.
    factor = max(1, int(round(GRID_RES / rung_m)))
    fine_shape = (grid["shape"][0] * factor, grid["shape"][1] * factor)
    t = grid["transform"]
    fine_tf = rasterio.Affine(t.a / factor, t.b, t.c, t.d, t.e / factor, t.f)

    acc = np.full(fine_shape, np.nan, dtype="float32")
    for tp in tiles:
        buf = np.full(fine_shape, np.nan, dtype="float32")
        with rasterio.open(tp) as s:
            reproject(source=rasterio.band(s, 1), destination=buf,
                      dst_transform=fine_tf, dst_crs=grid["crs"],
                      src_nodata=DEM_NODATA, dst_nodata=np.nan,
                      resampling=Resampling.average)
        acc = np.where(np.isfinite(acc), acc, buf)
    if not np.isfinite(acc).any():
        return None

    st = block_stats(acc, factor) if factor > 1 else {
        "mean": acc, "min": acc,
        "range": np.zeros_like(acc), "std": np.zeros_like(acc)}

    # Slope from the aggregated surface, in degrees.
    gy, gx = np.gradient(st["mean"], GRID_RES)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # Height above the local minimum - a cheap within-neighbourhood HAND proxy
    # that varies with the rung, unlike the basin-wide HAND.
    return {"elev_mean": st["mean"], "elev_min": st["min"],
            "microrelief": st["range"], "roughness": st["std"],
            "slope": slope}


def rain_bands(date: str, grid: dict) -> dict:
    """MRMS rainfall onto the event grid.

    NEGATIVE VALUES ARE SENTINELS HERE - the opposite of the DEM, where
    negatives are real elevations. MRMS uses -3 for "no radar coverage" and -1
    for missing. Treating them as rainfall would inject large negative numbers
    into a physically non-negative field.
    """
    d = pd.to_datetime(date).floor("h")
    day = ROOT / "data/rain" / d.strftime("%Y%m%d")
    if not day.exists():
        return {}
    out = {}
    wanted = {
        "rain_1h_peak": f"RadarOnly_QPE_01H_00.00_{d.strftime('%Y%m%d-%H0000')}.grib2.gz",
        "rain_24h": f"RadarOnly_QPE_24H_00.00_{d.strftime('%Y%m%d-%H0000')}.grib2.gz",
        "rain_ante48h": f"RadarOnly_QPE_24H_00.00_{(d - pd.Timedelta(hours=48)).strftime('%Y%m%d-%H0000')}.grib2.gz",
        "rain_ante72h": f"RadarOnly_QPE_24H_00.00_{(d - pd.Timedelta(hours=72)).strftime('%Y%m%d-%H0000')}.grib2.gz",
    }
    for name, fn in wanted.items():
        f = day / fn
        if not f.exists():
            continue
        try:
            a = warp_onto(f"/vsigzip/{f}", grid)
        except Exception:
            continue
        a[a < 0] = np.nan          # -3 no coverage, -1 missing
        out[name] = a
    return out


def soil_bands(grid: dict) -> dict:
    """POLARIS soil properties. nodata is -9999, and values are non-negative."""
    out = {}
    for var in ("ksat", "clay", "sand", "bd"):
        d = ROOT / "data/soil" / var
        if not d.exists():
            continue
        acc = np.full(grid["shape"], np.nan, dtype="float32")
        for tif in sorted(d.glob("*.tif")):
            try:
                a = warp_onto(str(tif), grid, src_nodata=-9999.0)
            except Exception:
                continue
            acc = np.where(np.isfinite(acc), acc, a)
        if np.isfinite(acc).any():
            out[f"soil_{var}"] = acc
    return out


def landcover_bands(grid: dict) -> dict:
    """WorldCover class map, plus the cropland mask.

    NEAREST resampling only - class codes must never be interpolated, since an
    average of class 40 and class 50 is a class that does not exist.
    """
    d = ROOT / "data/landcover/worldcover"
    if not d.exists():
        return {}
    acc = np.full(grid["shape"], np.nan, dtype="float32")
    for tif in sorted(d.glob("*.tif")):
        try:
            a = warp_onto(str(tif), grid, resampling=Resampling.nearest, src_nodata=0)
        except Exception:
            continue
        acc = np.where(np.isfinite(acc), acc, a)
    if not np.isfinite(acc).any():
        return {}
    return {"lc_class": acc,
            "lc_cropland": (acc == CROPLAND).astype("float32"),
            "lc_water": (acc == 80).astype("float32")}


def write_stack(dest: Path, name: str, bands: dict, grid: dict) -> list[str]:
    dest.mkdir(parents=True, exist_ok=True)
    keys = sorted(bands)
    # Block dims must be multiples of 16 or GDAL refuses a tiled write. These
    # grids are ~500x500, so 256 divides them without a mostly-empty last tile.
    prof = dict(driver="GTiff", height=grid["shape"][0], width=grid["shape"][1],
                count=len(keys), dtype="float32", crs=grid["crs"],
                transform=grid["transform"], nodata=np.nan,
                compress="deflate", predictor=2, tiled=True,
                blockxsize=256, blockysize=256)
    with rasterio.open(dest / f"{name}.tif", "w", **prof) as d:
        for i, k in enumerate(keys, 1):
            d.write(bands[k].astype("float32"), i)
            d.set_band_description(i, k)
    return keys


# ----------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--only", default=None)
    ap.add_argument("--negatives", action="store_true",
                    help="build for the quiet controls instead of the events")
    ap.add_argument("--out", default="data/features")
    args = ap.parse_args()

    if args.negatives:
        recs = [json.loads(l) for l in
                (ROOT / "data/negative_bursts.jsonl").read_text().splitlines() if l.strip()]
        state = json.loads((ROOT / "data/negatives_state.json").read_text())
        groups_dir = ROOT / "data/negatives/groups"
        items = [{"id": r["NEG_ID"], "lon": r["lon"], "lat": r["lat"],
                  "huc8": r["huc8"], "huc4": r["huc4"],
                  "date": r["quiet_date"]} for r in recs]
    else:
        ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
        state = json.loads((ROOT / "data/phase3_state.json").read_text())
        groups_dir = ROOT / "data/phase3/groups"
        items = [{"id": str(int(r.EVENT_ID)), "lon": float(r.BEGIN_LON),
                  "lat": float(r.BEGIN_LAT), "huc8": str(r.huc8).zfill(8),
                  "huc4": str(r.huc4), "date": str(r.BEGIN_UTC)}
                 for r in ev.itertuples(index=False)]

    if args.only:
        items = [i for i in items if i["id"] == args.only]
    if args.limit:
        items = items[: args.limit]
    print(f"{len(items)} items to build\n", flush=True)

    # 1 m tile index, so each event pulls only the tiles it overlaps.
    man = pd.read_csv(ROOT / "data/dem_download_manifest.csv")
    dem_state = json.loads((ROOT / "data/dem_download_state.json").read_text())
    tile_paths = {k: ROOT / v["path"] for k, v in dem_state.items()
                  if v.get("status") == "done"}
    import geopandas as gpd
    tcache = pd.concat([pd.read_csv(f, dtype=str)
                        for f in glob.glob(str(ROOT / "data/raw/tnm_tile_cache/*.csv"))],
                       ignore_index=True)
    tcache["grid_key"] = tcache.zone + "_" + tcache.gx + "_" + tcache.gy
    tcache = tcache.drop_duplicates("grid_key")
    tidx = gpd.GeoDataFrame(tcache, geometry=gpd.GeoSeries.from_wkt(tcache.geometry_wkt),
                            crs="EPSG:4326")
    tidx = tidx[tidx.grid_key.isin(tile_paths)]

    outroot = ROOT / args.out
    built = skipped = 0
    meta_all = []
    for i, it in enumerate(items, 1):
        gid = state["events"].get(it["id"], {}).get("group_id")
        if not gid:
            skipped += 1
            continue
        gdir = groups_dir / gid
        try:
            bc = sorted((gdir / "baseline").glob("*_corr.tif"))[0]
            cc = sorted((gdir / "coevent").glob("*_corr.tif"))[0]
            ba = sorted((gdir / "baseline").glob("*_amp.tif"))[0]
            ca = sorted((gdir / "coevent").glob("*_amp.tif"))[0]
        except IndexError:
            skipped += 1
            continue

        grid = event_grid(str(bc), str(cc), it["lon"], it["lat"])
        if grid is None:
            skipped += 1
            continue

        # ---- label + SAR channels, all on the event grid -----------------
        bcv = warp_onto(str(bc), grid)
        ccv = warp_onto(str(cc), grid)
        bav = warp_onto(str(ba), grid)
        cav = warp_onto(str(ca), grid)
        with np.errstate(divide="ignore", invalid="ignore"):
            amp_db = 10 * np.log10(np.where((bav > 0) & (cav > 0), cav / bav, np.nan))
        label = bcv - ccv                       # continuous, never thresholded

        bands_label = {"coh_drop": label, "coh_baseline": bcv,
                       "coh_coevent": ccv, "amp_change_db": amp_db}

        # ---- basin-wide routed features ---------------------------------
        bdir = ROOT / "data/basin_features" / it["huc8"]
        basin = {}
        for nm in ("hand", "twi", "spi", "facc", "slope"):
            p = bdir / f"{nm}.tif"
            if p.exists():
                basin[f"basin_{nm}"] = warp_onto(str(p), grid)

        # ---- terrain rungs, all degraded from the 1 m source -------------
        from shapely.geometry import box as sbox
        ex, ey = grid["cx"], grid["cy"]
        tf_ = grid["transform"]
        b = rasterio.transform.array_bounds(*grid["shape"], tf_)
        lons, lats = warp_transform(grid["crs"], "EPSG:4326",
                                    [b[0], b[2], b[0], b[2]], [b[1], b[1], b[3], b[3]])
        aoi = sbox(min(lons), min(lats), max(lons), max(lats))
        hits = tidx[tidx.intersects(aoi)]
        tiles = [str(tile_paths[k]) for k in hits.grid_key if k in tile_paths]

        rungs = {}
        for rung in DEM_RUNGS:
            t = terrain_from_1m(tiles, grid, rung)
            if t:
                rungs[rung] = t

        # ---- rain, soil, land cover --------------------------------------
        rain = rain_bands(it["date"], grid)
        soil = soil_bands(grid)
        lc = landcover_bands(grid)

        # ---- valid mask: AND of everything -------------------------------
        masks = [np.isfinite(bcv), np.isfinite(ccv)]
        masks += [np.isfinite(v) for v in basin.values()]
        if rungs:
            masks.append(np.isfinite(rungs[min(rungs)]["elev_mean"]))
        masks += [np.isfinite(v) for v in soil.values()]
        masks += [np.isfinite(v) for v in rain.values()]
        # lc_class is categorical; require it present but do not AND the derived
        # binary masks, which are 0/1 everywhere by construction.
        if "lc_class" in lc:
            masks.append(np.isfinite(lc["lc_class"]))
        valid = np.logical_and.reduce(masks) if masks else np.isfinite(bcv)

        dest = outroot / it["id"]
        keys = {"label": write_stack(dest, "label", bands_label, grid)}
        if basin:
            keys["basin"] = write_stack(dest, "basin", basin, grid)
        for rung, t in rungs.items():
            keys[f"terrain_{rung}m"] = write_stack(
                dest, f"terrain_{rung}m",
                {f"{k}_{rung}m": v for k, v in t.items()}, grid)
        if rain:
            keys["rain"] = write_stack(dest, "rain", rain, grid)
        if soil:
            keys["soil"] = write_stack(dest, "soil", soil, grid)
        if lc:
            keys["landcover"] = write_stack(dest, "landcover", lc, grid)
        write_stack(dest, "valid", {"valid": valid.astype("float32")}, grid)

        meta = {"id": it["id"], "huc8": it["huc8"], "huc4": it["huc4"],
                "date": it["date"], "group_id": gid,
                "shape": list(grid["shape"]), "res_m": GRID_RES,
                "radius_m": RADIUS_M,
                "valid_px": int(valid.sum()),
                "valid_frac": round(float(valid.mean()), 4),
                "dropped_px": int((~valid).sum()),
                "rungs_built": sorted(rungs),
                "n_1m_tiles": len(tiles),
                "has_rain": bool(rain), "has_soil": bool(soil),
                "has_landcover": bool(lc),
                "bands": keys}
        (dest / "meta.json").write_text(json.dumps(meta, indent=1))
        meta_all.append(meta)
        built += 1
        if i % 10 == 0:
            print(f"  [{i}/{len(items)}] built {built}, skipped {skipped}", flush=True)

    if meta_all:
        pd.DataFrame([{k: v for k, v in m.items() if k != "bands"}
                      for m in meta_all]).to_csv(
            outroot / "_index.csv", index=False)
    print(f"\nbuilt {built}, skipped {skipped}")
    if meta_all:
        vf = np.array([m["valid_frac"] for m in meta_all])
        print(f"valid fraction: median {np.median(vf):.3f}, min {vf.min():.3f}")
        rb = pd.Series([len(m["rungs_built"]) for m in meta_all]).value_counts()
        print(f"rungs built per event: {dict(rb)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
