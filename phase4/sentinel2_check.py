#!/usr/bin/env python3
"""
Third independent validation: Sentinel-2 optical, before vs after.

NEXT_STEPS 4c asks for optical cross-checking. The naive version - "look for
water in the post-flood image" - almost never works here: Sentinel-2 revisits
every 5 days and a desert flash flood drains in hours, so the water has gone.
For the Gate B flood the nearest clear scene is six days late.

So this uses the same before/after logic coherence does, on a different
instrument. MNDWI (green - SWIR) / (green + SWIR) rises with surface water AND
with wet soil and freshly deposited sediment, which persist for days after the
water leaves. An increase between a pre-flood and a post-flood scene is
independent evidence that the surface changed.

Independent of the SAR in every way that matters: different satellite, different
physics (reflected sunlight, not radar backscatter), different failure modes
(clouds rather than decorrelation).

The test: does MNDWI increase MORE where coherence says the flood was?

Output: data/sentinel2_check.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

STAC = "https://earth-search.aws.element84.com/v1/search"
RADIUS_M = 5000
MAX_CLOUD = 25
PRE_DAYS = 45          # look this far back for a clear pre-flood scene
POST_DAYS = 12         # and this far forward after it
# SCL classes to exclude: 0 nodata, 1 saturated, 3 cloud shadow, 8/9 cloud,
# 10 thin cirrus. Keeping them would read cloud as water.
SCL_BAD = {0, 1, 3, 8, 9, 10}


def search(lon: float, lat: float, start: str, end: str) -> list[dict]:
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}/{end}", "limit": 20,
            "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}}}
    try:
        r = requests.post(STAC, json=body, timeout=120)
        r.raise_for_status()
        return r.json().get("features", [])
    except Exception:
        return []


def read_mndwi(feat: dict, grid: dict) -> np.ndarray | None:
    """MNDWI on the event grid, clouds masked out via the scene classifier."""
    a = feat["assets"]
    if not all(k in a for k in ("green", "swir16", "scl")):
        return None
    out = {}
    for key in ("green", "swir16", "scl"):
        buf = np.full(grid["shape"], np.nan, dtype="float32")
        try:
            with rasterio.open(a[key]["href"]) as s:
                reproject(source=rasterio.band(s, 1), destination=buf,
                          dst_transform=grid["transform"], dst_crs=grid["crs"],
                          src_nodata=0, dst_nodata=np.nan,
                          resampling=Resampling.nearest if key == "scl"
                          else Resampling.bilinear)
        except Exception:
            return None
        out[key] = buf
    # Cast only where finite - NaN -> int is undefined and warns.
    scl = out["scl"]
    finite = np.isfinite(scl)
    codes = np.zeros(scl.shape, dtype="int16")
    codes[finite] = scl[finite].astype("int16")
    good = finite & ~np.isin(codes, list(SCL_BAD))
    g, sw = out["green"], out["swir16"]
    with np.errstate(divide="ignore", invalid="ignore"):
        mndwi = (g - sw) / (g + sw)
    mndwi[~good] = np.nan
    return mndwi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=120)
    ap.add_argument("--threshold", type=float, default=0.15)
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    ev = ev.sample(min(args.sample, len(ev)), random_state=0)
    print(f"checking {len(ev)} events against Sentinel-2\n", flush=True)

    rows = []
    n_nopair = n_nogrid = 0
    for i, r in enumerate(ev.itertuples(index=False), 1):
        gid = st["events"].get(str(int(r.EVENT_ID)), {}).get("group_id")
        if not gid:
            continue
        d = ROOT / "data/phase3/groups" / gid
        try:
            bc = sorted((d / "baseline").glob("*_corr.tif"))[0]
            cc = sorted((d / "coevent").glob("*_corr.tif"))[0]
        except IndexError:
            continue

        with rasterio.open(cc) as s:
            ex, ey = warp_transform("EPSG:4326", s.crs, [r.BEGIN_LON], [r.BEGIN_LAT])
            win = from_bounds(ex[0] - RADIUS_M, ey[0] - RADIUS_M,
                              ex[0] + RADIUS_M, ey[0] + RADIUS_M, transform=s.transform)
            win = win.intersection(Window(0, 0, s.width, s.height))
            if win.width < 50 or win.height < 50:
                n_nogrid += 1
                continue
            grid = {"crs": s.crs, "transform": s.window_transform(win),
                    "shape": (int(win.height), int(win.width))}
            bounds = rasterio.windows.bounds(win, s.transform)

        t0 = r.BEGIN_UTC
        pre = search(r.BEGIN_LON, r.BEGIN_LAT,
                     (t0 - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%dT00:00:00Z"),
                     (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%dT00:00:00Z"))
        post = search(r.BEGIN_LON, r.BEGIN_LAT,
                      t0.strftime("%Y-%m-%dT00:00:00Z"),
                      (t0 + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%dT00:00:00Z"))
        if not pre or not post:
            n_nopair += 1
            continue
        # Clearest scene on each side, and the post scene closest to the flood.
        pre_f = min(pre, key=lambda f: f["properties"].get("eo:cloud_cover", 100))
        post_f = min(post, key=lambda f: (f["properties"].get("eo:cloud_cover", 100),
                                          f["properties"]["datetime"]))

        m_pre = read_mndwi(pre_f, grid)
        m_post = read_mndwi(post_f, grid)
        if m_pre is None or m_post is None:
            n_nopair += 1
            continue
        dm = m_post - m_pre

        try:
            (a, b), _, _ = read_aligned([str(bc), str(cc)], aoi=bounds)
        except Exception:
            continue
        if a.shape != dm.shape:
            h = min(a.shape[0], dm.shape[0]); w = min(a.shape[1], dm.shape[1])
            a, b, dm = a[:h, :w], b[:h, :w], dm[:h, :w]
        sar_ok = np.isfinite(a) & np.isfinite(b)
        drop = np.where(sar_ok, a - b, np.nan)
        ok = sar_ok & np.isfinite(dm)
        if ok.sum() < 2000:
            n_nopair += 1
            continue

        flood = ok & (drop > args.threshold)
        dry = ok & (drop <= args.threshold)
        if flood.sum() < 200 or dry.sum() < 200:
            continue

        rows.append({
            "EVENT_ID": int(r.EVENT_ID), "huc4": str(r.huc4),
            "pre_date": pre_f["properties"]["datetime"][:10],
            "post_date": post_f["properties"]["datetime"][:10],
            "post_lag_days": (pd.to_datetime(post_f["properties"]["datetime"][:10])
                              - t0.normalize()).days,
            "pre_cloud": round(pre_f["properties"].get("eo:cloud_cover", 0), 1),
            "post_cloud": round(post_f["properties"].get("eo:cloud_cover", 0), 1),
            "px": int(ok.sum()),
            # The test: MNDWI change where SAR says flooded, vs where it says dry
            "dmndwi_flood": float(np.nanmean(dm[flood])),
            "dmndwi_dry": float(np.nanmean(dm[dry])),
        })
        if i % 10 == 0:
            print(f"  [{i}/{len(ev)}] {len(rows)} usable, "
                  f"{n_nopair} without a clear pair", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/sentinel2_check.csv", index=False)
    print(f"\nusable events: {len(df)} of {len(ev)} "
          f"({n_nopair} lacked a cloud-free before/after pair)")
    if len(df) < 8:
        print("too few for a conclusion — Sentinel-2 coverage is the limit here")
        return 0

    df["contrast"] = df["dmndwi_flood"] - df["dmndwi_dry"]
    wins = int((df["contrast"] > 0).sum())
    import math
    k = min(wins, len(df) - wins)
    p = min(1.0, 2 * sum(math.comb(len(df), i) for i in range(k + 1)) / 2 ** len(df))
    print()
    print("=" * 68)
    print("SENTINEL-2 CROSS-CHECK — wetness change where SAR says flood")
    print("=" * 68)
    print(f"  mean MNDWI change, SAR-flood pixels : {df.dmndwi_flood.mean():+.5f}")
    print(f"  mean MNDWI change, SAR-dry pixels   : {df.dmndwi_dry.mean():+.5f}")
    print(f"  contrast (flood - dry)              : {df.contrast.mean():+.5f}")
    print(f"  events where flood pixels got wetter: {wins}/{len(df)} "
          f"({100*wins/len(df):.0f}%)")
    print(f"  sign test p                         : {p:.3g}")
    print(f"  median post-flood lag               : {df.post_lag_days.median():.0f} days")
    print()
    print("per region:")
    for reg, g in df.groupby("huc4"):
        if len(g) >= 3:
            print(f"  {reg}: {int((g.contrast>0).sum())}/{len(g)}")
    print()
    print("Positive contrast = the optical instrument sees extra wetness exactly")
    print("where the radar says the flood was. Two unrelated physics agreeing.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
