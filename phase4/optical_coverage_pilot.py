#!/usr/bin/env python3
"""
Pilot test, on a small sample, before touching the full 407 events that still
have no optical label. Question: does allowing more cloud cover in the whole
scene actually recover usable, trustworthy coverage over our small tile, or
does it just let in garbage.

WHY THIS IS WORTH TESTING
    optical_labels.py rejects a whole scene if its OVERALL cloud cover exceeds
    30%, even though every pixel is already individually cloud-masked (SCL
    band) before use. A scene that is 50% cloudy somewhere else in its frame
    can still be perfectly clear directly over our 2.24 km tile. Rejecting the
    whole scene over that is possibly throwing away real, usable coverage.

HOW THIS STAYS HONEST
    Same fixed sample of events, run twice, current setting (30% cloud cap)
    and a relaxed one, so any difference is caused by the parameter, not by
    which events got picked. Success is not just "found a scene" - a relaxed
    scene is only counted as a WIN if it also clears the existing per-tile
    valid-fraction bar (--min-valid, unchanged), the same quality floor
    already used everywhere else.

Output: data/optical_coverage_pilot.json
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import Resampling, reproject

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
STAC = "https://earth-search.aws.element84.com/v1/search"
PRE_DAYS, POST_DAYS = 60, 15
SCL_BAD = {0, 1, 3, 8, 9, 10}
WATER = 0.0
SAMPLE_N = 20


def search(lon, lat, start, end, max_cloud):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 30,
            "query": {"eo:cloud_cover": {"lt": max_cloud}}}
    try:
        r = requests.post(STAC, json=body, timeout=120)
        r.raise_for_status()
        return r.json().get("features", [])
    except Exception:
        return []


def mndwi_on(feat, grid):
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
    scl = out["scl"]
    fin = np.isfinite(scl)
    codes = np.zeros(scl.shape, "int16")
    codes[fin] = scl[fin].astype("int16")
    good = fin & ~np.isin(codes, list(SCL_BAD))
    g, sw = out["green"], out["swir16"]
    with np.errstate(divide="ignore", invalid="ignore"):
        m = (g - sw) / (g + sw)
    m[~good] = np.nan
    return m


def try_event(r, max_cloud: float, min_valid: float):
    lab = ROOT / "data/features" / r.EVENT_ID / "label.tif"
    if not lab.exists():
        return None
    with rasterio.open(lab) as s:
        grid = {"crs": s.crs, "transform": s.transform,
                "shape": (s.height, s.width)}
    f = r.BEGIN_UTC
    pre = search(r.BEGIN_LON, r.BEGIN_LAT,
                 (f - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d"),
                 (f - pd.Timedelta(days=1)).strftime("%Y-%m-%d"), max_cloud)
    post = search(r.BEGIN_LON, r.BEGIN_LAT, f.strftime("%Y-%m-%d"),
                  (f + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"), max_cloud)
    if not pre or not post:
        return {"outcome": "no_pair"}
    pre = sorted(pre, key=lambda x: x["properties"]["datetime"])[-1]
    post = sorted(post, key=lambda x: x["properties"]["datetime"])[0]
    mp, mq = mndwi_on(pre, grid), mndwi_on(post, grid)
    if mp is None or mq is None:
        return {"outcome": "no_pair"}
    valid = np.isfinite(mp) & np.isfinite(mq)
    vf = float(valid.mean())
    if vf < min_valid:
        return {"outcome": "too_cloudy", "valid_frac": vf}
    flood = (mq > WATER) & (mp <= WATER) & valid
    return {"outcome": "built", "valid_frac": vf,
            "flood_frac": float(flood[valid].mean()),
            "scene_cloud_pre": pre["properties"]["eo:cloud_cover"],
            "scene_cloud_post": post["properties"]["eo:cloud_cover"]}


def main() -> int:
    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    ev["EVENT_ID"] = ev.EVENT_ID.astype(int).astype(str)
    have = {p.name for p in (ROOT / "data/features").iterdir() if p.is_dir()}
    ev = ev[ev.EVENT_ID.isin(have)]
    already = {p.stem for p in (ROOT / "data/optical_labels").glob("*.tif")}
    missing = ev[~ev.EVENT_ID.isin(already)]
    sample = missing.sample(n=min(SAMPLE_N, len(missing)), random_state=0)
    print(f"testing {len(sample)} events that currently have NO optical label\n",
          flush=True)

    rows = []
    for setting, max_cloud, min_valid in (("current (30% cap)", 30, 0.30),
                                          ("relaxed (70% cap)", 70, 0.30)):
        built = 0
        for n, r in enumerate(sample.itertuples(index=False), 1):
            res = try_event(r, max_cloud, min_valid)
            outcome = res["outcome"] if res else "no_label_file"
            if outcome == "built":
                built += 1
            rows.append({"setting": setting, "event": r.EVENT_ID, **({} if res is None else res)})
            vf = f"  valid={res['valid_frac']:.2f}" if res and "valid_frac" in res else ""
            print(f"  [{setting}] [{n}/{len(sample)}] event {r.EVENT_ID}: "
                  f"{outcome}{vf}", flush=True)
        print(f"  {setting}: {built} / {len(sample)} recovered "
              f"({100*built/len(sample):.0f}%)", flush=True)

    pd.DataFrame(rows).to_csv(ROOT / "data/optical_coverage_pilot.csv", index=False)
    print("\nwrote data/optical_coverage_pilot.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
