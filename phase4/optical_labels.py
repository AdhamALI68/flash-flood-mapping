#!/usr/bin/env python3
"""
Phase 4 - build an INDEPENDENT per-pixel flood label from Sentinel-2 optical.

    ./.venv/bin/python -u phase4/optical_labels.py --sample 60

WHY THIS IS THE MOST IMPORTANT SCRIPT IN THE PROJECT
    Every number we have is graded against InSAR coherence drop, and we measured
    that label's limit: HyP3 gave 5 looks per pixel, so the drop carries about
    +/-0.09 of estimation noise against a 0.15 threshold, and a 3-sigma cut
    fires on 0.00% of pixels. Most of what that label calls flood sits inside
    its own noise.

    If that is really the ceiling, then the SAME model scored against a cleaner
    label must do better. This builds the cleaner label so the claim can be
    tested rather than asserted.

    A confirmed result turns the project's weakness into its headline: not
    "our model gets 0.69" but "InSAR coherence at 5 looks cannot support
    pixel-level flood labelling, and here is the measurement".

THE OPTICAL LABEL
    MNDWI = (green - swir) / (green + swir). Water reflects green and absorbs
    shortwave infrared, so open water sits above 0 and dry ground below.

        flood = water AFTER the event AND not water BEFORE it

    Requiring "not before" removes rivers, reservoirs and irrigation, which are
    wet in both scenes and are not floods.

    It is independent of the SAR in every way that matters: different satellite,
    different physics (reflected sunlight, not radar phase), different failure
    modes (cloud rather than decorrelation).

WHAT IT CANNOT DO
    Sentinel-2 revisits every 5 days and a desert flash flood drains in hours,
    so the post-flood scene is usually days late and the water has gone. What
    survives is wet soil and fresh sediment, which still raise MNDWI. Events
    with no clear pair are skipped rather than guessed at, and the count of
    usable events is reported honestly.

Output: data/optical_labels/<event>.tif  (flood mask + delta MNDWI + validity)
"""

from __future__ import annotations

import argparse
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
OUT = ROOT / "data/optical_labels"
STAC = "https://earth-search.aws.element84.com/v1/search"
MAX_CLOUD = 30
PRE_DAYS = 60
POST_DAYS = 15
SCL_BAD = {0, 1, 3, 8, 9, 10}      # nodata, saturated, shadow, cloud, cirrus
WATER = 0.0                         # MNDWI above this is open water


def search(lon, lat, start, end):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            # RFC3339 or the API returns 400. Plain YYYY-MM-DD is rejected.
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 30,
            "query": {"eo:cloud_cover": {"lt": MAX_CLOUD}}}
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    ap.add_argument("--events", default="",
                    help="path to a JSON file whose 'events' keys are the "
                         "event ids to build. Used to make the optical labels "
                         "cover exactly the looks-pilot events, so the two "
                         "experiments can be joined - they were built from "
                         "different orderings the first time and shared none")
    ap.add_argument("--min-valid", type=float, default=0.30,
                    help="skip events where clouds leave less than this "
                         "fraction of the grid usable in BOTH scenes")
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    ev["EVENT_ID"] = ev.EVENT_ID.astype(int).astype(str)
    have = {p.name for p in (ROOT / "data/features").iterdir() if p.is_dir()}
    ev = ev[ev.EVENT_ID.isin(have)]
    # Prefer events in the held-out test regions - those are the ones the
    # headline number is computed on, so a cleaner label there is worth most.
    if args.events:
        want = set(json.loads(Path(args.events).read_text())["events"])
        ev = ev[ev.EVENT_ID.isin(want)]
        print(f"restricted to {len(ev)} events from {args.events}")
    else:
        ev["prio"] = ev.huc4.astype(str).isin(["1606", "1809"]).map(
            {True: 0, False: 1})
        ev = ev.sort_values(["prio"]).head(args.sample * 3)

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"attempting {len(ev)} events, want {args.sample}\n", flush=True)

    made, no_pair, too_cloudy = 0, 0, 0
    rows = []
    for n, r in enumerate(ev.itertuples(index=False), 1):
        if made >= args.sample:
            break
        dest = OUT / f"{r.EVENT_ID}.tif"
        if dest.exists():
            made += 1
            continue
        lab = ROOT / "data/features" / r.EVENT_ID / "label.tif"
        if not lab.exists():
            continue
        with rasterio.open(lab) as s:
            grid = {"crs": s.crs, "transform": s.transform,
                    "shape": (s.height, s.width)}

        f = r.BEGIN_UTC
        pre = search(r.BEGIN_LON, r.BEGIN_LAT,
                     (f - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d"),
                     (f - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))
        post = search(r.BEGIN_LON, r.BEGIN_LAT,
                      f.strftime("%Y-%m-%d"),
                      (f + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"))
        if not pre or not post:
            no_pair += 1
            print(f"  [{n}] {r.EVENT_ID} no clear pair", flush=True)
            continue
        pre = sorted(pre, key=lambda x: x["properties"]["datetime"])[-1]
        post = sorted(post, key=lambda x: x["properties"]["datetime"])[0]

        mp, mq = mndwi_on(pre, grid), mndwi_on(post, grid)
        if mp is None or mq is None:
            no_pair += 1
            print(f"  [{n}] {r.EVENT_ID} no clear pair (read failed)", flush=True)
            continue
        valid = np.isfinite(mp) & np.isfinite(mq)
        if valid.mean() < args.min_valid:
            too_cloudy += 1
            print(f"  [{n}] {r.EVENT_ID} too cloudy, valid {100*valid.mean():.0f}%", flush=True)
            continue

        # water after AND not water before -> flood, not a permanent river
        flood = (mq > WATER) & (mp <= WATER) & valid
        d = np.where(valid, mq - mp, np.nan).astype("float32")

        prof = dict(driver="GTiff", height=grid["shape"][0],
                    width=grid["shape"][1], count=3, dtype="float32",
                    crs=grid["crs"], transform=grid["transform"],
                    nodata=np.nan, compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256)
        with rasterio.open(dest, "w", **prof) as o:
            o.write(flood.astype("float32"), 1); o.set_band_description(1, "flood_optical")
            o.write(d, 2); o.set_band_description(2, "dmndwi")
            o.write(valid.astype("float32"), 3); o.set_band_description(3, "valid")
        made += 1
        lag = (pd.Timestamp(post["properties"]["datetime"][:10]) - f.normalize()).days
        rows.append({"EVENT_ID": r.EVENT_ID, "huc4": r.huc4,
                     "pre": pre["properties"]["datetime"][:10],
                     "post": post["properties"]["datetime"][:10],
                     "lag_days": lag,
                     "valid_frac": round(float(valid.mean()), 3),
                     "flood_frac": round(float(flood[valid].mean()), 4)})
        print(f"  [{n}] {r.EVENT_ID} r{r.huc4}  lag {lag}d  "
              f"valid {100*valid.mean():.0f}%  flood {100*flood[valid].mean():.2f}%",
              flush=True)

    if rows:
        # merge, never overwrite: earlier runs' rows were lost this way before
        csv = ROOT / "data/optical_labels.csv"
        new = pd.DataFrame(rows)
        if csv.exists():
            new = pd.concat([pd.read_csv(csv, dtype={"EVENT_ID": str}), new])
        new.drop_duplicates("EVENT_ID", keep="last").to_csv(csv, index=False)
    print(f"\nbuilt {made} optical labels   "
          f"(no clear pair {no_pair}, too cloudy {too_cloudy})")
    print(f"output: {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
