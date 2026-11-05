#!/usr/bin/env python3
"""
Hand-label pilot, step 1: fetch true-colour Sentinel-2 photos before and after
one flood on the event's own grid, at 10 m, so a person (or Claude) can look.

    ./.venv/bin/python -u phase7/look_one_flood.py --event 983830 \
        --before 2021-08-05 --after 2021-08-20,2021-08-25

Output: handoff/hand_label_pilot/<event>_photos.npz and <event>_photos.png
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parent.parent
STAC = "https://earth-search.aws.element84.com/v1/search"
OUT = ROOT / "handoff/hand_label_pilot"


def scene(lon, lat, day):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z", "limit": 10}
    f = requests.post(STAC, json=body, timeout=120).json()["features"]
    return min(f, key=lambda x: x["properties"]["eo:cloud_cover"])


def rgb_on(feat, grid):
    buf = np.zeros((3,) + grid["shape"], dtype="uint8")
    with rasterio.open(feat["assets"]["visual"]["href"]) as s:
        for b in range(3):
            reproject(source=rasterio.band(s, b + 1), destination=buf[b],
                      dst_transform=grid["transform"], dst_crs=grid["crs"],
                      resampling=Resampling.bilinear)
    return np.moveaxis(buf, 0, -1)


def stretch(img, lo, hi):
    return np.clip((img.astype("float32") - lo) / (hi - lo), 0, 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    r = ev[ev.EVENT_ID.astype(int) == int(args.event)].iloc[0]
    with rasterio.open(ROOT / f"data/features/{args.event}/label.tif") as s:
        b, crs = s.bounds, s.crs
    res = 10.0
    grid = {"crs": crs, "shape": (int((b.top - b.bottom) / res), int((b.right - b.left) / res)),
            "transform": from_origin(b.left, b.top, res, res)}

    days = [args.before] + args.after.split(",")
    imgs, names = [], []
    for d in days:
        f = scene(r.BEGIN_LON, r.BEGIN_LAT, d)
        imgs.append(rgb_on(f, grid))
        names.append(f"{d} (cloud {f['properties']['eo:cloud_cover']:.0f}%)")
        print("fetched", names[-1], flush=True)

    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"{args.event}_photos.npz", days=np.array(days),
                        imgs=np.stack(imgs), bounds=np.array(tuple(b)), crs=str(crs))

    lo, hi = np.percentile(np.stack(imgs), [1, 99])
    n = len(imgs)
    fig, ax = plt.subplots(1, n, figsize=(6.5 * n, 6.8))
    for k in range(n):
        ax[k].imshow(stretch(imgs[k], lo, hi))
        ax[k].set_title(("BEFORE  " if k == 0 else "AFTER  ") + names[k], fontsize=12)
        ax[k].set_xticks(np.arange(0, grid["shape"][1] + 1, 100))
        ax[k].set_yticks(np.arange(0, grid["shape"][0] + 1, 100))
        ax[k].tick_params(labelsize=7); ax[k].grid(color="white", alpha=0.25, lw=0.5)
    fig.suptitle(f"event {args.event}  {r.basin_name}  flood {str(r.BEGIN_UTC)[:10]}  "
                 f"report point marked on grid  (10 km x 10 km, 10 m pixels, grid = 1 km)",
                 fontsize=12)
    # the reported flood point
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform(r.BEGIN_LON, r.BEGIN_LAT)
    for k in range(n):
        ax[k].plot((x - b.left) / res, (b.top - y) / res, "o", mfc="none", mec="red", ms=14, mew=2)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUT / f"{args.event}_photos.png", dpi=110)
    print("wrote", OUT / f"{args.event}_photos.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
