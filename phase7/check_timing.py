#!/usr/bin/env python3
"""
Hand-label pilot: did the marks appear around the flood day, or later?

For every Sentinel-2 date from before to after, measure how much paler the
marked area is than the rest of the square (clear pixels only, clouds and
shadows removed with the SCL band). If the marks come from the flood, the gap
jumps between the last photo before the flood and the first one after it.
Also lists every other storm report within 30 km in the same window.

    ./.venv/bin/python -u phase7/check_timing.py --event 983830 \
        --dates 2021-07-26,2021-07-31,2021-08-05,2021-08-10,2021-08-15,2021-08-20,2021-08-25,2021-09-09
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
SCL_BAD = [0, 1, 3, 8, 9, 10]


def scene(lon, lat, day):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{day}T00:00:00Z/{day}T23:59:59Z", "limit": 10}
    f = requests.post(STAC, json=body, timeout=120).json()["features"]
    return min(f, key=lambda x: x["properties"]["eo:cloud_cover"])


def read(href, grid, bands, nearest=False):
    buf = np.zeros((len(bands),) + grid["shape"], dtype="float32")
    with rasterio.open(href) as s:
        for k, b in enumerate(bands):
            reproject(source=rasterio.band(s, b), destination=buf[k],
                      dst_transform=grid["transform"], dst_crs=grid["crs"],
                      resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return buf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--dates", required=True)
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    r = ev[ev.EVENT_ID.astype(int) == int(args.event)].iloc[0]
    with rasterio.open(ROOT / f"data/features/{args.event}/label.tif") as s:
        b, crs = s.bounds, s.crs
    grid = {"crs": crs, "shape": (1000, 1000), "transform": from_origin(b.left, b.top, 10, 10)}
    mark = np.load(OUT / f"{args.event}_mark_filled10m.npy")

    rows, thumbs = [], []
    for d in args.dates.split(","):
        f = scene(r.BEGIN_LON, r.BEGIN_LAT, d)
        rgb = read(f["assets"]["visual"]["href"], grid, [1, 2, 3])
        scl = read(f["assets"]["scl"]["href"], grid, [1], nearest=True)[0]
        clear = ~np.isin(scl.astype(int), SCL_BAD) & (rgb.sum(0) > 0)
        br = rgb.mean(0)
        inside, outside = mark & clear, ~mark & clear
        gap = float(br[inside].mean() - br[outside].mean()) if inside.sum() >= 0.5 * mark.sum() else np.nan
        rows.append({"date": d, "clear_pct": round(100 * clear.mean()),
                     "marked_area_seen_pct": round(100 * inside.sum() / mark.sum()),
                     "paler_by": round(gap, 1)})
        print(rows[-1], flush=True)
        thumbs.append((d, np.moveaxis(rgb, 0, -1), clear))

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"{args.event}_timing.csv", index=False)

    lo, hi = np.percentile(np.concatenate([t[1][t[2]] for t in thumbs]), [1, 99])
    n = len(thumbs)
    fig, ax = plt.subplots(1, n, figsize=(3.3 * n, 3.7))
    flood = r.BEGIN_UTC.normalize()
    for k, (d, img, clear) in enumerate(thumbs):
        im = np.clip((img - lo) / (hi - lo), 0, 1)
        im[~clear] = [0.55, 0.58, 0.62]
        ax[k].imshow(im[::2, ::2]); ax[k].set_xticks([]); ax[k].set_yticks([])
        days = (pd.Timestamp(d) - flood).days
        when = f"{-days} days before" if days < 0 else f"{days} days after"
        g = tab.paler_by.iloc[k]
        ax[k].set_title(f"{d[5:]}  ({when})\n" + ("too cloudy to tell" if np.isnan(g) else f"paths paler by {g:.0f}"),
                        fontsize=10, color="#b01820" if days >= 0 else "#17212b")
    fig.suptitle("Grey = clouds removed. Flood reported 14 Aug. Number = how much paler the marked flood paths are than the rest.", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    fig.savefig(OUT / "page/timing.jpg", dpi=110)

    # other storm reports near this square in the same window
    near = ev[(np.hypot((ev.BEGIN_LAT - r.BEGIN_LAT) * 111, (ev.BEGIN_LON - r.BEGIN_LON) * 93) < 30)
              & (ev.BEGIN_UTC >= pd.Timestamp(args.dates.split(",")[0]))
              & (ev.BEGIN_UTC <= pd.Timestamp(args.dates.split(",")[-1]))]
    print("\nstorm reports within 30 km in this window:")
    print(near[["EVENT_ID", "BEGIN_UTC", "CZ_NAME", "DAMAGE_PROPERTY"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
