#!/usr/bin/env python3
"""
Class balance for the Gate B event.

Answers two questions before Phase 3 commits to a labelling threshold:

  1. what fraction of pixels would be labelled flood, at each candidate threshold
  2. is there permanent water in the scene, and how much

Neither has a single answer. "Flood percent" depends entirely on the coherence
drop threshold chosen, so this reports a sweep rather than one number. And it
depends just as much on how tightly the scene is cropped around the flood - the
whole burst is 92 x 37 km of mostly-dry desert, which is not what a training
tile looks like.

Permanent water is estimated from SAR amplitude: smooth water reflects the radar
away from the sensor, so standing water is persistently dark. Dark in BOTH the
before and after pair means permanent, not flood.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parent
EVENT = 983830


def path(kind: str, which: str) -> str:
    return glob.glob(str(ROOT / f"{EVENT}_{which}" / "*" / f"*_{kind}.tif"))[0]


def main() -> int:
    # All four rasters must sit on one grid - the baseline and co-event products
    # have different origins (see align.py).
    from align import read_aligned

    (base, coev, amp_b, amp_c), transform, crs = read_aligned([
        path("corr", "baseline"), path("corr", "coevent"),
        path("amp", "baseline"), path("amp", "coevent"),
    ])

    valid = np.isfinite(base) & np.isfinite(coev)
    drop = np.where(valid, base - coev, np.nan)

    binfo = json.loads((ROOT / f"{EVENT}_bursts.json").read_text())
    rows, cols = np.indices(base.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs).reshape(base.shape); ys = np.array(ys).reshape(base.shape)
    ex, ey = warp_transform("EPSG:4326", crs, [binfo["lon"]], [binfo["lat"]])
    dist_km = np.sqrt((xs - ex[0]) ** 2 + (ys - ey[0]) ** 2) / 1000.0

    px_km2 = 400 / 1e6
    print(f"burst extent : {base.shape[1]*20/1000:.0f} x {base.shape[0]*20/1000:.0f} km")
    print(f"valid pixels : {int(valid.sum()):,}  ({valid.sum()*px_km2:,.0f} km2)")
    print()

    # ---- permanent water -------------------------------------------------
    print("=" * 68)
    print("PERMANENT WATER  (persistently dark in SAR amplitude)")
    print("=" * 68)
    ampv = np.isfinite(amp_b) & np.isfinite(amp_c)
    db_b = 10 * np.log10(amp_b, where=ampv, out=np.full_like(amp_b, np.nan))
    db_c = 10 * np.log10(amp_c, where=ampv, out=np.full_like(amp_c, np.nan))
    med = np.nanmedian(db_b)
    print(f"median amplitude (dB, pre)  : {med:.1f}")
    for k in (10, 12, 15):
        thr = med - k
        pw = ampv & (db_b < thr) & (db_c < thr)
        print(f"  dark in BOTH, < median-{k:>2} dB : {pw.sum():>9,} px  "
              f"{100*pw.sum()/valid.sum():>6.3f}%  ({pw.sum()*px_km2:>7.1f} km2)")
    perm = ampv & (db_b < med - 12) & (db_c < med - 12)

    # ---- flood, threshold sweep -----------------------------------------
    print()
    print("=" * 68)
    print("FLOOD FRACTION vs COHERENCE-DROP THRESHOLD")
    print("=" * 68)
    print(f"{'threshold':>10} | {'whole burst':>22} | {'within 10 km':>22}")
    print(f"{'':>10} | {'px':>10} {'%':>10} | {'px':>10} {'%':>10}")
    near = valid & (dist_km <= 10)
    for thr in (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40):
        f_all = valid & (drop > thr) & ~perm
        f_near = near & (drop > thr) & ~perm
        print(f"{thr:>10.2f} | {int(f_all.sum()):>10,} {100*f_all.sum()/valid.sum():>9.3f}% "
              f"| {int(f_near.sum()):>10,} {100*f_near.sum()/near.sum():>9.3f}%")

    print()
    print("drop distribution (whole burst):")
    qs = np.nanpercentile(drop[valid], [50, 75, 90, 95, 99, 99.9])
    for q, v in zip([50, 75, 90, 95, 99, 99.9], qs):
        print(f"  p{q:<5} {v:+.4f}")

    print()
    print("for comparison, trial 1: flood 9.43% of pixels, permanent water 0.97%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
