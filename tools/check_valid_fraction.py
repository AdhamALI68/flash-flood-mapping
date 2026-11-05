#!/usr/bin/env python3
"""
How empty are the tiles really? Measure before choosing a threshold.

build_tiles.py keeps a tile when >=60% of its pixels are valid, which I picked
without measuring anything. If most tiles are ~99% valid, 60% is a harmless
safety net. If many sit near 60%, then a large share of training pixels are
masked out and the threshold is doing real damage quietly.

Measured on the coherence rasters cropped to the 5 km event radius - the same
crop build_features.py uses. Coherence is the dominant source of invalid pixels
(a Sentinel-1 burst sits diagonally inside a rectangular raster, so the corners
are empty by construction), so this is the binding constraint.

Reports the distribution per tile and per pixel, and what each candidate
threshold would keep or discard.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window, from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

RADIUS_M = 5000
TILE = 112


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=60)
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    rows = ev.to_dict("records")
    random.Random(0).shuffle(rows)

    per_event, tile_fracs = [], []
    n = 0
    for r in rows:
        if n >= args.sample:
            break
        gid = st["events"].get(str(int(r["EVENT_ID"])), {}).get("group_id")
        if not gid:
            continue
        d = ROOT / "data/phase3/groups" / gid
        try:
            bc = sorted((d / "baseline").glob("*_corr.tif"))[0]
            cc = sorted((d / "coevent").glob("*_corr.tif"))[0]
        except IndexError:
            continue
        # Crop to the event radius, exactly as build_features does.
        with rasterio.open(cc) as s:
            ex, ey = warp_transform("EPSG:4326", s.crs,
                                    [r["BEGIN_LON"]], [r["BEGIN_LAT"]])
            win = from_bounds(ex[0] - RADIUS_M, ey[0] - RADIUS_M,
                              ex[0] + RADIUS_M, ey[0] + RADIUS_M,
                              transform=s.transform)
            win = win.intersection(Window(0, 0, s.width, s.height))
            if win.width < 50 or win.height < 50:
                continue
            bounds = rasterio.windows.bounds(win, s.transform)
        try:
            (a, b), tf, crs = read_aligned([str(bc), str(cc)],
                                           aoi=(bounds[0], bounds[1], bounds[2], bounds[3]))
        except Exception:
            continue
        m = np.isfinite(a) & np.isfinite(b)
        if m.size < 1000:
            continue
        per_event.append({"event": int(r["EVENT_ID"]), "huc4": str(r["huc4"]),
                          "px": int(m.size), "valid_frac": float(m.mean())})
        H, W = m.shape
        for i in range(0, H - TILE + 1, TILE):
            for j in range(0, W - TILE + 1, TILE):
                tile_fracs.append(float(m[i:i + TILE, j:j + TILE].mean()))
        n += 1
        if n % 15 == 0:
            print(f"  {n}/{args.sample} events", flush=True)

    pe = pd.DataFrame(per_event)
    tf_ = np.array(tile_fracs)

    print("\n" + "=" * 72)
    print(f"VALID-PIXEL FRACTION — {len(pe)} events, {len(tf_)} candidate tiles")
    print("=" * 72)
    print("\nper EVENT (5 km crop):")
    print(f"  median {pe.valid_frac.median():.3f}   "
          f"p10 {pe.valid_frac.quantile(0.1):.3f}   min {pe.valid_frac.min():.3f}")
    print(f"  events >95% valid: {100*(pe.valid_frac>0.95).mean():.0f}%")

    print("\nper TILE (112x112):")
    for q in (0.01, 0.05, 0.10, 0.25, 0.50):
        print(f"  p{int(q*100):<3} {np.quantile(tf_, q):.3f}")
    print(f"  fully valid (=1.0):        {100*(tf_ >= 0.999).mean():.1f}% of tiles")
    print(f"  >99% valid:                {100*(tf_ > 0.99).mean():.1f}%")
    print(f"  >90% valid:                {100*(tf_ > 0.90).mean():.1f}%")

    print("\nwhat each threshold would do:")
    print(f"  {'threshold':>10} {'tiles kept':>12} {'% kept':>8} "
          f"{'masked px in kept tiles':>24}")
    for t in (0.0, 0.5, 0.6, 0.8, 0.9, 0.95, 0.99):
        keep = tf_ >= t
        if keep.sum() == 0:
            continue
        wasted = 1 - tf_[keep].mean()
        print(f"  {t:>10.2f} {int(keep.sum()):>12,} {100*keep.mean():>7.1f}% "
              f"{100*wasted:>23.2f}%")

    print("\n" + "=" * 72)
    frac_bad = float((tf_ < 0.999).mean())
    if (tf_ >= 0.999).mean() > 0.9:
        print("Nearly every tile is FULLY valid. The 5 km crop sits well inside")
        print("the burst, so the empty corners never reach it. A 60% threshold is")
        print("a harmless safety net, and could be raised to 0.99 at almost no cost.")
    else:
        print(f"{100*frac_bad:.0f}% of tiles are partly empty — the threshold is")
        print("doing real work and should be chosen deliberately, not guessed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
