#!/usr/bin/env python3
"""
How much flood is actually in the data? Measured across many events.

Every flood-fraction figure so far came from ONE event (983830, a $25M flood,
probably larger than typical), which is a thin basis for a training decision.
This samples real groups on disk and reports the distribution.

Read-only, so it is safe while the DEM download runs.

Two framings, because they answer different questions:
  - fraction of PIXELS above a coherence-drop threshold (class balance)
  - fraction of TILES containing any flood (how many training tiles are usable,
    which is what skip-empty-tile training turns on)

Restricted to a 10 km neighbourhood of the event, since that is what a training
tile set would cover - not the whole 92 x 37 km burst, most of which is desert
nowhere near the reported flood.
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

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

GROUPS = ROOT / "data/phase3/groups"
NEAR_KM = 10.0
TILE_PX = 112          # 2.24 km at 20 m, matching trial 1's ground extent
THRESHOLDS = (0.10, 0.15, 0.20)


def corr_paths(gdir: Path):
    b = sorted((gdir / "baseline").glob("*_corr.tif"))
    c = sorted((gdir / "coevent").glob("*_corr.tif"))
    return (str(b[0]), str(c[0])) if b and c else (None, None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    ev_state = st.get("events", {})
    recs = {}
    for line in (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[str(r["EVENT_ID"])] = r

    # one representative event per group, so groups are not double counted
    by_group = {}
    for eid, v in ev_state.items():
        if v.get("stage") == "done" and v.get("group_id") and eid in recs:
            by_group.setdefault(v["group_id"], (eid, v))
    gids = sorted(by_group)
    random.Random(args.seed).shuffle(gids)
    gids = gids[: args.sample]
    print(f"sampling {len(gids)} of {len(by_group)} completed groups\n", flush=True)

    rows = []
    for i, gid in enumerate(gids, 1):
        eid, v = by_group[gid]
        gdir = GROUPS / gid
        bp, cp = corr_paths(gdir)
        if not bp:
            continue
        try:
            (bc, cc), transform, crs = read_aligned([bp, cp])
        except Exception as e:
            print(f"  {gid}: skip ({type(e).__name__})", flush=True)
            continue

        valid = np.isfinite(bc) & np.isfinite(cc)
        drop = np.where(valid, bc - cc, np.nan)

        r = recs[eid]
        rr, cc_ix = np.indices(bc.shape)
        xs, ys = rasterio.transform.xy(transform, rr, cc_ix)
        xs = np.array(xs).reshape(bc.shape); ys = np.array(ys).reshape(bc.shape)
        ex, ey = warp_transform("EPSG:4326", crs, [r["lon"]], [r["lat"]])
        near = valid & (np.hypot(xs - ex[0], ys - ey[0]) / 1000.0 <= NEAR_KM)
        if near.sum() < 1000:
            continue

        row = {"group": gid, "event": eid, "huc4": r["huc4"],
               "basin": r["basin_name"], "near_px": int(near.sum())}
        for thr in THRESHOLDS:
            fl = near & (drop > thr)
            row[f"pix_{thr}"] = 100 * fl.sum() / near.sum()

            # tile-level: how many 2.24 km tiles contain any flood
            H, W = drop.shape
            n_tiles = n_any = n_5 = 0
            for a in range(0, H - TILE_PX + 1, TILE_PX):
                for b in range(0, W - TILE_PX + 1, TILE_PX):
                    sl = (slice(a, a + TILE_PX), slice(b, b + TILE_PX))
                    m = near[sl]
                    if m.mean() < 0.5:
                        continue
                    n_tiles += 1
                    f = fl[sl].sum() / max(m.sum(), 1)
                    if f > 0.01:
                        n_any += 1
                    if f > 0.05:
                        n_5 += 1
            row[f"tiles_{thr}"] = n_tiles
            row[f"tiles_any_{thr}"] = n_any
            row[f"tiles_5pct_{thr}"] = n_5
        rows.append(row)
        if i % 10 == 0:
            print(f"  {i}/{len(gids)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/flood_fraction_sample.csv", index=False)

    print("\n" + "=" * 74)
    print(f"FLOOD FRACTION, {len(df)} events, within {NEAR_KM:g} km of the report")
    print("=" * 74)
    print(f"{'threshold':>10} | {'pixels flooded (%)':^34} | tiles with flood")
    print(f"{'':>10} | {'median':>8} {'IQR':>14} {'max':>8} | {'>1%':>6} {'>5%':>6}")
    for thr in THRESHOLDS:
        p = df[f"pix_{thr}"]
        tt, ta, t5 = df[f"tiles_{thr}"].sum(), df[f"tiles_any_{thr}"].sum(), df[f"tiles_5pct_{thr}"].sum()
        print(f"{thr:>10.2f} | {p.median():>7.1f}% {p.quantile(.25):>6.1f}-{p.quantile(.75):<6.1f} "
              f"{p.max():>7.1f}% | {100*ta/tt:>5.0f}% {100*t5/tt:>5.0f}%")

    print()
    print(f"tiles per event (2.24 km): median {df['tiles_0.15'].median():.0f}")
    print()
    print("events with essentially NO flood signal (<1% pixels at 0.15):")
    n0 = (df["pix_0.15"] < 1).sum()
    print(f"  {n0} of {len(df)} ({100*n0/len(df):.0f}%)")
    print()
    print("by region (median % pixels at threshold 0.15):")
    print(df.groupby("huc4")["pix_0.15"].agg(["count", "median"]).round(1).to_string())
    print()
    print("for comparison, trial 1: 9.43% of pixels flood, 69.3% of tiles had none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
