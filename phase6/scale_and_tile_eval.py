#!/usr/bin/env python3
"""
Two evaluations the within-tile AUC could not answer.

WHY THIS EXISTS
    Within-tile AUC asks "can this feature rank pixels INSIDE one 2.24 km tile?"
    Measured variance shows rain barely varies inside a tile (ratio 0.09-0.15
    within vs between), so it CANNOT score above 0.5 there - not because it is
    uninformative, but because the question does not apply to it. Elevation is
    only slightly better (0.18). Judging them that way was a measurement error.

TEST 1 - TILE LEVEL
    Does a tile with more rain / lower ground / higher TWI flood MORE than other
    tiles? One value per tile, correlated against that tile's flood fraction.
    This is the question rain can actually answer.
    Reported per region and region-averaged, because pooling across regions lets
    a feature win by separating regions rather than tiles (section 4c).

TEST 2 - SCALE
    Trial 1 measured IoU rising from 0.24 at 10 m to 0.64 at 560 m: terrain
    predicted flooding far better at coarse scale. This aggregates both label and
    features into blocks of 20 m ... 560 m and recomputes within-tile AUC at each,
    to see whether the same holds here.

Output: data/scale_and_tile_eval.json
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = [1, 2, 4, 8, 14, 28]        # x20 m = 20, 40, 80, 160, 280, 560 m


def spearman(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(pd.Series(a[ok]).rank(), pd.Series(b[ok]).rank())[0, 1])


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() < 5 or (~y).sum() < 5:
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    """Average into k x k blocks. k=1 returns the array unchanged."""
    if k == 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return np.nanmean(a[:h, :w].reshape(h // k, k, w // k, k), axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/tiles")
    ap.add_argument("--rung", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--shards", type=int, default=0)
    args = ap.parse_args()

    d = ROOT / args.tiles
    chan = json.loads((d / "channels.json").read_text())
    names = list(chan["rung_channels"][str(args.rung)]) + list(chan["shared_channels"])
    idx = pd.read_csv(d / "index.csv")

    sh = sorted((d / "shared").glob("shard-*.npz"))
    rg = sorted((d / f"rung{args.rung}m").glob("shard-*.npz"))
    if args.shards:
        sh, rg = sh[: args.shards], rg[: args.shards]
    print(f"{len(names)} channels, {len(sh)} shards, rung {args.rung} m\n", flush=True)

    tile_rows = []                      # one row per tile: mean feature + flood frac
    scale_auc = {k: {n: [] for n in names} for k in BLOCKS}
    row_i = 0

    for ps, pr in zip(sh, rg):
        zs, zr = np.load(ps), np.load(pr)
        X = np.concatenate([zr["xt"], zs["xs"]], axis=1)
        Y, M = zs["y"], zs["m"]
        for t in range(X.shape[0]):
            m = M[t]
            if m.sum() < 200:
                row_i += 1
                continue
            y = np.where(m, Y[t], np.nan)
            frac = float(np.nanmean(y > args.threshold))
            rec = {"i": row_i,
                   "huc4": idx.huc4.iloc[row_i] if row_i < len(idx) else -1,
                   "flood_frac": frac}
            for ci, nm in enumerate(names):
                v = np.where(m, X[t, ci], np.nan)
                rec[nm] = float(np.nanmean(v))
            tile_rows.append(rec)

            # ---- scale test on this tile ----
            for k in BLOCKS:
                yb = block_mean(y, k)
                lab = yb > args.threshold
                if not np.isfinite(yb).any() or lab.sum() < 5 or (~lab).sum() < 5:
                    continue
                for ci, nm in enumerate(names):
                    sb = block_mean(np.where(m, X[t, ci], np.nan), k)
                    a = auc(-sb.ravel() if nm in NEG else sb.ravel(), lab.ravel())
                    if not np.isnan(a):
                        scale_auc[k][nm].append(a)
            row_i += 1

    tl = pd.DataFrame(tile_rows)
    print("=" * 78)
    print(f"TEST 1 — TILE LEVEL: does a feature predict WHICH TILES flood?")
    print(f"({len(tl)} tiles, {tl.huc4.nunique()} regions)")
    print("=" * 78)
    rows = []
    for nm in names:
        pooled = spearman(tl[nm], tl.flood_frac)
        per = [spearman(g[nm], g.flood_frac) for _, g in tl.groupby("huc4") if len(g) > 30]
        per = [p for p in per if np.isfinite(p)]
        rows.append({"feature": nm, "spearman_pooled": round(pooled, 4),
                     "spearman_region_avg": round(float(np.mean(per)), 4) if per else np.nan,
                     "n_regions": len(per)})
    t1 = pd.DataFrame(rows).sort_values("spearman_region_avg",
                                        key=lambda s: s.abs(), ascending=False)
    print(t1.to_string(index=False))

    print()
    print("=" * 78)
    print("TEST 2 — SCALE: within-tile AUC as cells get coarser")
    print("(trial 1: IoU 0.24 at 10 m rising to 0.64 at 560 m)")
    print("=" * 78)
    hdr = "  " + f"{'feature':<18}" + "".join(f"{k*20:>8}m" for k in BLOCKS)
    print(hdr)
    srows = []
    for nm in names:
        vals = [np.mean(scale_auc[k][nm]) if scale_auc[k][nm] else np.nan for k in BLOCKS]
        srows.append({"feature": nm, **{f"{k*20}m": round(float(v), 4)
                                        for k, v in zip(BLOCKS, vals)}})
        print("  " + f"{nm:<18}" + "".join(f"{v:>9.3f}" for v in vals))

    json.dump({"threshold": args.threshold, "rung": args.rung,
               "n_tiles": int(len(tl)),
               "tile_level": rows, "scale": srows,
               "note": "within-tile AUC cannot score features that are flat "
                       "inside a tile (rain ratio 0.09-0.15); tile-level "
                       "correlation is the applicable test for those"},
              (ROOT / "data/scale_and_tile_eval.json").open("w"), indent=1)
    print("\nwrote data/scale_and_tile_eval.json")
    return 0


# Features where LOW values mean more flooding. Direction fixed by physics.
NEG = {"elev_mean_10m", "elev_min_10m", "elev_mean_1m", "elev_min_1m",
       "elev_mean_2m", "elev_min_2m", "elev_mean_5m", "elev_min_5m",
       "elev_mean_30m", "elev_min_30m",
       "basin_hand", "basin_slope", "slope_10m", "slope_1m", "slope_2m",
       "slope_5m", "slope_30m", "soil_ksat"}

if __name__ == "__main__":
    raise SystemExit(main())
