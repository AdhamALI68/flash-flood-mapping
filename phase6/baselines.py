#!/usr/bin/env python3
"""
Phase 6 - baselines, and the metrics everything else is judged against.

Trial 1's central finding: a 961k-parameter U-Net scored 0.7779 against 0.7790
for simply ranking pixels from low to high elevation. It never beat the free
baseline. So the baseline is the result to beat, not zero.

WITHIN-TILE, NOT POOLED. Trial 1 measured 0.87 pooled against 0.64 within-tile
on identical predictions. Pooling lets a model win by learning which TILES are
wet rather than which PIXELS are - it ranks a dry pixel in a flooded tile above a
wet pixel in a dry tile and is rewarded. Within-tile AUC is computed per tile and
averaged, so only within-tile ordering counts. Both are reported; the within-tile
number is the honest one.

Baselines, weakest to strongest:
  random            sanity floor, should be 0.5
  elevation         rank by -elevation. Water flows downhill; this is free
  basin HAND        height above the nearest drainage, from complete-basin routing
  basin TWI         topographic wetness index
  best single       whichever channel scores highest on its own

Any model must beat the strongest of these, on within-tile AUC, by more than the
noise floor, to count.

Output: data/baselines_<rung>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def auc_within(scores: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> float:
    """AUC inside ONE tile. NaN if the tile has no contrast to measure."""
    s = scores[mask]
    y = labels[mask]
    if s.size < 50:
        return np.nan
    pos = y > 0
    if pos.sum() < 5 or (~pos).sum() < 5:
        return np.nan          # nothing to separate in this tile
    r = pd.Series(s).rank().values
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def auc_pooled(scores, labels, masks) -> float:
    s = np.concatenate([sc[m] for sc, m in zip(scores, masks)])
    y = np.concatenate([lb[m] for lb, m in zip(labels, masks)])
    pos = y > 0
    if pos.sum() < 5 or (~pos).sum() < 5:
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = int(pos.sum()), int((~pos).sum())
    return float((r[pos].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def load_shards(d: Path, rung: int, limit_shards: int = 0):
    """Yield (X, y, mask) with shared and rung-specific channels rejoined.

    build_tiles stores the 16 rung-invariant channels once in shared/ and each
    rung's 5 terrain channels separately. Tile order is identical across both by
    construction, so shard i row j is the same tile in each.
    """
    sh = sorted((d / "shared").glob("shard-*.npz"))
    rg = sorted((d / f"rung{rung}m").glob("shard-*.npz"))
    if len(sh) != len(rg):
        raise SystemExit(f"shard count mismatch: shared {len(sh)} vs "
                         f"rung{rung}m {len(rg)} - rebuild tiles")
    if limit_shards:
        sh, rg = sh[:limit_shards], rg[:limit_shards]
    for ps, pr in zip(sh, rg):
        zs, zr = np.load(ps), np.load(pr)
        xs, xt = zs["xs"], zr["xt"]
        if xs.shape[0] != xt.shape[0]:
            raise SystemExit(f"tile count mismatch in {ps.name}: "
                             f"{xs.shape[0]} vs {xt.shape[0]}")
        # rung channels first, matching the order in channels.json
        yield np.concatenate([xt, xs], axis=1), zs["y"], zs["m"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/tiles")
    ap.add_argument("--rung", type=int, default=10)
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="cutoff turning the continuous label binary, at SCORING time")
    ap.add_argument("--shards", type=int, default=0)
    args = ap.parse_args()

    d = ROOT / args.tiles
    chan = json.loads((d / "channels.json").read_text())
    names = list(chan["rung_channels"][str(args.rung)]) + list(chan["shared_channels"])
    idx = pd.read_csv(d / "index.csv")
    print(f"{len(idx)} tiles, {len(names)} channels (rung {args.rung}m), "
          f"threshold {args.threshold}")
    print(f"regions: {sorted(idx.huc4.unique())}\n", flush=True)

    # Candidate single-channel scores. Sign is set by physics, not fitted:
    # low ground floods, so elevation and HAND are negated.
    negate = {n for n in names
              if any(k in n for k in ("elev_", "basin_hand", "basin_slope", "slope_"))}

    per_tile = {n: [] for n in names}
    per_tile["random"] = []
    pooled_store = {n: ([], [], []) for n in names}
    rng = np.random.default_rng(0)
    n_tiles = n_scored = 0

    for X, Y, M in load_shards(d, args.rung, args.shards):
        yb = (Y > args.threshold)
        for t in range(X.shape[0]):
            n_tiles += 1
            m = M[t]
            lab = yb[t]
            if m.sum() < 50:
                continue
            got = False
            for ci, nm in enumerate(names):
                sc = X[t, ci]
                if nm in negate:
                    sc = -sc
                a = auc_within(sc, lab, m)
                if not np.isnan(a):
                    per_tile[nm].append(a)
                    got = True
                s_, l_, m_ = pooled_store[nm]
                if len(s_) < 400:          # cap pooled memory
                    s_.append(sc); l_.append(lab); m_.append(m)
            a = auc_within(rng.random(m.shape), lab, m)
            if not np.isnan(a):
                per_tile["random"].append(a)
            n_scored += int(got)

    rows = []
    for nm, vals in per_tile.items():
        if not vals:
            continue
        v = np.array(vals)
        row = {"feature": nm, "n_tiles_scored": len(v),
               "auc_within_tile": round(float(v.mean()), 4),
               "auc_within_std": round(float(v.std()), 4)}
        if nm in pooled_store and pooled_store[nm][0]:
            s_, l_, m_ = pooled_store[nm]
            row["auc_pooled"] = round(auc_pooled(s_, l_, m_), 4)
        rows.append(row)

    res = pd.DataFrame(rows).sort_values("auc_within_tile", ascending=False)
    print("=" * 84)
    print(f"SINGLE-FEATURE BASELINES — within-tile AUC ({n_scored} tiles scorable "
          f"of {n_tiles})")
    print("=" * 84)
    print(res.to_string(index=False))

    best = res.iloc[0]
    elev = res[res.feature.str.startswith("elev_mean")]
    print()
    print(f"STRONGEST SINGLE FEATURE : {best.feature} = {best.auc_within_tile}")
    if len(elev):
        e = elev.iloc[0]
        print(f"ELEVATION-ONLY RULE      : {e.feature} = {e.auc_within_tile}"
              f"   <- trial 1's 0.7790 equivalent")
    r = res[res.feature == "random"]
    if len(r):
        print(f"random sanity floor      : {r.iloc[0].auc_within_tile} (should be ~0.5)")
    print()
    print("A model counts as beating these only if it exceeds the strongest by")
    print("more than the noise floor, on WITHIN-TILE AUC, with a paired test.")

    if "auc_pooled" in res.columns:
        gap = res.dropna(subset=["auc_pooled"])
        if len(gap):
            print()
            print("pooled vs within-tile (trial 1 saw 0.87 vs 0.64 on identical preds):")
            for r_ in gap.head(5).itertuples(index=False):
                print(f"  {r_.feature:<24} pooled {r_.auc_pooled:.4f}  "
                      f"within {r_.auc_within_tile:.4f}")

    out = ROOT / f"data/baselines_rung{args.rung}m.json"
    out.write_text(json.dumps({"tiles": args.tiles, "threshold": args.threshold,
                               "n_tiles": n_tiles, "results": rows}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
