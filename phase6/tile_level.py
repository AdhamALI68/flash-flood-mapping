#!/usr/bin/env python3
"""
Phase 6 - predict how much of a catchment floods, not which pixel.

    ./.venv/bin/python -u phase6/tile_level.py

WHY THIS IS THE RIGHT UNIT FOR THIS LABEL
    Our label is InSAR coherence drop, and we measured its limit: 5 looks per
    pixel gives about +/-0.09 of noise on the drop against a 0.15 threshold.
    Per pixel that noise dominates.

    A tile holds 112 x 112 = 12,544 pixels. Averaging them cuts random label
    noise by roughly sqrt(12544) ~ 112x. The flood FRACTION of a tile is
    therefore a far cleaner target than any single pixel in it, from exactly the
    same data.

    It is also the more useful question. Nobody acts on a 20 m pixel; warnings
    are issued for catchments.

WHAT IS PREDICTED
    Two framings, both reported, because they answer different questions:

      regression      what fraction of this tile floods        (Spearman, R2)
      classification  does this tile flood at all              (AUC, recall)

    The quiet controls are included, so "does anything flood here today" is a
    real question with real negatives rather than an artefact of tile choice.

BASELINES IT MUST BEAT
    The same tile-mean features it is built from - the strongest of which
    already reach Spearman +/-0.23 on their own. A model that cannot beat its
    own best input has not earned anything.

Output: data/tile_level.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_tiles, region_split                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def tile_table(X, Y, M, idx, thr):
    """One row per tile: the mean of every channel, plus what actually flooded."""
    rows, keep = [], []
    for i in range(len(X)):
        m = M[i]
        if m.sum() < 2000:               # a tile that is mostly masked is noise
            continue
        y = Y[i][m]
        rows.append(np.array([X[i][c][m].mean() for c in range(X.shape[1])]))
        keep.append({"i": i, "flood_frac": float((y > thr).mean()),
                     "valid_px": int(m.sum())})
    return np.array(rows), pd.DataFrame(keep)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="10")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--drop", default="",
                    help="comma-separated channel name prefixes to exclude, "
                         "e.g. lc_ to test whether land cover is carrying the "
                         "result. Cropland correlates with the label at +0.37 "
                         "because ploughing and irrigation destroy coherence "
                         "with no flood involved")
    ap.add_argument("--flood-tile", type=float, default=0.05,
                    help="a tile counts as flooded if this fraction of it did")
    args = ap.parse_args()

    print("loading tiles...", flush=True)
    X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", args.rung)
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", args.rung)
    is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)

    if args.drop:
        pre = tuple(x for x in args.drop.split(",") if x)
        keepc = [i for i, n in enumerate(names) if not n.startswith(pre)]
        X = X[:, keepc]
        names = [names[i] for i in keepc]
        print(f"  dropped channels starting with {pre}: "
              f"{len(keepc)} features left")
    print("building the tile table...", flush=True)
    F, meta = tile_table(X, Y, M, idx, args.threshold)
    meta["huc4"] = idx.huc4.astype(str).values[meta.i.values]
    meta["is_neg"] = is_neg[meta.i.values]
    del X
    tr, va, te = region_split(idx.iloc[meta.i.values].reset_index(drop=True),
                              ["1606", "1809"], ["1507"])
    print(f"  {len(meta)} tiles x {F.shape[1]} features")
    print(f"  train {int(tr.sum())}  val {int(va.sum())}  test {int(te.sum())}")
    print(f"  flood fraction: median {meta.flood_frac.median():.3f}  "
          f"mean {meta.flood_frac.mean():.3f}")
    yb = (meta.flood_frac >= args.flood_tile).values
    print(f"  tiles counted as flooded (>={args.flood_tile}): "
          f"{100*yb.mean():.1f}%\n", flush=True)

    import xgboost as xgb
    res = {}

    # ---------- regression: how much of the tile floods --------------------
    reg = xgb.XGBRegressor(n_estimators=1500, max_depth=6, learning_rate=0.03,
                           subsample=0.8, colsample_bytree=0.7,
                           min_child_weight=8, reg_lambda=1.0, n_jobs=-1,
                           tree_method="hist", early_stopping_rounds=50,
                           random_state=0)
    reg.fit(F[tr], meta.flood_frac.values[tr],
            eval_set=[(F[va], meta.flood_frac.values[va])], verbose=False)
    pr = reg.predict(F[te])
    truth = meta.flood_frac.values[te]
    rho = stats.spearmanr(pr, truth).statistic
    r2 = 1 - ((pr - truth) ** 2).sum() / ((truth - truth.mean()) ** 2).sum()
    res["regression"] = {"spearman": float(rho), "r2": float(r2),
                         "mae": float(np.abs(pr - truth).mean())}
    print("=" * 70)
    print("HOW MUCH OF THE CATCHMENT FLOODS?  (regression on tile fraction)")
    print("=" * 70)
    print(f"  Spearman   {rho:.4f}")
    print(f"  R2         {r2:.4f}")
    print(f"  mean error {np.abs(pr-truth).mean():.4f} of the tile")

    # best single feature, same test tiles, for comparison
    best_f, best_r = None, 0.0
    for c, nm in enumerate(names):
        s = stats.spearmanr(F[te][:, c], truth).statistic
        if np.isfinite(s) and abs(s) > abs(best_r):
            best_f, best_r = nm, s
    print(f"\n  best single feature: {best_f} = {best_r:+.4f}")
    print(f"  model beats it by  : {abs(rho)-abs(best_r):+.4f}")
    res["best_single_feature"] = {"name": best_f, "spearman": float(best_r)}

    # ---------- classification: does this catchment flood at all ------------
    clf = xgb.XGBClassifier(n_estimators=1500, max_depth=6, learning_rate=0.03,
                            subsample=0.8, colsample_bytree=0.7,
                            min_child_weight=8, reg_lambda=1.0, n_jobs=-1,
                            tree_method="hist", eval_metric="aucpr",
                            early_stopping_rounds=50, random_state=0)
    clf.fit(F[tr], yb[tr], eval_set=[(F[va], yb[va])], verbose=False)
    pv, pt = clf.predict_proba(F[va])[:, 1], clf.predict_proba(F[te])[:, 1]
    auc = roc_auc_score(yb[te], pt)
    apr = average_precision_score(yb[te], pt)
    best_t, best_f1 = 0.5, 0.0
    for t in np.quantile(pv, np.linspace(0.3, 0.99, 80)):
        p = pv >= t
        tp = float((p & yb[va]).sum()); fp = float((p & ~yb[va]).sum())
        fn = float((~p & yb[va]).sum())
        if tp == 0:
            continue
        pr_, rc_ = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr_ * rc_ / (pr_ + rc_)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    p = pt >= best_t
    tp = float((p & yb[te]).sum()); fp = float((p & ~yb[te]).sum())
    fn = float((~p & yb[te]).sum()); tn = float((~p & ~yb[te]).sum())
    rec, pre = tp / (tp + fn), tp / (tp + fp)
    res["classification"] = {"auc": float(auc), "ap": float(apr),
                             "recall": float(rec), "precision": float(pre),
                             "f1": 2 * pre * rec / (pre + rec),
                             "accuracy": float((tp + tn) / len(p)),
                             "base": float(yb[te].mean())}
    print("\n" + "=" * 70)
    print("DOES THIS CATCHMENT FLOOD AT ALL?  (classification)")
    print("=" * 70)
    print(f"  AUC        {auc:.4f}")
    print(f"  AP         {apr:.4f}   (base rate {yb[te].mean():.4f})")
    print(f"  recall     {rec:.4f}")
    print(f"  precision  {pre:.4f}")
    print(f"  F1         {2*pre*rec/(pre+rec):.4f}")
    print(f"  accuracy   {(tp+tn)/len(p):.4f}")
    print(f"\n  per-PIXEL model, for comparison: AUC 0.6950  recall 0.5396  "
          f"precision 0.1010")
    print(f"  moving to catchment scale changes AUC by {auc-0.6950:+.4f}")

    imp = sorted(zip(names, clf.feature_importances_), key=lambda x: -x[1])[:10]
    print("\n  top features:")
    for n_, v in imp:
        print(f"    {n_:<34}{v:.4f}")
    res["importance"] = [(n_, float(v)) for n_, v in imp]
    res["n_tiles"] = int(len(meta))
    res["n_test"] = int(te.sum())

    tagf = "tile_level" + (f"_no_{args.drop.strip(',').replace(',','_')}"
                           if args.drop else "")
    json.dump(res, (ROOT / f"data/{tagf}.json").open("w"), indent=1)
    print(f"\nwrote data/{tagf}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
