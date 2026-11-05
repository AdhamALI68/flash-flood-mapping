#!/usr/bin/env python3
"""
Phase 6 - XGBoost baseline. Did we need a U-Net at all?

    ./.venv/bin/python -u phase6/xgb_baseline.py

WHY THIS EXISTS
    Every comparison so far has been the U-Net against single input channels.
    A reviewer will ask whether a gradient-boosted tree on the same features
    does just as well, and right now we cannot answer. This answers it.

BUILT FROM THE TEAM'S OWN NOTEBOOK, WITH THREE CORRECTIONS
    The feature engineering is taken from 14_validation_of_092_..., because
    trees cannot form ratios or products on their own and need them handed over:
        terrain_roughness   = elev_std / |elev_mean|
        local_depression    = elev_mean - elev_min
        twi_spi_interaction = basin_twi * basin_spi
    The protocol is ours, and differs in three ways that matter:

      1. THE TEST SET IS NOT FILTERED. The notebook drops every image under 15%
         flooded from train, valid AND test, aiming at a 50/50 balance. On our
         data that same filter moved precision 0.10 -> 0.55 while AUC fell
         0.695 -> 0.555: the model got worse and the score went up.

      2. NO TEST REGIONS ARE BORROWED FOR TRAINING. The notebook moves half the
         test regions into the training pool. Regions 1606 and 1809 appear here
         in test only.

      3. THE THRESHOLD IS CHOSEN ON VALIDATION. The notebook sweeps 500 cutoffs
         against y_test and keeps the best F1, which is choosing the answer
         after seeing the marks.

    With those three fixed, the number below is directly comparable to the
    U-Net's 0.6950 AUC on the same pixels.

Output: data/xgb_baseline.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_tiles, region_split                      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """The team's physics-informed interactions, on our channel names."""
    d = df.copy()
    if {"roughness_10m", "elev_mean_10m"} <= set(d.columns):
        d["terrain_roughness"] = d.roughness_10m / (d.elev_mean_10m.abs() + 1e-6)
    if {"elev_mean_10m", "elev_min_10m"} <= set(d.columns):
        d["local_depression"] = d.elev_mean_10m - d.elev_min_10m
    if {"basin_twi", "basin_spi"} <= set(d.columns):
        d["twi_spi_interaction"] = d.basin_twi * d.basin_spi
    if {"slope_10m", "microrelief_10m"} <= set(d.columns):
        d["slope_variability"] = d.microrelief_10m / (d.slope_10m.abs() + 1e-6)
    return d


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="10")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--only-local", action="store_true",
                    help="keep ONLY features computed from the pixel and its "
                         "immediate cell: elevation, slope, roughness, soil, "
                         "land cover, point rain. Drops every hydrologically "
                         "routed feature (basin_*, *_upstream_*, "
                         "catchment_area) so we can see how much of the score "
                         "comes from context that was computed for us rather "
                         "than learned")
    ap.add_argument("--tag", default="all")
    ap.add_argument("--sample", type=int, default=3_000_000,
                    help="pixels sampled per split; AUC is stable well below this")
    args = ap.parse_args()

    print("loading tiles...", flush=True)
    X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", args.rung)
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", args.rung)
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)
    tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])
    print(f"  train {int(tr.sum())} tiles, val {int(va.sum())}, "
          f"test {int(te.sum())} tiles", flush=True)

    rng = np.random.default_rng(0)

    def flatten(sel, cap):
        k = np.flatnonzero(sel)
        xs = np.concatenate([X[i][:, M[i]].T for i in k])       # pixels x channels
        ys = np.concatenate([(Y[i] > args.threshold)[M[i]] for i in k]).astype(int)
        if xs.shape[0] > cap:
            j = rng.choice(xs.shape[0], cap, replace=False)
            xs, ys = xs[j], ys[j]
        d = engineer(pd.DataFrame(xs, columns=names))
        if args.only_local:
            drop = [c for c in d.columns
                    if c.startswith("basin_") or "upstream" in c
                    or c.startswith("catchment_area") or "twi_spi" in c]
            d = d.drop(columns=drop)
        return d, ys

    Xtr, ytr = flatten(tr, args.sample)
    Xva, yva = flatten(va, args.sample // 3)
    Xte, yte = flatten(te, args.sample)
    print(f"  train {Xtr.shape[0]:,} px x {Xtr.shape[1]} features "
          f"({100*ytr.mean():.2f}% flooded)")
    print(f"  test  {Xte.shape[0]:,} px ({100*yte.mean():.2f}% flooded)\n",
          flush=True)

    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=2000, max_depth=8, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
        gamma=1, reg_alpha=0.1, reg_lambda=1.0, random_state=0,
        n_jobs=-1, tree_method="hist", eval_metric="aucpr",
        early_stopping_rounds=50)
    print("training XGBoost...", flush=True)
    clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=100)

    # Cutoff chosen on VALIDATION, then applied once to test.
    pv = clf.predict_proba(Xva)[:, 1]
    best_t, best_f1 = 0.5, 0.0
    for t in np.quantile(pv, np.linspace(0.50, 0.999, 120)):
        p = pv >= t
        tp = float((p & (yva == 1)).sum()); fp = float((p & (yva == 0)).sum())
        fn = float((~p & (yva == 1)).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1

    pt = clf.predict_proba(Xte)[:, 1]
    auc = roc_auc_score(yte, pt)
    apr = average_precision_score(yte, pt)
    p = pt >= best_t
    tp = float((p & (yte == 1)).sum()); fp = float((p & (yte == 0)).sum())
    fn = float((~p & (yte == 1)).sum()); tn = float((~p & (yte == 0)).sum())
    rec, pre = tp / (tp + fn), tp / (tp + fp)
    f1 = 2 * pre * rec / (pre + rec)

    print("\n" + "=" * 72)
    print("XGBOOST BASELINE - our protocol, unfiltered test, held-out regions")
    print("=" * 72)
    print(f"  cutoff {best_t:.4f} (chosen on validation, not test)")
    print(f"  AUC        {auc:.4f}")
    print(f"  AP         {apr:.4f}   (base rate {yte.mean():.4f})")
    print(f"  recall     {rec:.4f}")
    print(f"  precision  {pre:.4f}")
    print(f"  F1         {f1:.4f}")
    print(f"  accuracy   {(tp+tn)/(tp+fp+fn+tn):.4f}")
    print("\n  U-Net on the same pixels: AUC 0.6950  AP 0.0916  "
          "recall 0.5396  prec 0.1010")
    print(f"  difference: AUC {auc-0.6950:+.4f}   AP {apr-0.0916:+.4f}")
    print("  VERDICT: " + ("XGBoost matches or beats the U-Net - the network is "
                           "not earning its place" if auc >= 0.6950 - 0.005
                           else "the U-Net is ahead; the architecture is justified"))

    imp = sorted(zip(Xtr.columns, clf.feature_importances_),
                 key=lambda x: -x[1])[:15]
    print("\n  top features by gain:")
    for n_, v in imp:
        star = "  <- engineered" if n_ in ("terrain_roughness", "local_depression",
                                           "twi_spi_interaction",
                                           "slope_variability") else ""
        print(f"    {n_:<34}{v:.4f}{star}")

    json.dump({"auc": float(auc), "ap": float(apr), "recall": float(rec),
               "precision": float(pre), "f1": float(f1), "cutoff": best_t,
               "base_rate": float(yte.mean()),
               "unet": {"auc": 0.6950, "ap": 0.0916},
               "importance": [(n_, float(v)) for n_, v in imp]},
              (ROOT / f"data/xgb_baseline_{args.tag}.json").open("w"),
              indent=1)
    print(f"\nwrote data/xgb_baseline_{args.tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
