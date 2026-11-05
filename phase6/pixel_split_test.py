#!/usr/bin/env python3
"""
The pixel-split control. Same region, random PIXEL split (not event split).

This reconstructs the original pixel-split test whose numbers only ever existed
as printed log lines (data/learnable.log), never saved to JSON, and whose code
was later overwritten in learnable_test.py when it was changed to split by
event instead. That is a real traceability gap: the single most load-bearing
number in the split-comparison argument had no file behind it. This closes it,
on all six basins, saved to JSON like its event-split counterpart.

WHY THIS NUMBER IS EXPECTED TO BE HIGH, AND WHY THAT IS THE POINT
    Splitting pixels at random puts a pixel's own neighbours 20 m away in the
    training set. Terrain is smooth, so the model can score well by recognising
    ground it has already seen, without learning anything about water. This is
    the demonstration of that effect, not a claim about model skill.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from train import load_tiles
THR = 0.15

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
idx["huc4"] = idx.huc4.astype(str)
rng = np.random.default_rng(0)
import xgboost as xgb

print(f"  {'region':<10}{'pixels':>12}{'flooded':>9}{'pixel-split AUC':>18}"
      f"{'shuffled control':>19}")
res = {}
for reg in ["1501", "1503", "1507", "1606", "1809", "1810"]:
    sel = np.flatnonzero((idx.huc4 == reg).values)
    if len(sel) < 50:
        continue
    xs = np.concatenate([X[i][:, M[i]].T for i in sel])
    ys = np.concatenate([(Y[i] > THR)[M[i]] for i in sel]).astype(int)
    ok = np.isfinite(xs).all(axis=1)
    xs, ys = xs[ok], ys[ok]
    j = rng.choice(len(xs), min(1_200_000, len(xs)), replace=False)
    xs, ys = xs[j], ys[j]

    # RANDOM PIXEL split, on purpose: a pixel and its neighbour can land on
    # opposite sides. This is the leakage this test exists to demonstrate.
    n = len(ys) // 2
    perm = rng.permutation(len(ys))
    a_, b_ = perm[:n], perm[n:]

    def run(y):
        clf = xgb.XGBClassifier(n_estimators=600, max_depth=8,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=10,
                                reg_lambda=1.0, n_jobs=-1, tree_method="hist",
                                random_state=0)
        clf.fit(xs[a_], y[a_], verbose=False)
        from sklearn.metrics import roc_auc_score
        return roc_auc_score(y[b_], clf.predict_proba(xs[b_])[:, 1])

    real = run(ys)
    ysh = ys.copy(); rng.shuffle(ysh)
    shuf = run(ysh)
    res[reg] = {"auc": float(real), "shuffled": float(shuf),
               "n": int(len(ys)), "base": float(ys.mean())}
    print(f"  {reg:<10}{len(ys):>12,}{100*ys.mean():>8.2f}%{real:>18.4f}"
          f"{shuf:>19.4f}", flush=True)

m = np.mean([v["auc"] for v in res.values()])
sm = np.mean([v["shuffled"] for v in res.values()])
print("\n" + "=" * 74)
print(f"  mean PIXEL-split AUC, all six basins   {m:.4f}")
print(f"  mean shuffled control                  {sm:.4f}")
print(f"  (compare: event-split mean was 0.6748, our held-out result 0.6950)")
json.dump({"per_basin": res, "mean": float(m), "shuffled_mean": float(sm)},
          (ROOT / "data/pixel_split_test.json").open("w"), indent=1)
print("\nwrote data/pixel_split_test.json")
