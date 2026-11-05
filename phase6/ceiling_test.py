#!/usr/bin/env python3
"""
Is the limit the MODEL or the LABEL? The capacity test settles it.

THE LOGIC
    Every result so far is on held-out regions, so a low score could mean either
    "the label is unpredictable" or "the model does not generalise". Those are
    very different problems and we have been arguing about which one it is.

    So: fit a high-capacity model to the test pixels THEMSELVES and score it on
    the same pixels. This is deliberate overfitting - it is not a result, it is
    a diagnostic. It measures how much of the label is explainable by these 30
    features at all, with generalisation removed from the question.

  If the score stays low   -> the label is not a function of these features.
                              Irreducible. No model fixes it.
  If the score goes high   -> the features DO explain the label, and our
                              problem is generalisation, which is fixable.

For comparison the same test is run on a SHUFFLED label, which is pure noise,
to show what a memorising model scores when there is genuinely nothing to learn.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_tiles, region_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

rng = np.random.default_rng(0)
k = np.flatnonzero(te)
xs = np.concatenate([X[i][:, M[i]].T for i in k])
ys = np.concatenate([(Y[i] > THR)[M[i]] for i in k]).astype(int)
good = np.isfinite(xs).all(axis=1)
xs, ys = xs[good], ys[good]
j = rng.choice(len(xs), min(2_000_000, len(xs)), replace=False)
xs, ys = xs[j], ys[j]
print(f"test pixels: {len(ys):,}   flooded {100*ys.mean():.2f}%\n", flush=True)

import xgboost as xgb
def fit_score(y, depth, trees, tag):
    clf = xgb.XGBClassifier(n_estimators=trees, max_depth=depth,
                            learning_rate=0.3, subsample=1.0,
                            colsample_bytree=1.0, min_child_weight=1,
                            reg_lambda=0.0, n_jobs=-1, tree_method="hist",
                            random_state=0)
    clf.fit(xs, y, verbose=False)
    a = roc_auc_score(y, clf.predict_proba(xs)[:, 1])
    print(f"  {tag:<42}{a:.4f}", flush=True)
    return a

print("=" * 64)
print("CAPACITY TEST - fit the test pixels, score the same pixels")
print("=" * 64)
res = {}
for d, t in ((4, 200), (8, 400), (12, 800), (16, 1200)):
    res[f"real_d{d}"] = fit_score(ys, d, t, f"real label, depth {d}, {t} trees")

ysh = ys.copy(); rng.shuffle(ysh)
print("\n  the same, on a SHUFFLED label (pure noise, nothing to learn):")
for d, t in ((8, 400), (16, 1200)):
    res[f"shuffled_d{d}"] = fit_score(ysh, d, t, f"shuffled label, depth {d}")

print("\n" + "-" * 64)
best_real = max(v for k_, v in res.items() if k_.startswith("real"))
best_shuf = max(v for k_, v in res.items() if k_.startswith("shuffled"))
print(f"  best on the real label     {best_real:.4f}")
print(f"  best on pure noise         {best_shuf:.4f}  (memorisation floor)")
print(f"  our held-out result        0.6950")
print()
if best_real > 0.90:
    print("  VERDICT: the 30 features CAN explain the label. Our problem is")
    print("           generalisation between regions, which is fixable.")
elif best_real > 0.80:
    print("  VERDICT: partly learnable. Some headroom exists above 0.695, but")
    print("           the label is not a clean function of these features.")
else:
    print("  VERDICT: even with memorisation allowed, these features cannot")
    print("           explain the label. The limit is the label itself, and no")
    print("           model or architecture will move it.")
json.dump(res, (ROOT / "data/ceiling_test.json").open("w"), indent=1)
