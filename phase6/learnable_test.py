#!/usr/bin/env python3
"""
Is the label learnable AT ALL? Split inside one region, so transfer is not the
question and memorisation is not possible.

WHY THE PREVIOUS TEST FAILED
    Fitting the test pixels and scoring the same pixels gave 0.99 - but a
    SHUFFLED label scored 0.84 the same way. With 2 M rows and deep trees the
    model simply memorises, so that design cannot separate signal from capacity.

THIS DESIGN
    Split the pixels of ONE region at random: fit on half, score on the other
    half. Same region, same events, same everything - only different pixels. So:

      - cross-region transfer is removed from the question
      - the evaluation pixels were never seen, so memorisation does not help

    A shuffled label is run through the identical pipeline as the control. On
    truly random labels this MUST score 0.50; anything above that would mean the
    test itself is broken.

  score high (>0.85)  -> the label IS predictable from these features within a
                         region. Our problem is generalisation, which is fixable.
  score ~0.70         -> the label carries real but limited information. Our
                         held-out 0.695 is close to the achievable limit.
  score ~0.50         -> the label is noise with respect to these features.
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
idx["huc4"] = idx.huc4.astype(str)
rng = np.random.default_rng(0)
import xgboost as xgb

print("within-region, SPLIT BY EVENT (no spatial leakage)")
print(f"  {'region':<10}{'pixels':>12}{'flooded':>9}{'within-region AUC':>20}"
      f"{'shuffled control':>19}")
res = {}
# ALL SIX basins, not a subset. The first run used three, which left the
# ceiling estimate resting on half the data - a stated caveat that is cheap to
# remove, so it is removed.
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
    # SPLIT BY EVENT, NOT BY PIXEL.
    # A random pixel split puts a pixel's own neighbours in training. Terrain is
    # smooth, so the model can win by recognising ground it has already seen 20 m
    # away - spatial leakage, not learning. Splitting by event keeps the region
    # fixed while making the evaluation pixels genuinely unseen.
    evs = idx.event.astype(str).values[sel]
    ev_px = np.concatenate([np.full(int(M[i].sum()), e)
                            for i, e in zip(sel, evs)])[ok][j]
    uniq = np.array(sorted(set(ev_px.tolist())))
    rng.shuffle(uniq)
    half = set(uniq[: len(uniq) // 2].tolist())
    a_ = np.flatnonzero(np.isin(ev_px, list(half)))
    b_ = np.flatnonzero(~np.isin(ev_px, list(half)))
    if len(a_) < 5000 or len(b_) < 5000:
        print(f"  {reg}: too few pixels after the event split"); continue

    def run(y):
        clf = xgb.XGBClassifier(n_estimators=600, max_depth=8,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.8, min_child_weight=10,
                                reg_lambda=1.0, n_jobs=-1, tree_method="hist",
                                random_state=0)
        clf.fit(xs[a_], y[a_], verbose=False)
        return roc_auc_score(y[b_], clf.predict_proba(xs[b_])[:, 1])

    real = run(ys)
    ysh = ys.copy(); rng.shuffle(ysh)
    shuf = run(ysh)
    res[reg] = {"auc": float(real), "shuffled": float(shuf),
                "n": int(len(ys)), "base": float(ys.mean())}
    print(f"  {reg:<10}{len(ys):>12,}{100*ys.mean():>8.2f}%{real:>20.4f}"
          f"{shuf:>19.4f}", flush=True)

print("\n" + "=" * 72)
m = np.mean([v["auc"] for v in res.values()])
print(f"  mean within-region AUC   {m:.4f}")
print(f"  our HELD-OUT result      0.6950")
print(f"  shuffled control         {np.mean([v['shuffled'] for v in res.values()]):.4f}"
      f"   (must be ~0.50 or the test is broken)")
print()
if m > 0.85:
    print("  VERDICT: the label IS predictable within a region. The gap to our")
    print("           held-out 0.695 is a GENERALISATION problem - fixable with")
    print("           more regions, domain adaptation, or per-region calibration.")
elif m > 0.75:
    print("  VERDICT: partly learnable. There is headroom above 0.695, but the")
    print("           label is not a clean function of these features.")
else:
    print("  VERDICT: even within one region, with no transfer required, these")
    print("           features cannot predict the label much better than we")
    print("           already do. The limit is the label.")
json.dump(res, (ROOT / "data/learnable_test.json").open("w"), indent=1)
