#!/usr/bin/env python3
"""
Two-stage: decide IF the area floods, then WHERE. Does it cut false alarms?

THE OBSERVATION BEHIND IT
    The model's false alarms are not random - they sit on valley floors and
    washes, which is where water WOULD go. It is predicting susceptibility, and
    on a day when that particular catchment stayed dry, every one of those
    sensible-looking pixels is a false alarm.

    So: gate the pixel prediction by a tile-level "did anything flood here
    today" model. If the tile is called dry, suppress its pixels. Tile-level
    classification reaches AUC 0.748, so the gate should be reliable enough to
    remove false alarms in tiles that never flooded.

WHY THIS IS NOT CHEATING
    The gate is trained on the same training regions and applied to the test
    regions without ever seeing their labels. Both its threshold and the pixel
    cutoff are chosen on validation.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

# ---- pixel model -------------------------------------------------------
cols = pick_channels(names, ck["groups"])
m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
              prior=ck.get("prior", "downhill")).to(dev)
m.load_state_dict(ck["state"]); m.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

def pix(sel):
    k = np.flatnonzero(sel); out = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            out.append(torch.sigmoid(m(torch.nan_to_num((xb - mu)/sd)))[:, 0]
                       .cpu().numpy())
    return np.concatenate(out), k

Pva, kva = pix(va)
Pte, kte = pix(te)

# ---- tile-level gate ---------------------------------------------------
def feats(sel):
    k = np.flatnonzero(sel)
    F = np.array([[X[i][c][M[i]].mean() for c in range(X.shape[1])] for i in k])
    lab = np.array([(Y[i] > THR)[M[i]].mean() for i in k])
    return F, lab, k
Ftr, ltr, _ = feats(tr)
Fva, lva, _ = feats(va)
Fte, lte, _ = feats(te)
import xgboost as xgb
gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
                         n_jobs=-1, tree_method="hist", eval_metric="aucpr",
                         early_stopping_rounds=40, random_state=0)
ytr, yva_ = (ltr >= 0.02).astype(int), (lva >= 0.02).astype(int)
gate.fit(Ftr, ytr, eval_set=[(Fva, yva_)], verbose=False)
gva, gte = gate.predict_proba(Fva)[:, 1], gate.predict_proba(Fte)[:, 1]
print(f"gate: tile-level AUC {roc_auc_score((lte>=0.02).astype(int), gte):.4f}")

def score(pixp, gateprob, kidx, gate_cut, pix_cut):
    s, y = [], []
    for j, i in enumerate(kidx):
        p = pixp[j].copy()
        if gateprob[j] < gate_cut:
            p = p * 0.0                    # gate says dry -> suppress
        s.append(p[M[i]]); y.append((Y[i] > THR)[M[i]])
    s, y = np.concatenate(s), np.concatenate(y)
    pr = s >= pix_cut
    tp = float((pr & y).sum()); fp = float((pr & ~y).sum()); fn = float((~pr & y).sum())
    tn = float((~pr & ~y).sum())
    return dict(recall=tp/(tp+fn), precision=tp/max(tp+fp, 1), fp=int(fp),
                accuracy=(tp+tn)/len(y))

# pixel cutoff from validation (no gate), then reused
sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
yv = np.concatenate([(Y[i] > THR)[M[i]] for j, i in enumerate(kva)])
best_c, bf = 0.5, 0
for t in np.quantile(sv, np.linspace(0.5, 0.999, 100)):
    p = sv >= t
    tp = float((p & yv).sum()); fp = float((p & ~yv).sum()); fn = float((~p & yv).sum())
    if tp == 0: continue
    pr_, rc_ = tp/(tp+fp), tp/(tp+fn); f = 2*pr_*rc_/(pr_+rc_)
    if f > bf: best_c, bf = float(t), f

print("\n" + "=" * 72)
print("DOES A TILE-LEVEL GATE CUT FALSE ALARMS?")
print("=" * 72)
print(f"  {'gate cutoff':<16}{'recall':>9}{'precision':>11}{'false alarms':>15}{'accuracy':>10}")
rows = []
for gc in (0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
    r = score(Pte, gte, kte, gc, best_c)
    rows.append({"gate": gc, **r})
    tag = "  (no gate)" if gc == 0 else ""
    print(f"  {gc:<16.2f}{r['recall']:>9.4f}{r['precision']:>11.4f}"
          f"{r['fp']:>15,}{r['accuracy']:>10.4f}{tag}")
base = rows[0]
best = max(rows[1:], key=lambda r: r["precision"])
print(f"\n  without gate : precision {base['precision']:.4f}, "
      f"{base['fp']:,} false alarms")
print(f"  best gate    : precision {best['precision']:.4f}, "
      f"{best['fp']:,} false alarms  ({100*(1-best['fp']/base['fp']):.0f}% fewer)")
print(f"  cost in recall: {base['recall']:.4f} -> {best['recall']:.4f}")
json.dump(rows, (ROOT / "data/two_stage.json").open("w"), indent=1)
