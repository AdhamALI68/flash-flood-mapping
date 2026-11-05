#!/usr/bin/env python3
"""
What happens if we only score tiles that have a lot of flood in them?

THE TEMPTATION
    Precision is 0.10. Dropping tiles with little flood raises it immediately,
    and the model looks better. This measures how much of that rise is the model
    and how much is arithmetic.

WHY IT IS ARITHMETIC
    Precision depends on how common the positive class is. Keeping only tiles
    that are >=10% flooded raises the base rate, and a predictor that has not
    changed at all scores higher precision on a denser set. AUC is the control:
    it is invariant to the base rate, so if AUC does not move while precision
    does, nothing was learned - the test was just made easier.

    Selecting the TEST set using the label is also what a reviewer looks for
    first. Selecting the TRAINING set is a normal modelling choice and is a
    separate, legitimate experiment.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "terrain+basin+rain+rain_routed+soil+landcover"

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
           np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, ["1606", "1809"], ["1507"])

cols = pick_channels(names, ck["groups"])
model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
model.load_state_dict(ck["state"]); model.eval()
mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

Xte, Yte, Mte = X[te], Y[te], M[te]
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                 .cpu().numpy())
P = np.concatenate(P)
lab = Yte > ck["threshold"]
frac = np.array([lab[i][Mte[i]].mean() if Mte[i].sum() > 200 else np.nan
                 for i in range(len(P))])

print(f"model: {TAG}")
print(f"{len(P)} test tiles\n")
print("=" * 84)
print("KEEPING ONLY TILES ABOVE A FLOOD FRACTION  (test-set selection)")
print("=" * 84)
print(f"  {'keep tiles':<16}{'tiles':>7}{'pixels':>12}{'flooded':>9}"
      f"{'AUC':>8}{'recall':>8}{'prec':>8}{'F1':>8}")
for cut in (0.0, 0.02, 0.05, 0.10, 0.20, 0.30):
    keep = np.flatnonzero(np.isfinite(frac) & (frac >= cut))
    if len(keep) < 20:
        continue
    s = np.concatenate([P[i][Mte[i]] for i in keep])
    y = np.concatenate([lab[i][Mte[i]] for i in keep])
    if y.sum() < 50 or (~y).sum() < 50:
        continue
    auc = roc_auc_score(y, s)
    best = (0, 0, 0)
    for t in np.quantile(s, np.linspace(0.50, 0.999, 60)):
        p = s >= t
        tp = float((p & y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best[0]:
            best = (f1, pr, rc)
    print(f"  >= {100*cut:>4.0f}% flooded{len(keep):>7}{y.size:>12,}"
          f"{100*y.mean():>8.1f}%{auc:>8.4f}{best[2]:>8.4f}{best[1]:>8.4f}"
          f"{best[0]:>8.4f}")
print()
print("  AUC is the control: it does not depend on how common floods are.")
print("  If precision climbs while AUC stays flat, the test was made easier,")
print("  not the model better.")
