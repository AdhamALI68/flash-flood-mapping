#!/usr/bin/env python3
"""
Test-time flip augmentation on the real headline model (raw labels, honest
region split). No retraining: average the prediction for a tile with its
horizontal flip, vertical flip, and 180-degree rotation (each unflipped before
averaging). Cheap, and a legitimate way to squeeze noise out of a single model.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
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

cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                prior=ck.get("prior", "downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

FLIPS = [((), ()), ((2,), (1,)), ((3,), (2,)), ((2, 3), (1, 2))]   # (xb dims, pf dims)


def predict(sel, tta):
    k = np.flatnonzero(sel); P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i + 32]][:, cols]).to(dev)
            xb = torch.nan_to_num((xb - mu) / sd)
            if not tta:
                p = torch.sigmoid(net(xb))[:, 0]
            else:
                acc = torch.zeros(xb.shape[0], xb.shape[2], xb.shape[3], device=dev)
                for xdims, pdims in FLIPS:
                    xf = torch.flip(xb, xdims) if xdims else xb
                    pf = torch.sigmoid(net(xf))[:, 0]
                    acc += torch.flip(pf, pdims) if pdims else pf
                p = acc / len(FLIPS)
            P.append(p.cpu().numpy())
    return np.concatenate(P), k


def score(tta):
    Pva, kva = predict(va, tta); Pte, kte = predict(te, tta)
    sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
    yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
    op, bf = 0.5, 0.0
    for t in np.quantile(sv, np.linspace(0.5, 0.999, 150)):
        p = sv >= t
        tp = float((p & yv).sum()); fp = float((p & ~yv).sum()); fn = float((~p & yv).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn); f = 2 * pr * rc / (pr + rc)
        if f > bf:
            op, bf = float(t), f
    st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])
    yt = np.concatenate([(Y[i] > THR)[M[i]] for i in kte]).astype(int)
    AUC = roc_auc_score(yt, st); AP = average_precision_score(yt, st)
    pred = st >= op
    tp = float((pred & yt).sum()); fp = float((pred & ~yt).sum())
    fn = float((~pred & yt).sum()); tn = float((~pred & ~yt).sum())
    R = tp / (tp + fn); Pr = tp / (tp + fp); ACC = (tp + tn) / len(yt)
    return dict(auc=AUC, ap=AP, recall=R, precision=Pr, accuracy=ACC, cutoff=op, n=len(yt))


print("scoring single-pass (no TTA)...")
base = score(tta=False)
print(f"  no TTA : AUC {base['auc']:.4f}  recall {base['recall']:.4f}  "
      f"precision {base['precision']:.4f}  accuracy {base['accuracy']:.4f}")

print("scoring with flip TTA (4-way average)...")
tta = score(tta=True)
print(f"  TTA    : AUC {tta['auc']:.4f}  recall {tta['recall']:.4f}  "
      f"precision {tta['precision']:.4f}  accuracy {tta['accuracy']:.4f}")

print(f"\n  delta AUC from TTA: {tta['auc']-base['auc']:+.4f}")

import json
json.dump({"no_tta": base, "tta": tta}, open(ROOT / "data/tta_test.json", "w"), indent=1)
