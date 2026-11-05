#!/usr/bin/env python3
"""
Independent verification of the headline numbers.

Recomputes them through a DIFFERENT code path from phase7_final.py: separate
loading, separate metric implementations, sklearn where possible instead of our
own rank formula. If the two agree the number is arithmetically sound. If they
disagree, one of them is wrong and both need investigating.

This does not verify the EXPERIMENTAL DESIGN, only the arithmetic. Design
questions (is the split honest, is the cutoff fitted on the right set) are
listed separately in the output.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

THR = 0.15
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
dev = "cuda" if torch.cuda.is_available() else "cpu"
claimed = {r["row"]: r for r in json.load(open(ROOT / "data/FINAL.json"))}

ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_x_neg" if False else
                                 ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                prior=ck.get("prior", "downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

def run(sel):
    k = np.flatnonzero(sel); P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)
    s = np.concatenate([P[j][M[i]] for j, i in enumerate(k)])
    y = np.concatenate([(Y[i] > THR)[M[i]] for i in k]).astype(int)
    return s, y

sv, yv = run(va)
st, yt = run(te)

print("=" * 74)
print("INDEPENDENT RECOMPUTATION OF THE HEADLINE")
print("=" * 74)
ok = True

# 1. AUC via sklearn, not our rank formula
auc = roc_auc_score(yt, st)
c = claimed["model, 20 m"]["auc"]
print(f"  AUC            claimed {c:.4f}   recomputed {auc:.4f}   "
      f"{'OK' if abs(auc-c) < 1e-4 else '*** MISMATCH ***'}")
ok &= abs(auc - c) < 1e-4

ap = average_precision_score(yt, st)
c = claimed["model, 20 m"]["ap"]
print(f"  AP             claimed {c:.4f}   recomputed {ap:.4f}   "
      f"{'OK' if abs(ap-c) < 1e-4 else '*** MISMATCH ***'}")
ok &= abs(ap - c) < 1e-4

# 2. reproduce the flagging-rate cutoff and the confusion matrix
rate, bf = 0.2, 0.0
for q in np.linspace(0.02, 0.60, 120):
    p = sv >= np.quantile(sv, 1-q)
    tp = float((p & (yv == 1)).sum()); fp = float((p & (yv == 0)).sum())
    fn = float((~p & (yv == 1)).sum())
    if tp == 0: continue
    pr, rc = tp/(tp+fp), tp/(tp+fn); f = 2*pr*rc/(pr+rc)
    if f > bf: rate, bf = float(q), f
cut = np.quantile(st, 1-rate)
tn, fp_, fn_, tp_ = confusion_matrix(yt, (st >= cut).astype(int)).ravel()
rec, pre = tp_/(tp_+fn_), tp_/(tp_+fp_)
acc = (tp_+tn)/len(yt)
for nm, val, key in (("recall", rec, "recall"), ("precision", pre, "precision"),
                     ("accuracy", acc, "accuracy")):
    c = claimed["model, 20 m"][key]
    good = abs(val - c) < 2e-3
    ok &= good
    print(f"  {nm:<14} claimed {c:.4f}   recomputed {val:.4f}   "
          f"{'OK' if good else '*** MISMATCH ***'}")
c = claimed["model, 20 m"]["false_alarms"]
print(f"  false alarms   claimed {c:,}   recomputed {int(fp_):,}   "
      f"{'OK' if abs(int(fp_)-c) < 5000 else '*** MISMATCH ***'}")

# 3. sanity: the split really is disjoint
r_tr = set(idx.huc4.astype(str)[tr]); r_te = set(idx.huc4.astype(str)[te])
e_tr = set(idx.event.astype(str)[tr]); e_te = set(idx.event.astype(str)[te])
print(f"\n  train/test basin overlap : {sorted(r_tr & r_te) or 'none'}")
print(f"  train/test event overlap : {len(e_tr & e_te)}")
print(f"  cutoff fitted on         : validation basin only ({rate:.1%} flag rate)")
print(f"  test pixels              : {len(yt):,}   flooded {100*yt.mean():.2f}%")
print("\n" + ("ALL ARITHMETIC CHECKS PASSED" if ok else "*** SOMETHING DISAGREES ***"))
