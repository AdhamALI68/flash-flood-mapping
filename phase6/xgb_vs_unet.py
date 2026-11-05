#!/usr/bin/env python3
"""
XGBoost vs U-Net at MATCHED recall, so the comparison is not a cutoff argument.

Each model otherwise picks its own operating point: XGBoost landed at recall
0.77 / precision 0.084, the U-Net at 0.54 / 0.101. That looks like XGBoost wins
on recall and loses on precision, but it is the same trade-off seen from two
different cutoffs - their AUCs are 0.6954 and 0.6950.

Forcing both to the same recall makes the precision column mean something.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split
from xgb_baseline import engineer

ROOT = Path(__file__).resolve().parent.parent
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

# ---- U-Net predictions -------------------------------------------------
cols = pick_channels(names, ck["groups"])
m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
              prior=ck.get("prior", "downhill")).to(dev)
m.load_state_dict(ck["state"]); m.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
k = np.flatnonzero(te)
P = []
with torch.no_grad():
    for i in range(0, len(k), 32):
        xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
        P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0].cpu().numpy())
P = np.concatenate(P)
u_score = np.concatenate([P[j][M[i]] for j, i in enumerate(k)])
y = np.concatenate([(Y[i] > ck["threshold"])[M[i]] for i in k]).astype(int)

# ---- XGBoost on the same pixels ---------------------------------------
rng = np.random.default_rng(0)
def tab(sel, cap):
    kk = np.flatnonzero(sel)
    xs = np.concatenate([X[i][:, M[i]].T for i in kk])
    ys = np.concatenate([(Y[i] > ck["threshold"])[M[i]] for i in kk]).astype(int)
    good = np.isfinite(xs).all(axis=1)
    xs, ys = xs[good], ys[good]
    if len(xs) > cap:
        j = rng.choice(len(xs), cap, replace=False); xs, ys = xs[j], ys[j]
    return engineer(pd.DataFrame(xs, columns=names)), ys
Xtr, ytr = tab(tr, 2_000_000)
Xva, yva = tab(va, 700_000)
Xte_, yte_ = tab(te, 3_000_000)
import xgboost as xgb
clf = xgb.XGBClassifier(n_estimators=2000, max_depth=8, learning_rate=0.05,
                        subsample=0.8, colsample_bytree=0.7, min_child_weight=10,
                        gamma=1, reg_alpha=0.1, reg_lambda=1.0, random_state=0,
                        n_jobs=-1, tree_method="hist", eval_metric="aucpr",
                        early_stopping_rounds=50)
clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
x_score = clf.predict_proba(Xte_)[:, 1]

def at_recall(s, yy, target):
    o = np.argsort(-s)
    need = int(np.ceil(target * yy.sum()))
    cut = s[o][np.searchsorted(np.cumsum(yy[o]), need)]
    p = s >= cut
    tp = float((p & (yy == 1)).sum()); fp = float((p & (yy == 0)).sum())
    fn = float((~p & (yy == 1)).sum()); tn = float((~p & (yy == 0)).sum())
    return (tp/(tp+fn), tp/(tp+fp), (tp+tn)/len(p))

print("=" * 74)
print("XGBOOST vs U-NET AT MATCHED RECALL, per 20 m pixel, held-out regions")
print("=" * 74)
print(f"  U-Net   AUC {roc_auc_score(y, u_score):.4f}  AP {average_precision_score(y, u_score):.4f}")
print(f"  XGBoost AUC {roc_auc_score(yte_, x_score):.4f}  AP {average_precision_score(yte_, x_score):.4f}")
print(f"\n  {'recall':>8}{'U-Net prec':>13}{'XGB prec':>11}{'U-Net acc':>12}{'XGB acc':>10}   winner")
res = []
for t in (0.40, 0.50, 0.60, 0.70, 0.80, 0.90):
    ru, pu, au = at_recall(u_score, y, t)
    rx, px, ax = at_recall(x_score, yte_, t)
    w = "XGBoost" if px > pu else "U-Net"
    res.append({"recall": t, "unet_prec": pu, "xgb_prec": px,
                "unet_acc": au, "xgb_acc": ax, "winner": w})
    print(f"  {t:>8.2f}{pu:>13.4f}{px:>11.4f}{au:>12.4f}{ax:>10.4f}   {w}")
json.dump(res, (ROOT / "data/xgb_vs_unet.json").open("w"), indent=1)
print("\nwrote data/xgb_vs_unet.json")
