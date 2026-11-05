#!/usr/bin/env python3
"""
Is the model reading the flood's own scar? NEXT_STEPS 3b, finally run.

THE RISK
    74% of our tiles use terrain from lidar flown AFTER the flood. A flood cuts
    channels, strips vegetation and moves sediment, so post-flood terrain
    contains a record of where the water went. A model given that terrain could
    score well by reading the scar rather than by predicting the flow.

    That is leakage in our favour, and it would make every number in this
    project too high.

THE TEST
    Split the held-out test tiles by whether their DEM predates the flood, and
    score the same model on each. If performance holds on the clean tiles, the
    model is predicting. If it collapses, part of what we measured was scour.

    Nothing is retrained. Same model, same weights, two subsets.
"""
from __future__ import annotations
import sys, json
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
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, ["1606", "1809"], ["1507"])

cols = pick_channels(names, ck["groups"])
m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
              prior=ck.get("prior", "downhill")).to(dev)
m.load_state_dict(ck["state"]); m.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

Xte, Yte, Mte = X[te], Y[te], M[te]
ixte = idx[te].reset_index(drop=True)
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0].cpu().numpy())
P = np.concatenate(P)
lab = Yte > ck["threshold"]
clean = ixte.dem_clean.fillna(False).astype(bool).values

def score(sel):
    k = np.flatnonzero(sel)
    if len(k) < 20: return None
    s = np.concatenate([P[i][Mte[i]] for i in k])
    y = np.concatenate([lab[i][Mte[i]] for i in k]).astype(int)
    if y.sum() < 50 or (~y.astype(bool)).sum() < 50: return None
    auc = roc_auc_score(y, s); ap = average_precision_score(y, s)
    best = (0, 0, 0)
    for t in np.quantile(s, np.linspace(.5, .999, 60)):
        p = s >= t
        tp=float((p&y.astype(bool)).sum()); fp=float((p&~y.astype(bool)).sum())
        fn=float((~p&y.astype(bool)).sum())
        if tp==0: continue
        pr,rc = tp/(tp+fp), tp/(tp+fn); f1=2*pr*rc/(pr+rc)
        if f1>best[0]: best=(f1,pr,rc)
    return dict(tiles=len(k), pixels=y.size, base=y.mean(), auc=auc, ap=ap,
                recall=best[2], prec=best[1], f1=best[0])

print(f"model: {TAG}")
print(f"test tiles {len(P)}   clean-DEM {int(clean.sum())} "
      f"({100*clean.mean():.0f}%)   post-flood DEM {int((~clean).sum())}\n")
print("=" * 88)
print("DOES THE MODEL STILL WORK WHERE THE TERRAIN PREDATES THE FLOOD?")
print("=" * 88)
print(f"  {'subset':<34}{'tiles':>7}{'flooded':>9}{'AUC':>8}{'AP':>8}"
      f"{'recall':>8}{'prec':>8}")
rows = {}
for nm, sel in (("ALL test tiles", np.ones(len(P), bool)),
                ("DEM predates flood (CLEAN)", clean),
                ("DEM postdates flood (dirty)", ~clean)):
    r = score(sel)
    rows[nm] = r
    if r is None:
        print(f"  {nm:<34}  too few"); continue
    print(f"  {nm:<34}{r['tiles']:>7}{100*r['base']:>8.1f}%{r['auc']:>8.4f}"
          f"{r['ap']:>8.4f}{r['recall']:>8.4f}{r['prec']:>8.4f}")

c, d = rows.get("DEM predates flood (CLEAN)"), rows.get("DEM postdates flood (dirty)")
if c and d:
    print(f"\n  AUC on clean terrain minus dirty: {c['auc']-d['auc']:+.4f}")
    if c["auc"] >= d["auc"] - 0.02:
        print("  VERDICT: performance HOLDS on terrain that predates the flood.")
        print("           The model is not living off post-flood scour.")
    else:
        print("  VERDICT: performance DROPS on clean terrain. Part of the")
        print("           headline number is the flood's own scar. Must be")
        print("           reported, and the clean subset becomes the real result.")
json.dump({k: v for k, v in rows.items() if v},
          (ROOT / "data/eval_dem_vintage.json").open("w"), indent=1, default=float)
