#!/usr/bin/env python3
"""
Phase 6 - the two numbers the paper needs: cross-region spread, and an interval.

    ./.venv/bin/python -u phase6/final_numbers.py

WHY
    Everything reported so far is a point estimate from one split. A reviewer
    will ask two things: does it hold in every region, and how precise is the
    number? Both are cheap and neither is an attempt to improve anything.

  1  CROSS-REGION, PER PIXEL
     Six leave-one-region-out models already exist, but they were scored on
     within-tile AUC. The paper reports per-pixel AUC, so each fold is
     re-scored on its own held-out region with the metric we actually quote.

  2  BOOTSTRAP INTERVAL
     Resampling by EVENT, not by pixel. Pixels inside one event are heavily
     dependent - 12,544 of them share a tile - so a pixel bootstrap would give
     an absurdly tight interval. Resampling events respects that structure.

Output: data/final_numbers.json
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
CFG = "terrain+basin+rain+rain_routed+soil+landcover"
REGIONS = ["1501", "1503", "1507", "1606", "1809", "1810"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def predict(ck, X, Y, M, sel, names):
    cols = pick_channels(names, ck["groups"])
    m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
    k = np.flatnonzero(sel)
    P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0]
                     .cpu().numpy())
    return np.concatenate(P), k


print("loading tiles...", flush=True)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
           np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
idx["huc4"] = idx.huc4.astype(str)
idx["event"] = idx.event.astype(str)

res = {"folds": []}

# ---- 1. cross-region, per pixel ----------------------------------------
print("\n" + "=" * 68)
print("CROSS-REGION, PER PIXEL (each fold on its own held-out region)")
print("=" * 68)
print(f"  {'held-out region':<18}{'tiles':>8}{'pixels':>13}{'AUC':>9}{'AP':>9}")
for reg in REGIONS:
    p = ROOT / f"data/model_loro_{reg}_{CFG}.pt"
    if not p.exists():
        print(f"  {reg:<18}   (no model)"); continue
    ck = torch.load(p, map_location=dev, weights_only=False)
    sel = (idx.huc4 == reg).values
    if sel.sum() < 20:
        continue
    P, k = predict(ck, X, Y, M, sel, names)
    s = np.concatenate([P[j][M[i]] for j, i in enumerate(k)])
    y = np.concatenate([(Y[i] > ck["threshold"])[M[i]] for i in k]).astype(int)
    if y.sum() < 50:
        continue
    auc = roc_auc_score(y, s); apr = average_precision_score(y, s)
    res["folds"].append({"region": reg, "tiles": int(sel.sum()),
                         "pixels": int(y.size), "auc": float(auc),
                         "ap": float(apr), "base": float(y.mean())})
    print(f"  {reg:<18}{int(sel.sum()):>8}{y.size:>13,}{auc:>9.4f}{apr:>9.4f}",
          flush=True)

if res["folds"]:
    a = np.array([f["auc"] for f in res["folds"]])
    print(f"\n  MEAN {a.mean():.4f}   SD {a.std(ddof=1):.4f}   "
          f"range {a.min():.4f}-{a.max():.4f}   n={len(a)} regions")
    res["cross_region"] = {"mean": float(a.mean()), "sd": float(a.std(ddof=1)),
                           "min": float(a.min()), "max": float(a.max()),
                           "n": len(a)}

# ---- 2. bootstrap interval on the headline split -----------------------
print("\n" + "=" * 68)
print("BOOTSTRAP INTERVAL, headline split (resampled by EVENT)")
print("=" * 68)
ck = torch.load(ROOT / f"data/model_{CFG}.pt", map_location=dev,
                weights_only=False)
_, _, te = region_split(idx, ["1606", "1809"], ["1507"])
P, k = predict(ck, X, Y, M, te, names)
ev = idx.event.values[k]
per = {}
for j, i in enumerate(k):
    e = ev[j]
    per.setdefault(e, [[], []])
    per[e][0].append(P[j][M[i]])
    per[e][1].append((Y[i] > ck["threshold"])[M[i]].astype(int))
per = {e: (np.concatenate(v[0]), np.concatenate(v[1])) for e, v in per.items()}
events = list(per)
rng = np.random.default_rng(0)
boots = []
for b in range(200):
    pick = rng.choice(len(events), len(events), replace=True)
    s = np.concatenate([per[events[i]][0] for i in pick])
    y = np.concatenate([per[events[i]][1] for i in pick])
    if y.sum() < 50 or (y == 0).sum() < 50:
        continue
    if s.size > 1_500_000:
        j = rng.choice(s.size, 1_500_000, replace=False)
        s, y = s[j], y[j]
    boots.append(roc_auc_score(y, s))
boots = np.array(boots)
lo, hi = np.percentile(boots, [2.5, 97.5])
s_all = np.concatenate([per[e][0] for e in events])
y_all = np.concatenate([per[e][1] for e in events])
point = roc_auc_score(y_all, s_all)
print(f"  events resampled : {len(events)}")
print(f"  point estimate   : {point:.4f}")
print(f"  95% interval     : {lo:.4f} - {hi:.4f}")
print(f"  bootstrap SD     : {boots.std():.4f}   ({len(boots)} resamples)")
res["bootstrap"] = {"point": float(point), "lo": float(lo), "hi": float(hi),
                    "sd": float(boots.std()), "n_events": len(events),
                    "n_resamples": int(len(boots))}
json.dump(res, (ROOT / "data/final_numbers.json").open("w"), indent=1)
print("\nwrote data/final_numbers.json")
