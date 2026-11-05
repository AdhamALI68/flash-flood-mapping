#!/usr/bin/env python3
"""
Does the MODEL beat simply reading the rain gauge, at the day-level question?

    "Will this catchment flood today, yes or no?"

WHY THIS CHECK MATTERS
    The model scores 0.7975 separating flood days from quiet days. That sounds
    good until you ask what a single number would do. Quiet control dates were
    chosen as days with no flood reported, and those are mostly dry days: 0.18 mm
    of rain against 15.6 mm on flood days. So a rain threshold alone might
    already answer this, and if it beats the model then the model adds nothing
    to the day-level question and we should not claim it does.

    Everything here is measured on the SAME held-out test regions and the SAME
    tiles, so the comparison is fair.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "terrain+basin+rain+rain_routed+soil+landcover"
TEST = ["1606", "1809"]

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
X, Y, M = np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn])
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, TEST, ["1507"])

cols = pick_channels(names, ck["groups"])
model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
model.load_state_dict(ck["state"]); model.eval()
mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

Xte, Mte, neg = X[te], M[te], is_neg[te]
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                 .cpu().numpy())
P = np.concatenate(P)

def tile_mean(arr3, msk):
    return np.array([arr3[i][msk[i]].mean() if msk[i].sum() > 200 else np.nan
                     for i in range(len(arr3))])

def sep(score):
    ok = np.isfinite(score)
    a, b = score[ok & ~neg], score[ok & neg]
    if len(a) < 10 or len(b) < 10:
        return np.nan, np.nan
    u = stats.mannwhitneyu(a, b, alternative="greater")
    return u.statistic / (len(a) * len(b)), u.pvalue

rows = [("THE MODEL (30 channels, U-Net)", *sep(tile_mean(P, Mte)))]
for c in ("rain_24h", "rain_1h_peak", "rain_24h_upstream_depth_mm",
          "rain_24h_upstream_vol_m3", "rain_1h_upstream_vol_m3",
          "elev_mean_10m", "basin_hand"):
    if c in names:
        rows.append((f"just the value of {c}",
                     *sep(tile_mean(Xte[:, names.index(c)], Mte))))

print(f"model: {TAG}")
print(f"test regions {TEST}: {int((~neg).sum())} flood-day tiles, "
      f"{int(neg.sum())} quiet-day tiles\n")
print("=" * 72)
print('"WILL THIS SQUARE FLOOD TODAY?"  - separation AUC, same tiles')
print("=" * 72)
for nm, a, p in rows:
    if np.isfinite(a):
        print(f"  {nm:<42}{a:>8.4f}   p={p:.2g}")
best = max((r for r in rows[1:] if np.isfinite(r[1])), key=lambda r: r[1])
m = rows[0][1]
print(f"\n  best single feature : {best[0]} = {best[1]:.4f}")
print(f"  the model           : {m:.4f}")
print(f"  VERDICT: the model {'BEATS' if m > best[1] else 'DOES NOT BEAT'} "
      f"the single best feature ({m - best[1]:+.4f})")
json.dump({"tag": TAG, "rows": [(n, float(a) if np.isfinite(a) else None)
                                for n, a, _ in rows]},
          (ROOT / "data/eval_day_level.json").open("w"), indent=1)
print("\nwrote data/eval_day_level.json")
