#!/usr/bin/env python3
"""
Phase 4 - does agriculture contaminate the coherence flood label?

Ploughing, harvesting and irrigation destroy interferometric coherence with no
water involved. If cropland tiles show a higher apparent flood rate, part of
what the label calls flood is farming.

This was first computed ad hoc and printed to a terminal, which left the number
untraceable. It is now a script with a saved result.

Output: data/cropland_check.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from train import load_tiles
THR = 0.15

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
ci = names.index("lc_cropland")

rows = []
for i in range(len(X)):
    m = M[i]
    if m.sum() < 2000:
        continue
    rows.append({"crop": float(X[i][ci][m].mean()),
                 "ff": float((Y[i][m] > THR).mean()),
                 "huc4": str(idx.huc4.iloc[i])})
d = pd.DataFrame(rows)
d["bin"] = pd.cut(d.crop, [-.001, .001, .05, .2, 1.01],
                  labels=["none", "under 5%", "5 to 20%", "over 20%"])
by_bin = d.groupby("bin", observed=True).agg(tiles=("ff", "size"),
                                             mean_flood=("ff", "mean"))
rho = float(stats.spearmanr(d.crop, d.ff).statistic)

print(f"tiles: {len(d)}")
print(f"tiles with any cropland: {100*(d.crop>0.01).mean():.1f}%\n")
print(by_bin.round(4).to_string())
print(f"\nSpearman, cropland vs flood fraction: {rho:+.4f}")
print("\nby basin:")
per = {}
for r_, g in d.groupby("huc4"):
    s_ = float(stats.spearmanr(g.crop, g.ff).statistic) if g.crop.std() > 0 else float("nan")
    per[r_] = {"tiles": int(len(g)), "pct_with_crop": float((g.crop > 0.01).mean()),
               "rho": s_}
    print(f"  {r_}  tiles {len(g):>5}  with cropland {100*(g.crop>0.01).mean():>5.1f}%"
          f"  rho {s_:+.3f}")
json.dump({"spearman_overall": rho,
           "by_bin": {str(k): {"tiles": int(v.tiles), "mean_flood": float(v.mean_flood)}
                      for k, v in by_bin.iterrows()},
           "by_basin": per, "n_tiles": int(len(d))},
          (ROOT / "data/cropland_check.json").open("w"), indent=1)
print("\nwrote data/cropland_check.json")
