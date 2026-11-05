#!/usr/bin/env python3
"""Draw one sample tile: the label on top, then all 30 input channels."""
from pathlib import Path
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
z = np.load(HERE / "tiles_sample.npz")
X, y, m, ids, names = z["X"], z["y"], z["mask"], z["tile_id"], list(z["channels"])
info = pd.read_csv(HERE / "tiles_sample_index.csv")
# the tile with the most rain among the training flood tiles in the sample
cand = info.index[(info.split == "train") & info.has_flood]
i = max(cand, key=lambda k: X[k, names.index("rain_24h_upstream_depth_mm")].mean())
r = info.iloc[i]

fig, ax = plt.subplots(7, 5, figsize=(13, 19), constrained_layout=True)
for a in ax.flat:
    a.set_xticks([]); a.set_yticks([])
    for s in a.spines.values():
        s.set_visible(False)
def show(a, img, title, sub, cmap, **kw):
    a.imshow(np.ma.masked_where(~m[i], img), cmap=cmap, interpolation="nearest", **kw)
    a.set_title(f"{title}\n{sub}", fontsize=9, color="#222222")
yy = y[i]
show(ax[0, 0], yy, "LABEL coh_drop", f"{np.nanmin(yy[m[i]]):.2f} to {np.nanmax(yy[m[i]]):.2f}", "RdBu_r", vmin=-.6, vmax=.6)
show(ax[0, 1], (yy > .15).astype(float), "LABEL flood (coh_drop > 0.15)", f"{r['flood_frac_0.15']*100:.1f}% flood", "Blues", vmin=0, vmax=1)
show(ax[0, 2], m[i].astype(float), "MASK", f"{m[i].mean()*100:.0f}% valid", "Greys", vmin=0, vmax=1)
for a in ax[0, 3:]:
    a.axis("off")
ax[0, 3].text(0, .5, f"Tile {r.tile_id}\nBasin group {r.huc4}, {r.split} split\n112 x 112 pixels, 20 m each\nEach panel has its own color scale,\nthe numbers show its min and max.\nValues are raw, before log1p and z-score.",
              fontsize=10, va="center", transform=ax[0, 3].transAxes, color="#222222")
for k, n in enumerate(names):
    v = X[i, k][m[i]]
    a = ax[1 + k // 5, k % 5]
    lo, hi = np.percentile(v, [1, 99])
    show(a, X[i, k], f"{k}  {n}", f"{v.min():.3g} to {v.max():.3g}", "Blues", vmin=lo, vmax=hi if hi > lo else lo + 1)
fig.savefig(HERE / "one_tile_all_channels.png", dpi=95, facecolor="white")
print("drew", r.tile_id)
