#!/usr/bin/env python3
"""
10 sample images: raw label vs cleaned label, side by side, for the eye check
requested before any training happens.
"""
from __future__ import annotations
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_RED, MIN_HOLE = 15, 20
OUT = ROOT / "handoff/before"


def clean(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel()); keep = sizes >= MIN_RED; keep[0] = False
        mask = keep[lab]
    dry = ~mask
    lab2, n2 = ndimage.label(dry, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n2 > 0:
        border = set(lab2[0,:])|set(lab2[-1,:])|set(lab2[:,0])|set(lab2[:,-1])
        sizes = np.bincount(lab2.ravel()); fill = np.zeros(sizes.size, bool)
        for lb in range(1, sizes.size):
            if lb not in border and sizes[lb] < MIN_HOLE: fill[lb] = True
        mask = mask | fill[lab2]
    return mask


idx = pd.read_csv(ROOT / "data/tiles_v4/index.csv")
idx = idx[idx.valid_frac > 0.9].sort_values("flood_frac_0.15", ascending=False)
picks_idx = list(idx.head(300).sample(10, random_state=1).index)

OUT.mkdir(parents=True, exist_ok=True)
sh_files = sorted((ROOT / "data/tiles_v4/shared").glob("shard-*.npz"))
SHARD = 512
made = 0
for row_i in picks_idx:
    shard_no, within = divmod(row_i, SHARD)
    if shard_no >= len(sh_files): continue
    z = np.load(sh_files[shard_no])
    y, m = z["y"][within], z["m"][within]
    raw = np.where(m, y > THR, False)
    cleaned = clean(raw)

    fig, ax = plt.subplots(1, 2, figsize=(6.6, 3.4))
    ax[0].imshow(np.where(m, raw.astype(float), np.nan), cmap="Reds", vmin=0, vmax=1)
    ax[0].set_title("Raw label", fontsize=10); ax[0].set_xticks([]); ax[0].set_yticks([])
    ax[1].imshow(np.where(m, cleaned.astype(float), np.nan), cmap="Reds", vmin=0, vmax=1)
    ax[1].set_title("Cleaned label", fontsize=10); ax[1].set_xticks([]); ax[1].set_yticks([])
    r = idx.loc[row_i]
    pct = 100*float((raw!=cleaned)[m].sum())/max(int(m.sum()),1)
    fig.suptitle(f"event {r.event}  region {r.huc4}  tile row {r.row} col {r.col}"
                 f"  ({pct:.1f}% pixels changed)", fontsize=9)
    fig.tight_layout(rect=[0,0,1,0.90])
    fig.savefig(OUT / f"sample_{made+1:02d}_event{r.event}.png", dpi=130)
    plt.close(fig)
    made += 1
    print(f"  {made}/10 done", flush=True)
print(f"wrote {made} images to {OUT}")
