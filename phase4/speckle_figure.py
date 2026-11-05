#!/usr/bin/env python3
"""
Show it, don't just measure it: raw label vs cleaned label vs real photo truth.

Same cleaning as speckle_test2.py: remove red (flood) blobs under 15 pixels
(8-connectivity), fill white (dry) holes under 20 pixels (4-connectivity).
Picked events that have BOTH real optical flood coverage and a visible amount
of scattered noise, so the before/after is actually visible.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, rasterio, torch, pandas as pd
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_RED, MIN_HOLE = 15, 20
opt_dir = ROOT / "data/optical_labels"


def clean(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= MIN_RED; keep[0] = False
        mask = keep[lab]
    dry = ~mask
    lab2, n2 = ndimage.label(dry, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n2 > 0:
        border = set(lab2[0,:]) | set(lab2[-1,:]) | set(lab2[:,0]) | set(lab2[:,-1])
        sizes = np.bincount(lab2.ravel())
        fill = np.zeros(sizes.size, bool)
        for lb in range(1, sizes.size):
            if lb not in border and sizes[lb] < MIN_HOLE:
                fill[lb] = True
        mask = mask | fill[lab2]
    return mask


# pick events: optical coverage, some real flood, and visibly scattered label
cands = []
for p in sorted(opt_dir.glob("*.tif")):
    e = p.stem
    lab_p = ROOT / "data/features" / e / "label.tif"
    if not lab_p.exists():
        continue
    with rasterio.open(lab_p) as s:
        nm = [s.descriptions[i] for i in range(s.count)]
        drop = s.read(nm.index("coh_drop") + 1)
    with rasterio.open(p) as s:
        onm = [s.descriptions[i] for i in range(s.count)]
        of = s.read(onm.index("flood_optical") + 1) > 0.5
        ov = s.read(onm.index("valid") + 1) > 0.5
    m = np.isfinite(drop) & ov
    if m.sum() < 20000 or of[m].sum() < 20:
        continue
    base = np.where(m, drop > THR, False)
    frac = base[m].mean()
    if 0.03 < frac < 0.30:
        cands.append((e, base[m].sum(), of[m].sum()))
cands.sort(key=lambda x: -x[2])
picks = [c[0] for c in cands[:3]]
print("events chosen:", picks)

fig, ax = plt.subplots(len(picks), 3, figsize=(9.6, 3.15*len(picks)))
if len(picks) == 1: ax = ax[None, :]
for r, e in enumerate(picks):
    lab_p = ROOT / "data/features" / e / "label.tif"
    with rasterio.open(lab_p) as s:
        nm = [s.descriptions[i] for i in range(s.count)]
        drop = s.read(nm.index("coh_drop") + 1)
    with rasterio.open(opt_dir / f"{e}.tif") as s:
        onm = [s.descriptions[i] for i in range(s.count)]
        of = s.read(onm.index("flood_optical") + 1) > 0.5
        ov = s.read(onm.index("valid") + 1) > 0.5
    m = np.isfinite(drop) & ov
    base = np.where(m, drop > THR, False)
    cleaned = clean(base)
    removed = base & ~cleaned
    added = cleaned & ~base

    cls_before = np.where(m, base.astype(float), np.nan)
    cls_after = np.zeros(base.shape); cls_after[cleaned] = 1
    cls_after[removed] = 0.4          # show what got erased, in a mid tone
    cls_after = np.where(m, cls_after, np.nan)

    for c, (arr, ttl, cm, vr) in enumerate((
            (cls_before, "RAW label (before cleaning)", "Reds", (0, 1)),
            (cls_after, "CLEANED (red=kept flood,\norange=erased as noise)",
             "Reds", (0, 1)),
            (of.astype(float), "Independent optical truth\n(Sentinel-2, ground check)",
             "Blues", (0, 1)))):
        a_ = ax[r, c]
        a_.imshow(np.where(m, arr, np.nan) if c < 2 else np.where(ov, arr, np.nan),
                  cmap=cm, vmin=vr[0], vmax=vr[1], interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        if r == 0: a_.set_title(ttl, fontsize=9.5)
    ax[r, 0].set_ylabel(f"event {e}", fontsize=9)
    removed_pct = 100 * removed[m].sum() / max(base[m].sum(), 1)
    ax[r, 1].set_xlabel(f"{removed_pct:.0f}% of flood pixels erased as \"noise\"",
                        fontsize=8.5)

fig.suptitle("Does removing small red blobs make the label match the real photo "
             "better?\n(measured answer: no — agreement fell 0.615 -> 0.611)",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = ROOT / "data/report/speckle_cleaning_check.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
