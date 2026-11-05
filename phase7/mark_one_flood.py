#!/usr/bin/env python3
"""
Hand-label pilot, step 2: mark flood traces from the before/after photos only,
then show them next to the radar label and the old water-photo label.

The mark uses ONLY the two photos (brightening that is local, not scene-wide,
cleaned of specks). No terrain, basin or land-cover layer is used, because the
model is given those as inputs and a label built from them would leak.

    ./.venv/bin/python -u phase7/mark_one_flood.py --event 983830 --after-index 2
"""
from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "handoff/hand_label_pilot"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--after-index", type=int, default=1)
    ap.add_argument("--change", type=float, default=12.0,
                    help="brightening, in 0-255 photo units after removing the "
                         "scene-wide shift, that counts as a fresh flood trace")
    ap.add_argument("--min-pixels", type=int, default=50, help="drop specks smaller than this (10 m pixels)")
    args = ap.parse_args()

    z = np.load(OUT / f"{args.event}_photos.npz")
    I = z["imgs"].astype("float32")
    pre, post = I[0].mean(-1), I[args.after_index].mean(-1)
    d = (post - np.median(post)) - (pre - np.median(pre))

    mark = d > args.change
    mark = ndimage.binary_opening(mark, iterations=1)
    lab, n = ndimage.label(mark)
    size = ndimage.sum(mark, lab, range(1, n + 1))
    mark = np.isin(lab, 1 + np.flatnonzero(size >= args.min_pixels))
    np.save(OUT / f"{args.event}_mark10m.npy", mark)

    with rasterio.open(ROOT / f"data/features/{args.event}/label.tif") as s:
        coh = s.read(4)
    radar = coh > 0.15
    with rasterio.open(ROOT / f"data/optical_labels/{args.event}.tif") as s:
        water_photo = s.read(1) > 0.5

    # compare on the 20 m label grid: a 20 m cell is flood if most of its 10 m pixels are
    mark20 = mark.reshape(mark.shape[0] // 2, 2, mark.shape[1] // 2, 2).mean((1, 3)) >= 0.5

    def agree(a, b):
        tp = (a & b).sum(); return tp / max(b.sum(), 1), tp / max(a.sum(), 1)

    rr, rp = agree(radar, mark20)
    wr, wp = agree(water_photo, mark20)
    print(f"marked flood: {100*mark20.mean():.1f}% of the area")
    print(f"radar label : {100*radar.mean():.1f}% of the area, finds {100*rr:.0f}% of marked flood, "
          f"{100*rp:.0f}% of its flood cells are marked flood")
    print(f"water photo : {100*water_photo.mean():.2f}% of the area, finds {100*wr:.0f}% of marked flood, "
          f"{100*wp:.0f}% of its flood cells are marked flood")

    lo, hi = np.percentile(I, [1, 99])
    photo = np.clip((I[args.after_index] - lo) / (hi - lo), 0, 1)
    fig, ax = plt.subplots(1, 4, figsize=(24, 6.6))
    ax[0].imshow(np.clip((I[0] - lo) / (hi - lo), 0, 1)); ax[0].set_title("BEFORE photo", fontsize=13)
    ax[1].imshow(photo); ax[1].imshow(np.ma.masked_where(~mark, mark), cmap="autumn", alpha=0.75)
    ax[1].set_title(f"AFTER photo + flood marks (red)  {100*mark20.mean():.0f}% of area", fontsize=13)
    ax[2].imshow(radar, cmap="Reds", vmin=0, vmax=1.3)
    ax[2].set_title(f"old RADAR label  {100*radar.mean():.0f}% of area", fontsize=13)
    ax[3].imshow(water_photo, cmap="Blues", vmin=0, vmax=1.0)
    ax[3].set_title(f"old WATER-photo label  {100*water_photo.mean():.2f}% of area", fontsize=13)
    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(f"event {args.event}, 10 km x 10 km", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUT / f"{args.event}_compare.png", dpi=90)
    print("wrote", OUT / f"{args.event}_compare.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
