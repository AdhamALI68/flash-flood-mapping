#!/usr/bin/env python3
"""
Hand-label pilot, step 3: make the flood marks whole instead of spotty.

Spotty marks hurt both scores: a hole inside the real flood turns a correct
model guess into a "false alarm", and a stray dot is a "flood" no model can
predict. Three versions, all from the two photos only (no terrain layers,
which the model sees as inputs):

  strict   - the first version: clear brightening, specks removed
  filled   - grow each strict mark into the paler ground joined to it, then
             close small gaps and fill small holes
  squares  - the filled marks on 50 m squares: a square is flood when at
             least a third of it is marked

    ./.venv/bin/python -u phase7/fill_marks.py --event 983830 --after-index 2
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "handoff/hand_label_pilot"


def drop_small(mask, min_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    size = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(size >= min_px))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", required=True)
    ap.add_argument("--after-index", type=int, default=1)
    ap.add_argument("--strong", type=float, default=12.0)
    ap.add_argument("--weak", type=float, default=6.0)
    ap.add_argument("--block", type=int, default=5, help="square size in 10 m pixels")
    ap.add_argument("--block-frac", type=float, default=1 / 3)
    args = ap.parse_args()

    z = np.load(OUT / f"{args.event}_photos.npz")
    I = z["imgs"].astype("float32")
    pre, post = I[0].mean(-1), I[args.after_index].mean(-1)
    d = ndimage.gaussian_filter((post - np.median(post)) - (pre - np.median(pre)), 1.0)

    strict = np.load(OUT / f"{args.event}_mark10m.npy")

    # grow: keep paler ground only where it touches a strict mark
    weak = d > args.weak
    lab, _ = ndimage.label(weak)
    filled = np.isin(lab, np.unique(lab[strict & weak])) & weak
    filled |= strict
    filled = ndimage.binary_closing(filled, structure=np.ones((3, 3)), iterations=2)
    holes = ndimage.binary_fill_holes(filled) & ~filled
    filled |= drop_small(holes, 1) & ~drop_small(holes, 150)   # holes under 1.5 ha
    filled = drop_small(filled, 50)

    b = args.block
    H, W = (filled.shape[0] // b) * b, (filled.shape[1] // b) * b
    frac = filled[:H, :W].reshape(H // b, b, W // b, b).mean((1, 3))
    squares = np.kron(frac >= args.block_frac, np.ones((b, b), dtype=bool))
    squares = np.pad(squares, ((0, filled.shape[0] - H), (0, filled.shape[1] - W)))

    np.save(OUT / f"{args.event}_mark_filled10m.npy", filled)
    np.save(OUT / f"{args.event}_mark_squares{b*10}m.npy", frac >= args.block_frac)

    lo, hi = np.percentile(I, [1, 99])
    photo = (np.clip((I[args.after_index] - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
    red = np.array([235, 40, 20], "float32")
    for name, m in (("strict", strict), ("filled", filled), ("squares", squares)):
        o = photo.copy()
        o[m] = (0.25 * o[m] + 0.75 * red).astype("uint8")
        Image.fromarray(o).save(OUT / f"page/marks_{name}.jpg", quality=88)
        lab_, n_ = ndimage.label(m)
        holes_ = ndimage.binary_fill_holes(m) & ~m
        print(f"{name:8s} {100*m.mean():5.1f}% of square   pieces {n_:5d}   "
              f"hole area inside marks {100*holes_.sum()/max(m.sum(),1):5.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
