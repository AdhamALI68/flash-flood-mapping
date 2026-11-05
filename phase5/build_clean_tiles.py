#!/usr/bin/env python3
"""
Clean the label of EVERY tile (salt<15px 8-conn removed, holes<20px 4-conn
filled), write a new tile set with the cleaned label baked in.

The cleaned mask is stored as +1.0 (flood) / -1.0 (dry) in place of the
original continuous coh_drop, so it still passes the -1..+1 range check and
every downstream script (train.py thresholds at 0.15) reproduces exactly the
cleaned binary mask without any other code changing.

Output: data/tiles_clean/, data/tiles_clean_neg/  (same shard format as
tiles_v4 / tiles_neg_v4, only "y" is replaced)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path
import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_RED, MIN_HOLE = 15, 20   # overridden by --min-red / --min-hole if passed


def clean(mask: np.ndarray, min_red=None, min_hole=None) -> np.ndarray:
    min_red = MIN_RED if min_red is None else min_red
    min_hole = MIN_HOLE if min_hole is None else min_hole
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= min_red; keep[0] = False
        mask = keep[lab]
    dry = ~mask
    lab2, n2 = ndimage.label(dry, structure=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]))
    if n2 > 0:
        border = set(lab2[0, :]) | set(lab2[-1, :]) | set(lab2[:, 0]) | set(lab2[:, -1])
        sizes = np.bincount(lab2.ravel())
        fill = np.zeros(sizes.size, bool)
        for lb in range(1, sizes.size):
            if lb not in border and sizes[lb] < min_hole:
                fill[lb] = True
        mask = mask | fill[lab2]
    return mask


def process(src: Path, dst: Path, min_red=None, min_hole=None):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "shared").mkdir(exist_ok=True)
    for r in (1, 2, 5, 10, 30):
        (dst / f"rung{r}m").mkdir(exist_ok=True)
    shutil.copy(src / "channels.json", dst / "channels.json")
    shutil.copy(src / "index.csv", dst / "index.csv")
    for r in (1, 2, 5, 10, 30):
        for p in sorted((src / f"rung{r}m").glob("shard-*.npz")):
            shutil.copy(p, dst / f"rung{r}m" / p.name)

    n_flip = n_tot = 0
    for p in sorted((src / "shared").glob("shard-*.npz")):
        z = np.load(p)
        xs, y, m = z["xs"], z["y"].copy(), z["m"]
        for t in range(y.shape[0]):
            raw = np.where(m[t], y[t] > THR, False)
            cleaned = clean(raw, min_red, min_hole)
            n_flip += int((raw != cleaned).sum()); n_tot += int(m[t].sum())
            y[t] = np.where(cleaned, 1.0, -1.0).astype("float32")
        np.savez_compressed(dst / "shared" / p.name, xs=xs, y=y, m=m)
        print(f"  {p.name}: done", flush=True)
    print(f"pixels flipped by cleaning: {n_flip:,} / {n_tot:,} "
          f"({100*n_flip/max(n_tot,1):.2f}%)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pos", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--out-pos", default="data/tiles_clean")
    ap.add_argument("--out-neg", default="data/tiles_clean_neg")
    ap.add_argument("--min-red", type=int, default=MIN_RED)
    ap.add_argument("--min-hole", type=int, default=MIN_HOLE)
    args = ap.parse_args()
    print(f"cleaning with min_red={args.min_red} min_hole={args.min_hole}")
    print("cleaning positive (flood) tiles...")
    process(ROOT / args.pos, ROOT / args.out_pos, args.min_red, args.min_hole)
    print("cleaning negative (quiet) tiles...")
    process(ROOT / args.neg, ROOT / args.out_neg, args.min_red, args.min_hole)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
