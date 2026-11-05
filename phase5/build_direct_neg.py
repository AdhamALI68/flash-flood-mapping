#!/usr/bin/env python3
"""
Quiet-control tiles for the photo-label experiment, graded with the SAME ruler.

The first direct_photo run kept the quiet tiles' InSAR coherence-drop label.
That label marks 2.84% of quiet-day pixels as flood (noise), while the photo
label marks only 0.16% of flood-day pixels as flood. The model learned "quiet
days flood more" and scored quiet days higher than flood days (0.33).

A quiet date is chosen because no flood was reported, so here every valid pixel
is labelled no-flood. Inputs and masks are copied unchanged.

Output: data/tiles_direct_neg/
"""
from __future__ import annotations
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    src, dst = ROOT / "data/tiles_neg_v4", ROOT / "data/tiles_direct_neg"
    (dst / "shared").mkdir(parents=True, exist_ok=True)
    for d in src.glob("rung*m"):
        if not (dst / d.name).exists():
            shutil.copytree(d, dst / d.name)
    shutil.copy(src / "channels.json", dst / "channels.json")

    for p in sorted((src / "shared").glob("shard-*.npz")):
        z = np.load(p)
        y = np.full(z["y"].shape, -1.0, dtype="float32")
        np.savez_compressed(dst / "shared" / p.name, xs=z["xs"], y=y, m=z["m"])

    idx = pd.read_csv(src / "index.csv")
    idx["flood_frac_0.15"] = 0.0
    idx["has_flood"] = False
    idx.to_csv(dst / "index.csv", index=False)
    print(f"wrote {len(idx)} quiet tiles, all labelled no-flood, to {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
