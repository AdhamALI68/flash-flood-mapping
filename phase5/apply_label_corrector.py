#!/usr/bin/env python3
"""
Apply the trained label corrector (train_label_corrector.py) to EVERY tile in
tiles_v4 / tiles_neg_v4, including the roughly 400 events that never got an
optical photo. No training data is lost: every event keeps a label, just a
corrected one.

The corrected label is stored the same way build_clean_tiles.py stores its
cleaned label: +1.0 (flood) / -1.0 (dry) in place of the continuous coh_drop,
so every downstream script (train.py, phase7_final.py) works unchanged.

Output: data/tiles_corrected/, data/tiles_corrected_neg/
"""
from __future__ import annotations
import json, shutil
from pathlib import Path

import numpy as np
import xgboost as xgb

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["coh_drop", "basin_hand", "basin_slope", "basin_twi",
            "basin_spi", "lc_class", "lc_cropland", "lc_water"]


def process(src: Path, dst: Path, clf, cutoff: float, ci: dict):
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
        T, H, W = y.shape
        for t in range(T):
            mk = m[t]
            if mk.sum() == 0:
                y[t] = -1.0
                continue
            cols = [y[t][mk]]
            for nm in FEATURES:
                if nm == "coh_drop":
                    continue
                cols.append(xs[t, ci[nm]][mk])
            F = np.stack(cols, axis=1)
            F = np.nan_to_num(F, nan=0.0)
            prob = clf.predict_proba(F)[:, 1]
            corrected = np.zeros((H, W), bool)
            corrected[mk] = prob >= cutoff
            raw = np.where(mk, y[t] > 0.15, False)
            n_flip += int((raw != corrected).sum()); n_tot += int(mk.sum())
            y[t] = np.where(corrected, 1.0, -1.0).astype("float32")
        np.savez_compressed(dst / "shared" / p.name, xs=xs, y=y, m=m)
        print(f"  {p.name}: done", flush=True)
    print(f"pixels flipped by the corrector vs the raw 0.15 rule: "
          f"{n_flip:,} / {n_tot:,} ({100 * n_flip / max(n_tot, 1):.2f}%)")


def main() -> int:
    result = json.loads((ROOT / "data/label_corrector_result.json").read_text())
    cutoff = result["cutoff"]
    print(f"applying corrector, decision cutoff {cutoff:.3f} "
          f"(chosen on held-out events during training)")

    clf = xgb.XGBClassifier()
    clf.load_model(str(ROOT / "data/label_corrector.json"))

    chan = json.loads((ROOT / "data/tiles_v4/channels.json").read_text())
    sc = chan["shared_channels"]
    ci = {n: sc.index(n) for n in FEATURES if n != "coh_drop"}

    print("correcting positive (flood) tiles...")
    process(ROOT / "data/tiles_v4", ROOT / "data/tiles_corrected", clf, cutoff, ci)
    print("correcting negative (quiet) tiles...")
    process(ROOT / "data/tiles_neg_v4", ROOT / "data/tiles_corrected_neg", clf, cutoff, ci)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
