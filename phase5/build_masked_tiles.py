#!/usr/bin/env python3
"""
Third attempt at fixing the noisy label, and a different kind of fix from the
first two. Both earlier correctors tried to RELABEL pixels (decide a new
flood/dry answer using land cover and terrain). Both failed, for two different,
diagnosed reasons: the first leaned on land cover and terrain so heavily that
the main model just reconstructed its rule from its own inputs (circularity);
the second, kept deliberately narrow to avoid that, ended up too weak to say
anything useful.

This attempt does not relabel anything. It only EXCLUDES pixels we have
direct, measured evidence not to trust, exactly the same principle already
used elsewhere in this project (build_tiles.py excludes pixels with missing
amplitude data rather than guessing a label for them: "Pixels with no
amplitude are EXCLUDED, not called dry. Calling them dry would invent
negatives out of missing data.")

WHAT COUNTS AS "NOT TRUSTED" HERE
    Using the 210 events with a real optical photo, measure, for every land
    cover CLASS separately, the false-positive rate specifically among pixels
    the raw rule already calls "flood" (coh_drop > 0.15): of those, what
    fraction does the photo say were actually dry. Classes with a
    significantly higher false-positive rate than the overall average are
    excluded, but ONLY their flood-flagged pixels - a pixel already called
    dry in a risky class is not disputed by anything measured here, so it is
    left alone.

WHY THIS CANNOT CAUSE THE SAME CIRCULARITY
    Nothing is being predicted or relabeled. A masked pixel simply does not
    exist for training purposes, exactly like a cloud-covered pixel does not
    exist for the optical check. There is no new rule for the main model to
    reverse-engineer from its own inputs, because no new information was
    manufactured, information was only removed where it was shown unreliable.

Output: data/tiles_masked/, data/tiles_masked_neg/ (same shard format,
        the label "y" is UNCHANGED, only the mask "m" shrinks)
"""
from __future__ import annotations

import json, shutil
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_SAMPLE = 300           # a class needs at least this many raw-positive
                            # pixels in the check before its rate is trusted
FP_RATIO_CUTOFF = 1.5      # excluded if its false-positive rate is at least
                            # this many times the overall average


def load_shared_shards(d: Path):
    chan = json.loads((d / "channels.json").read_text())
    sc = chan["shared_channels"]
    idx = pd.read_csv(d / "index.csv")
    Xs, Y, M = [], [], []
    for p in sorted((d / "shared").glob("shard-*.npz")):
        z = np.load(p)
        Xs.append(z["xs"]); Y.append(z["y"]); M.append(z["m"])
    Xs = np.concatenate(Xs); Y = np.concatenate(Y); M = np.concatenate(M)
    if len(idx) != len(Xs):
        raise SystemExit(f"{d}: index has {len(idx)} rows, shards hold {len(Xs)}")
    return Xs, Y, M, idx, sc


def measure_false_positive_rates() -> set[int]:
    opt_dir = ROOT / "data/optical_labels"
    opt_events = {p.stem for p in opt_dir.glob("*.tif")}
    Xs, Y, M, idx, sc = load_shared_shards(ROOT / "data/tiles_v4")
    ci_class = sc.index("lc_class")

    import rasterio
    cache: dict[str, tuple] = {}
    cls_l, y_l, raw_pos_l = [], [], []
    for i in range(len(idx)):
        ev = str(idx.event.iloc[i])
        if ev not in opt_events:
            continue
        if ev not in cache:
            with rasterio.open(opt_dir / f"{ev}.tif") as s:
                cache[ev] = s.read()
        flood_o, _, valid_o = cache[ev]
        r0, c0 = int(idx.row.iloc[i]), int(idx.col.iloc[i])
        fo = flood_o[r0:r0 + 112, c0:c0 + 112]
        vo = valid_o[r0:r0 + 112, c0:c0 + 112]
        raw_pos = M[i] & (Y[i] > THR)
        mask = raw_pos & (vo > 0.5) & np.isfinite(fo)   # only where BOTH raw
                                                          # says flood AND we
                                                          # have a real photo
        if mask.sum() == 0:
            continue
        cls_l.append(Xs[i, ci_class][mask])
        y_l.append(fo[mask].astype(int))

    cls = np.concatenate(cls_l); y = np.concatenate(y_l)
    good = np.isfinite(cls)
    cls, y = cls[good], y[good]
    overall_fp = float((1 - y).mean())
    print(f"raw-positive pixels checked against a real photo: {len(y):,}")
    print(f"overall false-positive rate (raw says flood, photo says dry): "
          f"{100*overall_fp:.1f}%\n")

    print(f"{'class':>8}{'n':>10}{'false-pos rate':>16}{'ratio to avg':>14}"
          f"  excluded?")
    excluded = set()
    for c in sorted(set(cls.astype(int).tolist())):
        m = cls.astype(int) == c
        n = int(m.sum())
        if n < MIN_SAMPLE:
            continue
        fp = float((1 - y[m]).mean())
        ratio = fp / max(overall_fp, 1e-6)
        drop = ratio >= FP_RATIO_CUTOFF
        if drop:
            excluded.add(c)
        print(f"{c:>8}{n:>10,}{100*fp:>15.1f}%{ratio:>13.2f}x  "
              f"{'YES' if drop else ''}")
    print(f"\nexcluded classes: {sorted(excluded)}")
    return excluded


def process(src: Path, dst: Path, excluded: set[int]):
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "shared").mkdir(exist_ok=True)
    for r in (1, 2, 5, 10, 30):
        (dst / f"rung{r}m").mkdir(exist_ok=True)
    shutil.copy(src / "channels.json", dst / "channels.json")
    shutil.copy(src / "index.csv", dst / "index.csv")
    for r in (1, 2, 5, 10, 30):
        for p in sorted((src / f"rung{r}m").glob("shard-*.npz")):
            shutil.copy(p, dst / f"rung{r}m" / p.name)

    chan = json.loads((src / "channels.json").read_text())
    ci_class = chan["shared_channels"].index("lc_class")

    n_masked = n_tot = 0
    for p in sorted((src / "shared").glob("shard-*.npz")):
        z = np.load(p)
        xs, y, m = z["xs"], z["y"], z["m"].copy()
        for t in range(y.shape[0]):
            raw_pos = m[t] & (y[t] > THR)
            risky_class = np.isin(xs[t, ci_class], list(excluded))
            drop = raw_pos & risky_class
            n_masked += int(drop.sum()); n_tot += int(m[t].sum())
            m[t] = m[t] & ~drop
        np.savez_compressed(dst / "shared" / p.name, xs=xs, y=y, m=m)
        print(f"  {p.name}: done", flush=True)
    print(f"pixels excluded (untrustworthy positive calls): "
          f"{n_masked:,} / {n_tot:,} ({100*n_masked/max(n_tot,1):.2f}%)")


def main() -> int:
    print("measuring false-positive rate by land-cover class, "
          "using the 210 events with a real photo...\n")
    excluded = measure_false_positive_rates()

    print("\nbuilding data/tiles_masked (positive/flood tiles)...")
    process(ROOT / "data/tiles_v4", ROOT / "data/tiles_masked", excluded)
    print("\nbuilding data/tiles_masked_neg (negative/quiet tiles)...")
    process(ROOT / "data/tiles_neg_v4", ROOT / "data/tiles_masked_neg", excluded)

    json.dump({"excluded_lc_classes": sorted(excluded),
               "min_sample": MIN_SAMPLE, "fp_ratio_cutoff": FP_RATIO_CUTOFF},
              open(ROOT / "data/masked_tiles_meta.json", "w"), indent=1)
    print("\nwrote data/masked_tiles_meta.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
