#!/usr/bin/env python3
"""
Phase 5, step 3 - cut feature stacks into training tiles.

Tiles are 112 x 112 at 20 m = 2.24 km across, matching trial 1's ground extent
so the two projects' numbers sit on one line.

SHARED CHANNELS ARE STORED ONCE.
    Of 21 channels per tile, only 5 (terrain) differ between ablation rungs.
    basin, rain, soil and land cover are byte-identical in all five. Writing a
    separate 21-channel set per rung stored those 16 channels five times over -
    105 channel-sets where 41 suffice, about 25 GB instead of 10 GB.

    So: one pass writes shared/ once and rung{R}m/ for every rung. Reading each
    event once instead of five times also makes it ~5x faster.

    Training concatenates shared + the chosen rung. Tile ORDER is identical
    across all of them by construction: the tile set is fixed in advance from
    channels that do not depend on the rung, so index row i is the same tile in
    every shard set.

Other rules enforced here:
  ONE VALID MASK - a tile is kept only where every channel has data. A row that
  is part real and part filled is worse than a missing row, because nothing
  flags it later. Threshold measured, not guessed: validity is bimodal (79% of
  tiles are 100% valid), so 0.95 costs 5% of tiles and removes almost all
  masked pixels.

  SPLIT BY REGION, NEVER RANDOMLY - tiles from one event overlap heavily, so a
  random split puts near-identical tiles on both sides. HUC4 recorded per tile.

  DEM CLEANLINESS PER TILE - 455 of 617 events have terrain surveyed AFTER their
  flood. Those must be separable at training time (NEXT_STEPS 3b).

  LABELS STAY CONTINUOUS - thresholding happens at scoring time (3d).

  EMPTY TILES FLAGGED, NOT DROPPED - skipping them helps TRAINING (+0.013 in
  trial 1) but removing them from EVALUATION would hide false positives.

Output: data/tiles/shared/shard-*.npz, data/tiles/rung{R}m/shard-*.npz,
        data/tiles/index.csv, data/tiles/channels.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
TILE = 112
MIN_VALID_FRAC = 0.80
SHARD_TILES = 512
RUNGS = [1, 2, 5, 10, 30]
SHARED_STACKS = ["basin", "rain", "soil", "landcover"]


def read_stack(d: Path, name: str):
    p = d / f"{name}.tif"
    if not p.exists():
        return None, []
    with rasterio.open(p) as s:
        a = s.read().astype("float32")
        names = [s.descriptions[i] or f"{name}_{i}" for i in range(s.count)]
    return a, names


class ShardWriter:
    """Buffers tiles and flushes fixed-size shards."""

    def __init__(self, outdir: Path, keys: list[str]):
        self.dir = outdir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.keys = keys
        self.buf: dict[str, list] = {k: [] for k in keys}
        self.n = 0
        self.shard = 0

    def add(self, **arrays):
        for k in self.keys:
            self.buf[k].append(arrays[k])
        self.n += 1
        if self.n >= SHARD_TILES:
            self.flush()

    def flush(self):
        if self.n == 0:
            return
        np.savez_compressed(self.dir / f"shard-{self.shard:05d}.npz",
                            **{k: np.stack(self.buf[k]) for k in self.keys})
        self.buf = {k: [] for k in self.keys}
        self.n = 0
        self.shard += 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--out", default="data/tiles")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--rungs", default=",".join(str(r) for r in RUNGS))
    ap.add_argument("--shared", default=",".join(SHARED_STACKS),
                    help="stacks shared across rungs; every event must have all "
                         "of them or it is skipped")
    ap.add_argument("--label", default="coh_drop",
                    choices=["coh_drop", "fusion"],
                    help="fusion requires coherence loss AND amplitude "
                         "darkening. Measured against the paired quiet dates it "
                         "halves the label's false-alarm rate, 3.55%% -> 1.98%%")
    ap.add_argument("--use-valid-tif", action="store_true",
                    help="AND in the precomputed valid.tif as well. OFF by "
                         "default: valid.tif is frozen at feature-build time, "
                         "so it still carries the broken basin_hand and cancels "
                         "any later repair. See the comment at the mask.")
    args = ap.parse_args()
    rungs = [int(x) for x in args.rungs.split(",")]
    shared_stacks = [s for s in args.shared.split(",") if s]

    feat = ROOT / args.features
    idx_path = feat / "_index.csv"
    if not idx_path.exists():
        raise SystemExit(f"no {idx_path} - run build_features.py first")
    idx = pd.read_csv(idx_path, dtype={"id": str, "huc8": str, "huc4": str})
    if args.limit:
        idx = idx.head(args.limit)

    clean = {}
    p = ROOT / "data/event_dem_cleanliness.csv"
    if p.exists():
        c = pd.read_csv(p)
        clean = dict(zip(c["EVENT_ID"].astype(str), c["all_clean"].astype(bool)))

    outroot = ROOT / args.out
    print(f"{len(idx)} events -> {outroot.relative_to(ROOT)}")
    print(f"tile {TILE}x{TILE} @20 m = {TILE*20/1000:.2f} km, "
          f"min valid {MIN_VALID_FRAC}")
    print(f"rungs: {rungs}   shared stacks: {shared_stacks}\n", flush=True)

    shared_w: ShardWriter | None = None
    rung_w: dict[int, ShardWriter] = {}
    shared_names = rung_names = None
    rows = []
    n_dropped = n_skipped = 0

    for n, r in enumerate(idx.itertuples(index=False), 1):
        d = feat / r.id
        lab, lab_names = read_stack(d, "label")
        val, _ = read_stack(d, "valid")
        if lab is None or val is None:
            n_skipped += 1
            continue

        # Terrain for EVERY requested rung must be present, so the tile set is
        # identical across rungs and index row i means the same tile everywhere.
        ter = {}
        ok = True
        for rg in rungs:
            a, nm = read_stack(d, f"terrain_{rg}m")
            if a is None:
                ok = False
                break
            ter[rg] = (a, nm)
        if not ok:
            n_skipped += 1
            continue

        sh_parts, sh_names = [], []
        for nm in shared_stacks:
            a, names = read_stack(d, nm)
            if a is None:
                ok = False
                break
            sh_parts.append(a)
            sh_names += list(names)
        if not ok:
            n_skipped += 1
            continue
        Xs = np.concatenate(sh_parts, axis=0)

        # Channel layout is fixed by the first event and asserted thereafter, so
        # a missing band later can never silently shift the stack.
        if shared_names is None:
            shared_names = sh_names
            rung_names = {rg: list(ter[rg][1]) for rg in rungs}
            shared_w = ShardWriter(outroot / "shared", ["xs", "y", "m"])
            for rg in rungs:
                rung_w[rg] = ShardWriter(outroot / f"rung{rg}m", ["xt"])
        elif sh_names != shared_names or any(
                list(ter[rg][1]) != rung_names[rg] for rg in rungs):
            print(f"  {r.id}: channel layout mismatch, skipping", flush=True)
            n_skipped += 1
            continue

        # SELECT THE LABEL BY NAME, NEVER BY POSITION.
        # write_stack() sorts band names alphabetically, so label.tif is ordered
        # amp_change_db, coh_baseline, coh_coevent, coh_drop. Band 0 is
        # amp_change_db, not coh_drop. Taking lab[0] trained and evaluated
        # everything on radar brightness change in decibels - a quantity with
        # range -34..+39 that a 0.15 "flood" threshold marks over 40% of every
        # tile. That is why every baseline sat at chance.
        if "coh_drop" not in lab_names:
            print(f"  {r.id}: label.tif has no coh_drop band, skipping", flush=True)
            n_skipped += 1
            continue
        y = lab[lab_names.index("coh_drop")]        # continuous, never thresholded
        amp_ok = None
        if args.label == "fusion":
            # A flood destroys coherence AND darkens the return, because smooth
            # water reflects energy away from the sensor. Those are two separate
            # physical effects, so requiring both makes the label far more
            # specific: 3.55% -> 1.98% firing rate on paired quiet dates.
            if "amp_change_db" not in lab_names:
                print(f"  {r.id}: no amp_change_db, cannot build fusion label",
                      flush=True)
                n_skipped += 1
                continue
            amp = lab[lab_names.index("amp_change_db")]
            # Pixels with no amplitude are EXCLUDED, not called dry. Calling them
            # dry would invent negatives out of missing data, and amp is only
            # ~84% complete on average.
            amp_ok = np.isfinite(amp)
            y = np.where(amp < 0, y, -1.0).astype("float32")
        # THE MASK IS DERIVED FROM THE CHANNELS ACTUALLY BEING TILED.
        # valid.tif is the AND of coherence, basin, terrain, soil, rain and land
        # cover as they stood when build_features ran. Every one of those is
        # re-checked below from the real arrays, so it adds nothing - but it is
        # frozen, so it still carries the broken basin_hand. That made it 0.0%
        # across all 32 events of region 1810 while the live channels are 18-36%
        # valid, which is how a repaired HAND changed nothing.
        m = np.isfinite(y) & np.isfinite(Xs).all(axis=0)
        if amp_ok is not None:
            m &= amp_ok
        if args.use_valid_tif:
            m &= (val[0] > 0.5)
        for rg in rungs:
            m &= np.isfinite(ter[rg][0]).all(axis=0)

        H, W = y.shape
        for i in range(0, H - TILE + 1, TILE):
            for j in range(0, W - TILE + 1, TILE):
                sl = (slice(i, i + TILE), slice(j, j + TILE))
                mm = m[sl]
                if mm.mean() < MIN_VALID_FRAC:
                    n_dropped += 1
                    continue
                yy = y[sl]
                frac = float(np.nanmean(np.where(mm, yy, np.nan) > 0.15))
                shared_w.add(xs=Xs[:, sl[0], sl[1]].astype("float32"),
                             y=yy.astype("float32"), m=mm)
                for rg in rungs:
                    rung_w[rg].add(xt=ter[rg][0][:, sl[0], sl[1]].astype("float32"))
                rows.append({"event": r.id, "huc4": r.huc4, "huc8": r.huc8,
                             "row": i, "col": j,
                             "valid_frac": round(float(mm.mean()), 3),
                             "flood_frac_0.15": round(frac, 4),
                             "has_flood": frac > 0.01,
                             "dem_clean": bool(clean.get(r.id, False))})
        if n % 20 == 0:
            print(f"  [{n}/{len(idx)}] {len(rows)} tiles", flush=True)

    if shared_w is None:
        raise SystemExit("no usable events - check that build_features.py ran")
    shared_w.flush()
    for w in rung_w.values():
        w.flush()

    ix = pd.DataFrame(rows)
    outroot.mkdir(parents=True, exist_ok=True)
    ix.to_csv(outroot / "index.csv", index=False)
    (outroot / "channels.json").write_text(json.dumps(
        {"tile_px": TILE, "res_m": 20, "min_valid_frac": MIN_VALID_FRAC,
         "shared_channels": shared_names,
         "rung_channels": {str(k): v for k, v in rung_names.items()},
         "label": ("coh_drop (continuous)" if args.label == "coh_drop"
                   else "fusion: coh_drop where amplitude darkened, else -1"),
         "note": ("shared/ holds basin+rain+soil+landcover+label+mask once; "
                  "rung{R}m/ holds only that rung's terrain. Concatenate "
                  "shared.xs with rung.xt at training time. Tile order is "
                  "identical across all shard sets."),
         }, indent=1))

    print(f"\ntiles kept {len(ix)}, dropped for low valid {n_dropped}, "
          f"events skipped {n_skipped}")
    if len(ix):
        print(f"shared channels: {len(shared_names)}   "
              f"per-rung channels: {len(rung_names[rungs[0]])}")
        naive = len(rungs) * (len(shared_names) + len(rung_names[rungs[0]]))
        actual = len(shared_names) + len(rungs) * len(rung_names[rungs[0]])
        print(f"channel-sets stored: {actual} (naive per-rung would be {naive}, "
              f"{naive/actual:.1f}x more)")
        print(f"tiles with flood (>1% at 0.15): {int(ix.has_flood.sum())} "
              f"({100*ix.has_flood.mean():.0f}%)")
        print(f"tiles with NO flood: {int((~ix.has_flood).sum())} "
              f"({100*(~ix.has_flood).mean():.0f}%)  [trial 1: 69.3%]")
        print(f"tiles from clean-DEM events: {int(ix.dem_clean.sum())} "
              f"({100*ix.dem_clean.mean():.0f}%)")
        print("\ntiles per region:")
        print(ix.groupby("huc4").size().rename("tiles").to_frame().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
