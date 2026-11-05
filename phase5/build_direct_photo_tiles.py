#!/usr/bin/env python3
"""
The one thing not yet tried tonight: train directly on the real photos, no
correction, no guessing, no rule applied to the storms that lack one. Only
the 210 events (2,099 tiles) that have a real Sentinel-2 photo are kept, with
the photo's own answer as the label. Everything else about the pipeline is
identical to the raw-label headline.

Quiet-control tiles are kept AS-IS (still their original coh_drop label),
since they never make a positive claim, they were already independently
validated by the flood-vs-quiet check, and no photo exists for a quiet date
to check them against anyway.

Output: data/tiles_direct/, data/tiles_direct_neg/ (= tiles_neg_v4, copied
        unchanged, kept separate only so the run has its own self-contained
        folder pair)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", choices=["photo", "insar"], default="photo",
                    help="insar keeps the coherence-drop label on the exact same "
                         "tiles and pixels: a control for the photo run")
    args = ap.parse_args()
    src = ROOT / "data/tiles_v4"
    dst = ROOT / ("data/tiles_direct" if args.label == "photo" else "data/tiles_direct_insar")
    opt_dir = ROOT / "data/optical_labels"
    opt_events = {p.stem for p in opt_dir.glob("*.tif")}
    if args.label == "insar":
        # pin to the photo run's events, not labels the photo build added since
        opt_events &= set(pd.read_csv(ROOT / "data/tiles_direct/index.csv").event.astype(str))

    idx = pd.read_csv(src / "index.csv")
    keep_mask = idx.event.astype(str).isin(opt_events)
    keep_pos = np.flatnonzero(keep_mask.values)
    print(f"{len(idx)} tiles total, {len(keep_pos)} belong to an optically "
          f"labeled event, keeping those only", flush=True)

    dst.mkdir(parents=True, exist_ok=True)
    (dst / "shared").mkdir(exist_ok=True)
    for r in (1, 2, 5, 10, 30):
        (dst / f"rung{r}m").mkdir(exist_ok=True)
    shutil.copy(src / "channels.json", dst / "channels.json")

    cache: dict[str, tuple] = {}
    new_rows = []
    SHARD_TILES = 512
    buf_xs, buf_y, buf_m = [], [], []
    buf_xt = {r: [] for r in (1, 2, 5, 10, 30)}
    shard_n = 0

    def flush():
        nonlocal shard_n, buf_xs, buf_y, buf_m, buf_xt
        if not buf_xs:
            return
        np.savez_compressed(dst / "shared" / f"shard-{shard_n:05d}.npz",
                            xs=np.stack(buf_xs), y=np.stack(buf_y), m=np.stack(buf_m))
        for r in (1, 2, 5, 10, 30):
            np.savez_compressed(dst / f"rung{r}m" / f"shard-{shard_n:05d}.npz",
                                xt=np.stack(buf_xt[r]))
        shard_n += 1
        buf_xs, buf_y, buf_m = [], [], []
        for r in (1, 2, 5, 10, 30):
            buf_xt[r] = []

    # walk the existing shards in order, tracking a running position so we
    # can match global tile index i (from idx.csv) to (shard file, local row)
    sh_files = sorted((src / "shared").glob("shard-*.npz"))
    rung_files = {r: sorted((src / f"rung{r}m").glob("shard-*.npz")) for r in (1, 2, 5, 10, 30)}
    pos = 0
    n_dropped_novalid = 0
    for k, p in enumerate(sh_files):
        z = np.load(p)
        xs_all, y_all, m_all = z["xs"], z["y"], z["m"]
        rt_all = {r: np.load(rung_files[r][k])["xt"] for r in (1, 2, 5, 10, 30)}
        T = y_all.shape[0]
        for t in range(T):
            i = pos + t
            if not keep_mask.iloc[i]:
                continue
            ev = str(idx.event.iloc[i])
            if ev not in cache:
                with rasterio.open(opt_dir / f"{ev}.tif") as s:
                    cache[ev] = s.read()
            flood_o, _, valid_o = cache[ev]
            r0, c0 = int(idx.row.iloc[i]), int(idx.col.iloc[i])
            fo = flood_o[r0:r0 + 112, c0:c0 + 112]
            vo = valid_o[r0:r0 + 112, c0:c0 + 112]
            new_m = m_all[t] & (vo > 0.5) & np.isfinite(fo)
            if new_m.sum() < 500:
                n_dropped_novalid += 1
                continue
            new_y = (y_all[t] if args.label == "insar"
                     else np.where(new_m & (fo > 0.5), 1.0, -1.0).astype("float32"))
            buf_xs.append(xs_all[t]); buf_y.append(new_y); buf_m.append(new_m)
            for r in (1, 2, 5, 10, 30):
                buf_xt[r].append(rt_all[r][t])
            new_rows.append({"event": idx.event.iloc[i], "huc4": idx.huc4.iloc[i],
                             "huc8": idx.huc8.iloc[i], "row": idx.row.iloc[i],
                             "col": idx.col.iloc[i],
                             "valid_frac": float(new_m.mean()),
                             "flood_frac_0.15": float((fo[new_m] > 0.5).mean()),
                             "has_flood": bool((fo[new_m] > 0.5).any()),
                             "dem_clean": bool(idx.dem_clean.iloc[i])})
            if len(buf_xs) >= SHARD_TILES:
                flush()
        pos += T
        print(f"  processed shard {k+1}/{len(sh_files)}, kept {len(new_rows)} "
              f"tiles so far", flush=True)
    flush()

    pd.DataFrame(new_rows).to_csv(dst / "index.csv", index=False)
    print(f"\nfinal tile count: {len(new_rows)} "
          f"(dropped {n_dropped_novalid} for insufficient overlap with the photo)")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
