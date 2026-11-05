#!/usr/bin/env python3
"""
Training tiles with the flood-trace label instead of the radar label.

Same tiles, same inputs and same order as data/tiles_v4. For each tile of a
flood that the marking kept, the label becomes the trace mark (+1 flood, -1
not flood) and the mask becomes the old mask AND the photos' clear ground.
Floods the marking skipped contribute no tiles. Tiles with under 500 usable
cells are dropped. Quiet-day tiles come from data/tiles_direct_neg (every
cell no-flood), built by phase5/build_direct_neg.py.

    ./.venv/bin/python -u phase7/build_trace_tiles.py --labels data/trace_v3/labels/rules_v1

Output: data/tiles_trace/ (index.csv, channels.json, shared/, rung*m/)
"""
from __future__ import annotations
import argparse, json, shutil
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

ROOT = Path(__file__).resolve().parent.parent
RUNGS = (1, 2, 5, 10, 30)
TILE = 112


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="folder of kept-flood label tifs from apply_rules.py")
    ap.add_argument("--out", default="data/tiles_trace")
    args = ap.parse_args()

    src, dst = ROOT / "data/tiles_v4", ROOT / args.out
    labdir = ROOT / args.labels
    kept = {p.stem for p in labdir.glob("*.tif")}
    idx = pd.read_csv(src / "index.csv")
    keep = idx.event.astype(str).isin(kept).values
    print(f"{len(kept)} clean floods, {int(keep.sum())} of {len(idx)} tiles belong to them", flush=True)

    if dst.exists():
        shutil.rmtree(dst)
    (dst / "shared").mkdir(parents=True)
    for r in RUNGS:
        (dst / f"rung{r}m").mkdir()
    shutil.copy(src / "channels.json", dst / "channels.json")

    cache: dict[str, tuple] = {}
    rows, n_small, shard_n = [], 0, 0
    buf = {"xs": [], "y": [], "m": [], **{r: [] for r in RUNGS}}

    def flush():
        nonlocal shard_n
        if not buf["xs"]:
            return
        np.savez_compressed(dst / "shared" / f"shard-{shard_n:05d}.npz",
                            xs=np.stack(buf["xs"]), y=np.stack(buf["y"]), m=np.stack(buf["m"]))
        for r in RUNGS:
            np.savez_compressed(dst / f"rung{r}m" / f"shard-{shard_n:05d}.npz", xt=np.stack(buf[r]))
            buf[r] = []
        buf["xs"], buf["y"], buf["m"] = [], [], []
        shard_n += 1

    sh = sorted((src / "shared").glob("shard-*.npz"))
    rg = {r: sorted((src / f"rung{r}m").glob("shard-*.npz")) for r in RUNGS}
    pos = 0
    for k, p in enumerate(sh):
        z = np.load(p)
        T = z["y"].shape[0]
        sel = np.flatnonzero(keep[pos:pos + T])
        if sel.size:
            xs, m_all = z["xs"], z["m"]
            xt = {r: np.load(rg[r][k])["xt"] for r in RUNGS}
            for t in sel:
                i = pos + t
                ev = str(idx.event.iloc[i])
                if ev not in cache:
                    with rasterio.open(labdir / f"{ev}.tif") as s:
                        cache = {ev: (s.read(1) > 0.5, s.read(2) > 0.5)}
                trace, valid = cache[ev]
                r0, c0 = int(idx.row.iloc[i]), int(idx.col.iloc[i])
                tr, va = trace[r0:r0 + TILE, c0:c0 + TILE], valid[r0:r0 + TILE, c0:c0 + TILE]
                m = m_all[t] & va
                if m.sum() < 500:
                    n_small += 1
                    continue
                buf["xs"].append(xs[t]); buf["m"].append(m)
                buf["y"].append(np.where(m & tr, 1.0, -1.0).astype("float32"))
                for r in RUNGS:
                    buf[r].append(xt[r][t])
                rows.append({"event": idx.event.iloc[i], "huc4": idx.huc4.iloc[i], "huc8": idx.huc8.iloc[i],
                             "row": r0, "col": c0, "valid_frac": round(float(m.mean()), 3),
                             "flood_frac_0.15": round(float(tr[m].mean()), 4),
                             "has_flood": bool(tr[m].mean() > 0.01),
                             "dem_clean": bool(idx.dem_clean.iloc[i])})
                if len(buf["xs"]) >= 512:
                    flush()
        pos += T
        print(f"  shard {k + 1}/{len(sh)}: {len(rows)} tiles so far", flush=True)
    flush()

    ix = pd.DataFrame(rows)
    ix.to_csv(dst / "index.csv", index=False)
    summary = {"clean_floods": len(kept), "tiles": len(ix), "dropped_small": n_small,
               "tiles_per_region": ix.huc4.astype(str).value_counts().to_dict() if len(ix) else {},
               "flood_cell_share": round(float((ix["flood_frac_0.15"] * ix.valid_frac).sum()
                                               / max(ix.valid_frac.sum(), 1e-9)), 4) if len(ix) else 0}
    (dst / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
