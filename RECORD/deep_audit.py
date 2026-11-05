#!/usr/bin/env python3
"""
Deep audit. Every dataset, every split, every saved experiment.

Checks facts that could invalidate a result, not code style. Each check prints
PASS, FAIL or WARN with the evidence, so a reader can disagree with the
judgement while still seeing the number.
"""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
R = {"pass": 0, "fail": 0, "warn": 0}


def chk(name, ok, detail, warn=False):
    tag = "PASS" if ok else ("WARN" if warn else "FAIL")
    R["pass" if ok else ("warn" if warn else "fail")] += 1
    print(f"  [{tag}] {name}")
    for line in str(detail).splitlines():
        print(f"         {line}")


print("=" * 78)
print("STAGE 1 - DATA INTEGRITY")
print("=" * 78)

# --- tile sets -----------------------------------------------------------
for ts in ("tiles_v4", "tiles_neg_v4", "tiles_x", "tiles_x_neg"):
    d = ROOT / "data" / ts
    if not d.exists():
        chk(f"{ts} exists", False, "directory missing"); continue
    ix = pd.read_csv(d / "index.csv")
    ch = json.loads((d / "channels.json").read_text())
    sh = sorted((d / "shared").glob("shard-*.npz"))
    n_sh = sum(len(np.load(p)["y"]) for p in sh)
    chk(f"{ts}: index matches shards", n_sh == len(ix),
        f"index rows {len(ix)}, shard tiles {n_sh}")
    z = np.load(sh[0]); y = z["y"][z["m"]]
    lo, hi = float(y.min()), float(y.max())
    chk(f"{ts}: label inside -1..+1", -1.001 < lo and hi < 1.001,
        f"range {lo:.4f} to {hi:.4f}  (coherence difference cannot exceed 1)")
    chk(f"{ts}: mask is not empty", z["m"].mean() > 0.3,
        f"{100*z['m'].mean():.1f}% of pixels valid in shard 0")
    chk(f"{ts}: grid is 20 m", ch["res_m"] == 20 and ch["tile_px"] == 112,
        f"{ch['tile_px']}x{ch['tile_px']} at {ch['res_m']} m")
    # no NaN inside the mask, on a sample
    bad = []
    for p in sh[:2]:
        zz = np.load(p)
        xs, mm = zz["xs"], zz["m"]
        for c in range(xs.shape[1]):
            if not np.isfinite(xs[:, c][mm]).all():
                bad.append(ch["shared_channels"][c])
    chk(f"{ts}: no NaN inside the mask", not bad,
        f"channels with NaN: {sorted(set(bad)) or 'none'}")

# --- flood events and controls are disjoint ------------------------------
p_ix = pd.read_csv(ROOT / "data/tiles_v4/index.csv", dtype={"event": str})
n_ix = pd.read_csv(ROOT / "data/tiles_neg_v4/index.csv", dtype={"event": str})
overlap = set(p_ix.event) & set(n_ix.event)
chk("flood events and quiet controls are disjoint", not overlap,
    f"shared ids: {sorted(overlap)[:5] or 'none'}  "
    f"({p_ix.event.nunique()} flood, {n_ix.event.nunique()} control)")
chk("control ids are marked as controls",
    all(str(e).startswith("NEG") for e in n_ix.event.unique()),
    f"sample: {sorted(n_ix.event.unique())[:3]}")

# --- duplicate tiles -----------------------------------------------------
dup = p_ix.duplicated(subset=["event", "row", "col"]).sum()
chk("no duplicate tiles in the flood set", dup == 0, f"{dup} duplicates")

# --- basins present ------------------------------------------------------
chk("six basins present", p_ix.huc4.nunique() == 6,
    f"{sorted(p_ix.huc4.unique())}")
print(f"\n  tiles per basin:\n{p_ix.groupby('huc4').size().to_string()}")
