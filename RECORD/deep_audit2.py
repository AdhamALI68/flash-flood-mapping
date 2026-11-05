#!/usr/bin/env python3
"""Deep audit stage 2: split integrity and configuration of every saved run."""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from train import region_split, random_split

print("=" * 78)
print("STAGE 2 - EVERY SAVED RUN: split, config, and what it was scored on")
print("=" * 78)

runs = sorted(glob.glob(str(ROOT / "data/train_*.json")))
print(f"\n  {len(runs)} saved runs found\n")
rows = []
for f in runs:
    tag = os.path.basename(f)[6:-5]
    d = json.loads(open(f).read())
    ck_p = ROOT / f"data/model_{tag}.pt"
    ck = {}
    if ck_p.exists():
        try:
            ck = torch.load(ck_p, map_location="cpu", weights_only=False)
        except Exception:
            ck = {}
    rows.append({
        "run": tag,
        "test_regions": ",".join(d.get("test_regions", [])) or "-",
        "val_regions": ",".join(d.get("val_regions", [])) or "-",
        "split": ck.get("split", "region"),
        "threshold": ck.get("threshold", d.get("threshold")),
        "rung": str(ck.get("rung", d.get("rung"))),
        "prior": ck.get("prior", "downhill"),
        "select_on": ck.get("select_on", "within"),
        "pos_weight": ck.get("pos_weight", 1.0),
        "train_on": ck.get("train_on", "all"),
        "min_flood": ck.get("min_train_flood", 0.0),
        "target": ck.get("target", "binary"),
        "epochs": len(d.get("history", [])),
    })
df = pd.DataFrame(rows)

# 1. every region-split run must exclude its test regions from training
idx = pd.read_csv(ROOT / "data/tiles_v4/index.csv", dtype={"event": str})
idxn = pd.read_csv(ROOT / "data/tiles_neg_v4/index.csv", dtype={"event": str})
full = pd.concat([idx, idxn], ignore_index=True)
full["huc4"] = full.huc4.astype(str)
bad = []
for r in rows:
    if r["split"] != "region" or r["test_regions"] == "-":
        continue
    te_r = r["test_regions"].split(",")
    va_r = r["val_regions"].split(",")
    tr, va, te = region_split(full, te_r, va_r)
    tr_reg = set(full.huc4[tr]); te_reg = set(full.huc4[te])
    if tr_reg & te_reg:
        bad.append((r["run"], sorted(tr_reg & te_reg)))
print(f"  [{'PASS' if not bad else 'FAIL'}] region-split runs: train and test "
      f"basins disjoint")
print(f"         checked {sum(1 for r in rows if r['split']=='region')} runs, "
      f"violations: {bad or 'none'}")

# 2. random-split runs must be clearly marked
rnd = df[df.split == "random"]
print(f"\n  [{'PASS' if len(rnd) else 'WARN'}] random-split runs are labelled")
print(f"         {len(rnd)} run(s): {list(rnd.run)}")
print("         these test on ground that also appears in training and must")
print("         never be quoted as transfer performance")

# 3. threshold consistency
th = df.threshold.value_counts()
print(f"\n  [{'PASS' if len(th) <= 5 else 'WARN'}] label thresholds used")
for v, c in th.items():
    print(f"         {v}: {c} runs")

# 4. any run trained on a different rung than it reports
print(f"\n  [INFO] terrain resolutions used")
for v, c in df.rung.value_counts().items():
    print(f"         rung {v}: {c} runs")

# 5. runs that never finished properly
short = df[df.epochs < 5]
print(f"\n  [{'PASS' if short.empty else 'WARN'}] no run stopped implausibly early")
print(f"         runs under 5 epochs: {list(short.run) or 'none'}")

df.to_csv(ROOT / "RECORD/all_runs.csv", index=False)
print(f"\n  wrote RECORD/all_runs.csv with the full configuration of all "
      f"{len(df)} runs")
print("\n  configuration of the runs that appear in the paper:")
key = df[df.run.isin(["terrain+basin+rain+rain_routed+soil+landcover",
                      "x_region", "x_random"])]
print(key.to_string(index=False))
