#!/usr/bin/env python3
"""
Phase 8 - leave-one-region-out cross validation. NEXT_STEPS 4b.

    ./.venv/bin/python -u phase8/cross_region.py

WHY THIS AND NOT ONE SPLIT
    A single held-out split gives one number with no error bar. Whether a model
    beats the elevation rule by +0.15 or +0.02 can then depend entirely on which
    two regions happened to be held out, and there is no way to tell from the
    number itself.

    Holding out each region in turn gives six numbers per configuration. The
    spread across them IS the error bar, and it answers the question that
    matters for a paper: does this transfer to ground the model has never seen,
    or does it only work where it was trained?

    A region here is a HUC4 basin group, hundreds of kilometres across. This is
    a genuinely hard test. A model can pass a random split by memorising
    overlapping tiles; it cannot pass this one that way.

VALIDATION REGION ROTATES, AND IS NEVER THE TEST REGION
    Each fold needs a validation set for early stopping, and taking it from the
    test region would leak. It rotates to the next region in the list, so every
    fold trains on four regions, validates on a fifth and tests on the sixth.

RESUMABLE
    Each fold writes its own JSON. A fold whose file already exists is skipped,
    so this can be stopped and restarted without losing a day of GPU time.

Output: data/train_loro_<region>_<config>.json, data/cross_region_summary.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PY = str(ROOT / ".venv/bin/python")

# Configurations to cross validate. Semicolons separate configurations,
# commas separate channel families inside one.
DEFAULT_CONFIGS = ";".join([
    "terrain",
    "terrain,basin,rain",
    "terrain,basin,rain,rain_routed,soil,landcover",
])

LABELS = {
    "terrain": "terrain only",
    "terrain,basin,rain": "terrain + basin + point rain",
    "terrain,basin,rain,rain_routed,soil,landcover": "everything, incl. routed rain",
}


def log(m: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def regions(tiles: str) -> list[str]:
    ix = pd.read_csv(ROOT / tiles / "index.csv")
    return sorted(ix.huc4.astype(str).unique())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tiles", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--rung", type=int, default=10)
    ap.add_argument("--configs", default=DEFAULT_CONFIGS)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--only-region", default="")
    args = ap.parse_args()

    regs = regions(args.tiles)
    cfgs = [c for c in args.configs.split(";") if c]
    if args.only_region:
        regs = [r for r in regs if r == args.only_region]

    log(f"regions: {regs}")
    log(f"configurations: {len(cfgs)}")
    log(f"total folds to run: {len(regs) * len(cfgs)}\n")

    for ci, cfg in enumerate(cfgs, 1):
        for ri, test in enumerate(regs):
            val = regs[(ri + 1) % len(regs)]        # never the test region
            tag = f"loro_{test}_{cfg.replace(',', '+')}"
            out = DATA / f"train_{tag}.json"
            if out.exists():
                log(f"SKIP  config {ci}/{len(cfgs)}  test {test}  (already done)")
                continue
            log(f"START config {ci}/{len(cfgs)} [{cfg}]  test region {test}, "
                f"val {val}")
            t0 = time.time()
            rc = subprocess.call(
                [PY, "-u", "phase6/train.py",
                 "--tiles", args.tiles, "--neg", args.neg,
                 "--rung", str(args.rung), "--groups", cfg,
                 "--test-regions", test, "--val-regions", val,
                 "--epochs", str(args.epochs), "--patience", str(args.patience),
                 "--tag", tag],
                cwd=ROOT,
                stdout=(DATA / "cross_region.log").open("a"),
                stderr=subprocess.STDOUT)
            mins = (time.time() - t0) / 60
            log(f"{'DONE ' if rc == 0 else 'FAIL '} test {test} in {mins:.1f} min"
                + ("" if rc == 0 else f" (rc={rc})"))

    # ---- summarise ---------------------------------------------------------
    rows = []
    for cfg in cfgs:
        for test in regs:
            p = DATA / f"train_loro_{test}_{cfg.replace(',', '+')}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            rows.append({"config": cfg, "test_region": test,
                         "tiles": d["test_tiles"],
                         "model": d["auc_model"],
                         "elevation": d["auc_elevation"],
                         "gain": d["paired_diff"],
                         "p": d.get("wilcoxon_p"),
                         "flood_vs_quiet": d.get("flood_vs_quiet_auc")})
    if not rows:
        log("no completed folds to summarise")
        return 0
    df = pd.DataFrame(rows)

    print("\n" + "=" * 82)
    print("LEAVE-ONE-REGION-OUT CROSS VALIDATION")
    print("=" * 82)
    for cfg in cfgs:
        g = df[df.config == cfg]
        if g.empty:
            continue
        print(f"\n{LABELS.get(cfg, cfg)}   ({len(g)} folds)")
        print(f"  {'test region':<14}{'tiles':>8}{'model':>9}{'elevation':>11}"
              f"{'gain':>9}{'flood vs quiet':>16}")
        for r in g.itertuples(index=False):
            fq = f"{r.flood_vs_quiet:.4f}" if r.flood_vs_quiet is not None else "-"
            print(f"  {r.test_region:<14}{r.tiles:>8}{r.model:>9.4f}"
                  f"{r.elevation:>11.4f}{r.gain:>+9.4f}{fq:>16}")
        print(f"  {'MEAN':<14}{'':>8}{g.model.mean():>9.4f}"
              f"{g.elevation.mean():>11.4f}{g.gain.mean():>+9.4f}"
              f"{g.flood_vs_quiet.mean():>16.4f}")
        print(f"  {'std across regions':<14}{'':>8}{g.model.std():>9.4f}"
              f"{g.elevation.std():>11.4f}{g.gain.std():>+9.4f}"
              f"{g.flood_vs_quiet.std():>16.4f}")
        wins = int((g.gain > 0).sum())
        print(f"  beats the elevation rule in {wins}/{len(g)} regions")

    if len(cfgs) > 1:
        print("\n" + "-" * 82)
        print("DOES THE EXTRA INFORMATION PAY? (paired by region)")
        print("-" * 82)
        base = df[df.config == cfgs[0]].set_index("test_region")
        for cfg in cfgs[1:]:
            g = df[df.config == cfg].set_index("test_region")
            common = base.index.intersection(g.index)
            if len(common) < 2:
                continue
            d = (g.loc[common, "model"] - base.loc[common, "model"])
            fq = (g.loc[common, "flood_vs_quiet"]
                  - base.loc[common, "flood_vs_quiet"])
            print(f"  {LABELS.get(cfg, cfg)}")
            print(f"    vs {LABELS.get(cfgs[0], cfgs[0])}: "
                  f"within-tile {d.mean():+.4f} (better in {int((d>0).sum())}"
                  f"/{len(common)} regions)")
            print(f"    flood-vs-quiet separation {fq.mean():+.4f}")

    out = DATA / "cross_region_summary.json"
    out.write_text(json.dumps({"rows": rows,
                               "regions": regs, "configs": cfgs}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
