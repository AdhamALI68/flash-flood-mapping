#!/usr/bin/env python3
"""
Score an already-trained model at coarser cell sizes. No retraining.

WHY
    Every number so far is at 20 m, which our own measurements say is the
    hardest scale there is. The elevation baseline climbs from 0.544 at 20 m to
    0.608 at 560 m, and trial 1 reached the same conclusion from the other side:
    "terrain-only flood products should be made at 300 to 600 m, not 10 m."

    So before concluding the model is weak, it is worth asking whether 0.63 is
    the model's ceiling or an artefact of scoring it at a resolution where the
    signal barely exists.

WHAT IT DOES
    Loads a saved checkpoint, predicts on the held-out test regions, then
    block-averages BOTH the prediction and the label to 40, 80, 160, 280 and
    560 m and recomputes within-tile AUC at each. The elevation rule is put
    through the identical treatment, so the comparison stays honest at every
    scale - a coarser grid helps the baseline too, and only the gap matters.

Output: data/eval_scales_<tag>.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet                                    # noqa: E402
from train import auc_within, load_tiles, pick_channels, region_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = [1, 2, 4, 8, 14, 28]          # x20 m = 20 .. 560 m


def block_mean(a: np.ndarray, k: int) -> np.ndarray:
    if k == 1:
        return a
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="terrain+basin+rain+rain_routed+soil+landcover")
    ap.add_argument("--tiles", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--test-regions", default="1606,1809")
    ap.add_argument("--val-regions", default="1507")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--label-frac", type=float, default=0.15,
                    help="after block-averaging, a coarse cell counts as "
                         "flooded if this fraction of it was flooded")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / f"data/model_{args.tag}.pt", map_location=dev,
                    weights_only=False)
    rung = ck["rung"]
    print(f"model {args.tag}\n  {len(ck['channels'])} channels, rung {rung} m",
          flush=True)

    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, rung)
    if (ROOT / args.neg).exists():
        Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / args.neg, rung)
        X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
                   np.concatenate([M, Mn]))
        idx = pd.concat([idx, idxn], ignore_index=True)
    _, _, te = region_split(idx, [s for s in args.test_regions.split(",") if s],
                            [s for s in args.val_regions.split(",") if s])
    cols = pick_channels(names, ck["groups"])
    if [names[i] for i in cols] != ck["channels"]:
        raise SystemExit("channel layout does not match the checkpoint")

    model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

    Xte, Yte, Mte = X[te], Y[te], M[te]
    print(f"  test tiles {len(Xte)} from regions {args.test_regions}\n", flush=True)
    P = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
            xb = torch.nan_to_num((xb - mean) / std)
            P.append(torch.sigmoid(model(xb))[:, 0].cpu().numpy())
    P = np.concatenate(P)

    ei = names.index(f"elev_mean_{rung}m")
    rows = []
    for k in BLOCKS:
        am, ae = [], []
        for i in range(len(P)):
            m = Mte[i]
            if m.sum() < 200:
                continue
            # A coarse cell is valid only where it is fully observed, so
            # partly-masked cells never invent data at the coarser scale.
            mb = block_mean(m.astype("float32"), k) > 0.999
            # A 112-px tile is only 4x4 cells at 560 m, so a fixed floor of 20
            # silently dropped the coarsest scale entirely. Scale the floor.
            if mb.sum() < max(10, (112 // k) ** 2 // 4):
                continue
            yb = block_mean(np.where(m, Yte[i] > args.threshold, 0.0), k)
            lab = yb > args.label_frac
            if lab[mb].sum() < 5 or (~lab[mb]).sum() < 5:
                continue
            pb = block_mean(np.where(m, P[i], 0.0), k)
            eb = block_mean(np.where(m, -Xte[i, ei], 0.0), k)
            am.append(auc_within(pb, lab.astype("float32"), mb))
            ae.append(auc_within(eb, lab.astype("float32"), mb))
        am = np.array(am, dtype=float)
        ae = np.array(ae, dtype=float)
        ok = np.isfinite(am) & np.isfinite(ae)
        rows.append({"cell_m": k * 20, "tiles": int(ok.sum()),
                     "model": float(np.nanmean(am[ok])),
                     "elevation": float(np.nanmean(ae[ok])),
                     "gain": float(np.nanmean(am[ok] - ae[ok]))})

    print("=" * 62)
    print("SCORE AT COARSER CELL SIZES (same model, no retraining)")
    print("=" * 62)
    print(f"  {'cell':>7}{'tiles':>8}{'model':>10}{'elevation':>12}{'gain':>9}")
    for r in rows:
        print(f"  {r['cell_m']:>5} m{r['tiles']:>8}{r['model']:>10.4f}"
              f"{r['elevation']:>12.4f}{r['gain']:>+9.4f}")
    best = max(rows, key=lambda r: r["model"])
    print(f"\n  best for the model: {best['cell_m']} m at {best['model']:.4f}")
    print(f"  at 20 m it was    : {rows[0]['model']:.4f}")

    out = ROOT / f"data/eval_scales_{args.tag}.json"
    out.write_text(json.dumps({"tag": args.tag, "rows": rows,
                               "label_frac": args.label_frac}, indent=1))
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
