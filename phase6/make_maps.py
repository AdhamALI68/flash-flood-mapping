#!/usr/bin/env python3
"""
Phase 6 - example maps: what the flood actually was, and what the model said.

    ./.venv/bin/python -u phase6/make_maps.py --tag <model tag>

WHY THESE ARE WORTH MAKING
    An AUC of 0.695 tells you the ranking is better than chance. It does not
    tell you whether the model is drawing water down the washes or smearing a
    grey blur over the whole tile. Only looking does that, and it is the first
    thing a reviewer or a supervisor asks to see.

WHAT EACH ROW SHOWS
    terrain      slope, as the eye's reference for where water should go
    rainfall     catchment-routed rain arriving at each pixel
    TRUTH        the label: where InSAR says the flood was
    MODEL        the predicted probability, same colour scale every panel
    agreement    green = both agree flood, red = model missed, orange = false alarm

    Tiles are chosen to be honest rather than flattering: the best, the worst and
    the middle of the held-out test set by model score, plus a quiet-day tile
    where the correct answer is "nothing". A page of only good examples is a
    sales brochure, not evidence.

Output: data/maps/*.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.colors import ListedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet                                       # noqa: E402
from train import auc_within, load_tiles, pick_channels, region_split  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/maps"


def panel(ax, a, mask, title, cmap="viridis", vmin=None, vmax=None):
    b = np.where(mask, a, np.nan)
    im = ax.imshow(b, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    return im


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="terrain+basin+rain+rain_routed+soil+landcover")
    ap.add_argument("--tiles", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--test-regions", default="1606,1809")
    ap.add_argument("--val-regions", default="1507")
    ap.add_argument("--n-each", type=int, default=2)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / f"data/model_{args.tag}.pt", map_location=dev,
                    weights_only=False)
    rung, thr = ck["rung"], ck["threshold"]

    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, rung)
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / args.neg, rung)
    is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)
    _, _, te = region_split(idx, [s for s in args.test_regions.split(",") if s],
                            [s for s in args.val_regions.split(",") if s])

    cols = pick_channels(names, ck["groups"])
    model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                      prior=ck.get("prior", "downhill")).to(dev)
    model.load_state_dict(ck["state"])
    model.eval()
    mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

    Xte, Yte, Mte, negte = X[te], Y[te], M[te], is_neg[te]
    ixte = idx[te].reset_index(drop=True)
    P = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
            P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)

    lab = (Yte > thr)

    # THE OPERATING THRESHOLD IS MEASURED, NOT ASSUMED TO BE 0.5.
    # Only ~10% of pixels flood, so a model trained on that imbalance correctly
    # outputs low probabilities almost everywhere - on the best test tile it
    # peaks near 0.4 while ranking almost perfectly (AUC 0.939). Drawing the
    # agreement panel at 0.5 therefore painted a good prediction entirely red.
    # The cutoff that maximises F1 over the whole test set is the honest one to
    # show, and it is stated on every figure so nothing is hidden.
    sflat = np.concatenate([P[i][Mte[i]] for i in range(len(P))])
    yflat = np.concatenate([lab[i][Mte[i]] for i in range(len(P))])
    op, best_f1 = 0.5, 0.0
    for t in np.quantile(sflat, np.linspace(0.50, 0.999, 60)):
        pb = sflat >= t
        tp = float((pb & yflat).sum()); fp = float((pb & ~yflat).sum())
        fn = float((~pb & yflat).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best_f1:
            op, best_f1 = float(t), f1
    print(f"  operating threshold {op:.3f} (best F1 {best_f1:.4f} on test); "
          f"0.5 would predict flood on "
          f"{100*float((sflat>=0.5).mean()):.2f}% of pixels", flush=True)

    auc = np.array([auc_within(P[i], lab[i].astype("float32"), Mte[i])
                    for i in range(len(P))])
    frac = np.array([lab[i][Mte[i]].mean() if Mte[i].sum() > 200 else np.nan
                     for i in range(len(P))])

    # honest selection: best, median and worst real floods, plus quiet days
    real = np.flatnonzero(~negte & np.isfinite(auc) & (frac > 0.05))
    quiet = np.flatnonzero(negte & (Mte.reshape(len(Mte), -1).sum(1) > 5000))
    order = real[np.argsort(-auc[real])]
    k = args.n_each
    mid = len(order) // 2
    picks = [("best", i) for i in order[:k]]
    picks += [("typical", i) for i in order[mid:mid + k]]
    picks += [("worst", i) for i in order[-k:]]
    picks += [("quiet day - correct answer is nothing", i) for i in quiet[:k]]

    OUT.mkdir(parents=True, exist_ok=True)
    si = names.index(f"slope_{rung}m")
    ri = names.index("rain_24h_upstream_depth_mm")
    agree_cmap = ListedColormap(["#f0f0f0", "#d73027", "#fdae61", "#1a9850"])
    made = []

    for kind, i in picks:
        m = Mte[i]
        pr, tr = P[i], lab[i]
        # 0 both dry, 1 missed by model, 2 false alarm, 3 both flood
        pb = pr >= op
        cls = np.zeros_like(pr, dtype=int)
        cls[(tr & ~pb)] = 1
        cls[(~tr & pb)] = 2
        cls[(tr & pb)] = 3

        fig, axes = plt.subplots(1, 5, figsize=(16, 3.6))
        panel(axes[0], Xte[i, si], m, f"slope\n(where water should go)", "cividis")
        panel(axes[1], Xte[i, ri], m, "catchment rainfall\narriving (mm)", "Blues")
        panel(axes[2], tr.astype(float), m, "TRUTH\n(InSAR label)", "Reds", 0, 1)
        im = panel(axes[3], pr, m, "MODEL\n(predicted probability)", "Reds",
                   0, max(0.35, float(np.nanpercentile(pr[m], 99.5))))
        panel(axes[4], cls, m, "green=hit  red=missed\norange=false alarm",
              agree_cmap, 0, 3)
        fig.colorbar(im, ax=axes[3], fraction=0.046)

        ev = ixte.event.iloc[i]
        reg = ixte.huc4.iloc[i]
        a = auc[i]
        fig.suptitle(f"{kind}   event {ev}   region {reg}   "
                     f"{100*frac[i]:.1f}% of tile flooded   "
                     f"within-tile AUC {a:.3f}   "
                     f"(2.24 km across, 20 m pixels, cutoff {op:.2f})",
                     fontsize=10)
        fig.tight_layout(rect=[0, 0, 1, 0.93])
        name = f"{kind.split()[0]}_{ev}_r{reg}_{i}.png".replace(" ", "_")
        fig.savefig(OUT / name, dpi=110)
        plt.close(fig)
        made.append({"file": name, "kind": kind, "event": str(ev),
                     "region": str(reg), "flood_frac": float(frac[i]),
                     "auc": None if not np.isfinite(a) else float(a)})
        print(f"  {name}", flush=True)

    (OUT / "index.json").write_text(json.dumps(
        {"tag": args.tag, "test_regions": args.test_regions,
         "label_threshold": thr, "operating_threshold": op,
         "best_f1_on_test": best_f1, "maps": made}, indent=1))
    print(f"\n{len(made)} maps in {OUT.relative_to(ROOT)}")
    print("selection is deliberately mixed: best, typical, worst and quiet-day")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
