#!/usr/bin/env python3
"""
Phase 6 - classification report and confusion matrix, per pixel.

    ./.venv/bin/python -u phase6/classification_report.py

The standard sklearn report over every 20 m pixel of the held-out regions,
plus the confusion matrix as a figure.

TWO CLASSES, NOT THREE
    An InSAR coherence-drop label says "this pixel changed" - it does not
    distinguish permanent water from flood water, so there is no third class to
    report. Permanent water is a trivially separable class in datasets that have
    it (it is already dark in the pre-event image), so its presence inflates
    macro-averaged scores. Ours has no such class to lean on.

ONE SCOPE: every valid 20 m pixel of the held-out regions, flood days and
quiet days together. That is the research question, and no other slice is
reported, because reporting several invites reading whichever looks best.

THE CUTOFF IS SHOWN AT THREE OPERATING POINTS
    Precision and recall trade against each other, and a single row hides that.
    The best-F1 cutoff is the headline, with a high-recall and a high-precision
    point either side so the trade is visible rather than chosen silently.

Output: data/report/classification_report.txt, data/report/confusion_matrix.png
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
import torch
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet                                   # noqa: E402
from train import load_tiles, pick_channels, region_split      # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/report"
NAMES = ["Dry (0)", "Flood (1)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="terrain+basin+rain+rain_routed+soil+landcover")
    ap.add_argument("--tiles", default="data/tiles_v4")
    ap.add_argument("--neg", default="data/tiles_neg_v4")
    ap.add_argument("--test-regions", default="1606,1809")
    ap.add_argument("--val-regions", default="1507")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / f"data/model_{args.tag}.pt", map_location=dev,
                    weights_only=False)
    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, ck["rung"])
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / args.neg, ck["rung"])
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    import pandas as pd
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

    Xte, Yte, Mte = X[te], Y[te], M[te]
    P = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
            P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)

    y_score = np.concatenate([P[i][Mte[i]] for i in range(len(P))])
    y_true = np.concatenate([(Yte[i] > ck["threshold"])[Mte[i]]
                             for i in range(len(P))]).astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap_ = average_precision_score(y_true, y_score)

    # operating points: best F1, plus one either side
    qs = np.quantile(y_score, np.linspace(0.50, 0.999, 120))
    best_t, best_f1 = 0.5, 0.0
    for t in qs:
        p = (y_score >= t).astype(int)
        tp = float((p & y_true).sum()); fp = float((p & ~y_true.astype(bool)).sum())
        fn = float(((1 - p) & y_true).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    # High recall means flagging MORE pixels, so a LOWER cutoff. Deriving both
    # from the headline cutoff guarantees the ordering is right rather than
    # relying on a quantile that can land on the wrong side of it.
    lo = y_score[y_score < best_t]
    hi = y_score[y_score > best_t]
    hi_recall = float(np.quantile(lo, 0.60)) if lo.size else best_t * 0.5
    hi_prec = float(np.quantile(hi, 0.85)) if hi.size else best_t * 1.5

    lines = []
    lines.append("=" * 66)
    lines.append("CLASSIFICATION REPORT - per 20 m pixel, held-out regions "
                 f"{args.test_regions}")
    lines.append("=" * 66)
    lines.append(f"pixels {y_true.size:,}   flooded {100*y_true.mean():.2f}%"
                 f"   AUC {auc:.4f}   average precision {ap_:.4f}")
    lines.append("")

    for label, t in (("BEST F1 (headline)", best_t),
                     ("HIGH RECALL", hi_recall),
                     ("HIGH PRECISION", hi_prec)):
        y_pred = (y_score >= t).astype(int)
        lines.append("-" * 66)
        lines.append(f"{label}   cutoff = {t:.4f}")
        lines.append("-" * 66)
        lines.append(classification_report(y_true, y_pred, target_names=NAMES,
                                           digits=4, zero_division=0))
        cm = confusion_matrix(y_true, y_pred)
        lines.append("confusion matrix (rows = actual, cols = predicted)")
        lines.append(f"              {'Dry':>12}{'Flood':>12}")
        for nm, r in zip(NAMES, cm):
            lines.append(f"  {nm:<12}{r[0]:>12,}{r[1]:>12,}")
        lines.append("")

    txt = "\n".join(lines)
    print(txt)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "classification_report.txt").write_text(txt)

    # ---- confusion matrix figure at the headline cutoff --------------------
    y_pred = (y_score >= best_t).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(cm, cmap="Oranges")
    ax.set_title("Flood U-Net confusion matrix\n(per 20 m pixel, held-out regions)",
                 fontsize=11)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks([0, 1], NAMES)
    ax.set_yticks([0, 1], NAMES)
    thr_col = cm.max() / 2.0
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=11,
                    color="white" if cm[i, j] > thr_col else "black")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(OUT / "confusion_matrix.png", dpi=130)
    plt.close(fig)

    json.dump({"tag": args.tag, "auc": float(auc), "average_precision": float(ap_),
               "cutoff_best_f1": best_t, "n_pixels": int(y_true.size),
               "positive_rate": float(y_true.mean()),
               "confusion_matrix": cm.tolist()},
              (OUT / "classification_report.json").open("w"), indent=1)
    print(f"wrote {(OUT / 'classification_report.txt').relative_to(ROOT)}")
    print(f"      {(OUT / 'confusion_matrix.png').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
