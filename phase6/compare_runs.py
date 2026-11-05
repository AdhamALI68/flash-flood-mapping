#!/usr/bin/env python3
"""
One table: every finished model on the single metric set, per 20 m pixel.

Test set is identical and unfiltered for every row, so the rows are comparable.
Average precision is the column that matters: recall and precision can be
traded against each other by moving the cutoff, AP cannot.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split, random_split

ROOT = Path(__file__).resolve().parent.parent
TAGS = sys.argv[1:] or ["terrain+basin+rain+rain_routed+soil+landcover"]
dev = "cuda" if torch.cuda.is_available() else "cpu"

X = Y = M = None
rows = []
for tag in TAGS:
    ckp = ROOT / f"data/model_{tag}.pt"
    if not ckp.exists():
        print(f"  (missing {tag})"); continue
    ck = torch.load(ckp, map_location=dev, weights_only=False)
    if X is None:
        X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
        Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
        X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
                   np.concatenate([M, Mn]))
        idx = pd.concat([idx, idxn], ignore_index=True)

    # EACH MODEL IS SCORED ON ITS OWN HELD-OUT SET.
    # A randomly-split model trained on 65% of every region would otherwise be
    # evaluated on the region test set it was trained on - scoring it against
    # its own training tiles. That produced a bogus 0.8907 before this fix.
    if ck.get("split") == "random":
        _, _, te = random_split(idx, seed=ck.get("seed", 0))
        scope = "random (same ground)"
    else:
        _, _, te = region_split(idx, ["1606", "1809"], ["1507"])
        scope = "region (unseen ground)"
    Xte, Yte, Mte = X[te], Y[te], M[te]
    y = np.concatenate([(Yte[i] > ck["threshold"])[Mte[i]]
                        for i in range(len(Xte))]).astype(int)
    cols = pick_channels(names, ck["groups"])
    m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=ck.get("base", 24)
                  if isinstance(ck.get("base"), int) else 24,
                  prior=ck.get("prior", "downhill")).to(dev)
    try:
        m.load_state_dict(ck["state"])
    except RuntimeError:
        for b in (40, 32, 24, 16):
            try:
                m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"],
                              base=b, prior=ck.get("prior", "downhill")).to(dev)
                m.load_state_dict(ck["state"]); break
            except RuntimeError:
                continue
    m.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
    P = []
    with torch.no_grad():
        for i in range(0, len(Xte), 32):
            xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
            P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)
    s = np.concatenate([P[i][Mte[i]] for i in range(len(P))])
    auc = roc_auc_score(y, s); ap = average_precision_score(y, s)
    best = (0.0, 0.0, 0.0)
    for t in np.quantile(s, np.linspace(0.50, 0.999, 60)):
        p = s >= t
        tp = float((p & y.astype(bool)).sum()); fp = float((p & ~y.astype(bool)).sum())
        fn = float((~p & y.astype(bool)).sum())
        if tp == 0: continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best[0]: best = (f1, pr, rc)
    rows.append({"run": tag, "AUC": auc, "AP": ap,
                 "recall": best[2], "precision": best[1], "F1": best[0],
                 "pos_weight": ck.get("pos_weight", 1.0),
                 "train_on": ck.get("train_on", "all"),
                 "min_flood": ck.get("min_train_flood", 0.0),
                 "scope": scope, "n_pix": int(y.size),
                 "base": float(y.mean())})
    print(f"  scored {tag}", flush=True)

df = pd.DataFrame(rows).sort_values("AP", ascending=False)
print("\n" + "=" * 96)
print("ALL RUNS - per 20 m pixel, each on ITS OWN held-out set")
print("=" * 96)
print(f"  {'run':<27}{'AUC':>8}{'AP':>8}{'recall':>8}{'prec':>8}"
      f"{'F1':>8}{'base':>8}  test set")
for r in df.itertuples(index=False):
    print(f"  {r.run[:26]:<27}{r.AUC:>8.4f}{r.AP:>8.4f}{r.recall:>8.4f}"
          f"{r.precision:>8.4f}{r.F1:>8.4f}{100*r.base:>7.2f}%  {r.scope}")
print("\n  AP must beat the base rate column to mean anything.")
df.to_csv(ROOT / "data/compare_runs.csv", index=False)
print("wrote data/compare_runs.csv")
