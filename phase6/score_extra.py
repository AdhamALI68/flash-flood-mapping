#!/usr/bin/env python3
"""Score the sink/distance models on BOTH splits, one protocol.

Each model is evaluated on ITS OWN held-out set - the region model on the
held-out deserts, the random model on its own random holdout. The cutoff is
fitted on validation as a flagging RATE and applied unchanged to test, since an
absolute probability does not transfer between regions.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split, random_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"

X = None
rows = []
for tag in ("x_region", "x_random"):
    ck = torch.load(ROOT / f"data/model_{tag}.pt", map_location=dev,
                    weights_only=False)
    if X is None:
        X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_x", "10")
        Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_x_neg", "10")
        X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
                   np.concatenate([M, Mn]))
        idx = pd.concat([idx, idxn], ignore_index=True)
    if ck.get("split") == "random":
        tr, va, te = random_split(idx, seed=ck.get("seed", 0))
        scope = "random (same ground)"
    else:
        tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])
        scope = "region (unseen ground)"
    cols = pick_channels(names, ck["groups"])
    net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                    prior=ck.get("prior", "downhill")).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

    def flat(sel):
        k = np.flatnonzero(sel); P = []
        with torch.no_grad():
            for i in range(0, len(k), 32):
                xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
                P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                         .cpu().numpy())
        P = np.concatenate(P)
        s = np.concatenate([P[j][M[i]] for j, i in enumerate(k)])
        y = np.concatenate([(Y[i] > THR)[M[i]] for i in k])
        return s, y

    sv, yv = flat(va)
    st, yt = flat(te)
    rate, bf = 0.2, 0.0
    for q in np.linspace(0.02, 0.60, 120):
        p = sv >= np.quantile(sv, 1-q)
        tp = float((p & yv).sum()); fp = float((p & ~yv).sum()); fn = float((~p & yv).sum())
        if tp == 0: continue
        pr, rc = tp/(tp+fp), tp/(tp+fn); f = 2*pr*rc/(pr+rc)
        if f > bf: rate, bf = float(q), f
    cut = np.quantile(st, 1-rate)
    p = st >= cut
    tp = float((p & yt).sum()); fp = float((p & ~yt).sum())
    fn = float((~p & yt).sum()); tn = float((~p & ~yt).sum())
    rec, pre = tp/(tp+fn), tp/(tp+fp)
    rows.append({"tag": tag, "scope": scope,
                 "auc": float(roc_auc_score(yt.astype(int), st)),
                 "ap": float(average_precision_score(yt.astype(int), st)),
                 "recall": rec, "precision": pre,
                 "f1": 2*pre*rec/(pre+rec), "accuracy": (tp+tn)/len(p),
                 "base": float(yt.mean()), "n": int(yt.size), "rate": rate})
    print(f"  scored {tag}", flush=True)

print("\n" + "=" * 92)
print("WITH sink_depth + dist_to_stream (33 channels), per 20 m pixel")
print("=" * 92)
print(f"  {'split':<26}{'AUC':>8}{'AP':>8}{'recall':>9}{'prec':>8}{'F1':>8}"
      f"{'acc':>8}{'flagged':>9}")
for r in rows:
    print(f"  {r['scope']:<26}{r['auc']:>8.4f}{r['ap']:>8.4f}{r['recall']:>9.4f}"
          f"{r['precision']:>8.4f}{r['f1']:>8.4f}{r['accuracy']:>8.4f}"
          f"{100*r['rate']:>8.0f}%")
print("\n  for comparison, WITHOUT the new features (30 channels):")
print(f"  {'region (unseen ground)':<26}{0.6950:>8.4f}{0.0916:>8.4f}"
      f"{0.3968:>9.4f}{0.1014:>8.4f}{0.1615:>8.4f}{0.7840:>8.4f}{21:>8.0f}%")
json.dump(rows, (ROOT / "data/score_extra.json").open("w"), indent=1)
print("\nwrote data/score_extra.json")
