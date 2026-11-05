#!/usr/bin/env python3
"""
The operational question: for EVERY pixel, on any day, flood or not?

WHY THIS IS A DIFFERENT NUMBER FROM THE 0.63
    Within-tile AUC asks only "inside this one 2.24 km square, rank the wet
    pixels above the dry ones". It deliberately throws away everything that
    distinguishes one square or one DAY from another - which is most of the
    signal, because on a quiet day nothing floods anywhere.

    Pooled pixel AUC keeps all of it: every pixel of every test tile, flood days
    and quiet days together, one ranking. That is what a real system faces.

    Both are reported, always, because pooling has a known trap. Trial 1 measured
    pooled 0.87 against within-tile 0.64 on IDENTICAL predictions - a model can
    look excellent pooled purely by knowing which squares are wet, while being
    useless at placing water inside them. Neither number alone is honest.

Also reports F1 and IoU at the best threshold, since that is what flood-mapping
papers quote and it is the only way to compare with them at all.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "terrain+basin+rain+rain_routed+soil+landcover"
TEST = ["1606", "1809"]

def pooled_auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() < 10 or (~y).sum() < 10:
        return np.nan
    n = min(3_000_000, s.size)
    i = np.linspace(0, s.size - 1, n).astype(int)
    s, y = s[i], y[i]
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))

def best_f1(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    qs = np.quantile(s, np.linspace(0.50, 0.999, 40))
    best = (0.0, 0.0, 0.0, 0.0)
    for t in qs:
        p = s >= t
        tp = float((p & y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
        if tp == 0:
            continue
        prec, rec = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best[0]:
            best = (f1, prec, rec, tp / (tp + fp + fn))
    return best

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
X, Y, M = np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn])
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, TEST, ["1507"])

cols = pick_channels(names, ck["groups"])
model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
model.load_state_dict(ck["state"]); model.eval()
mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

Xte, Yte, Mte, neg = X[te], Y[te], M[te], is_neg[te]
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                 .cpu().numpy())
P = np.concatenate(P)

flat = lambda a: a[Mte]
y = flat(Yte) > ck["threshold"]
pm = flat(P)
pe = flat(-Xte[:, names.index(f"elev_mean_{ck['rung']}m")])
ri = names.index("rain_24h_upstream_depth_mm")
pr = flat(Xte[:, ri])

print(f"model: {TAG}")
print(f"test regions {TEST}")
print(f"  {int((~neg).sum())} flood-day tiles + {int(neg.sum())} quiet-day tiles")
print(f"  {y.size:,} valid pixels, {100*y.mean():.2f}% of them flooded\n")
print("=" * 74)
print("EVERY PIXEL, ANY DAY: flood or not?   (pooled over all test tiles)")
print("=" * 74)
print(f"  {'predictor':<38}{'AUC':>8}{'F1':>8}{'IoU':>8}")
for nm, s in (("THE MODEL", pm), ("elevation only", pe),
              ("catchment rainfall only", pr)):
    a = pooled_auc(s, y); f1, prec, rec, iou = best_f1(s, y)
    print(f"  {nm:<38}{a:>8.4f}{f1:>8.4f}{iou:>8.4f}")

print("\n  and on FLOOD DAYS ONLY (the harder half: it rained, now where?)")
fd = ~neg
yf = (Yte[fd] > ck["threshold"])[Mte[fd]]
for nm, arr in (("THE MODEL", P[fd]), ("elevation only", -Xte[fd, names.index(f'elev_mean_{ck["rung"]}m')]),
                ("catchment rainfall only", Xte[fd, ri])):
    s = arr[Mte[fd]]
    a = pooled_auc(s, yf); f1, prec, rec, iou = best_f1(s, yf)
    print(f"    {nm:<36}{a:>8.4f}{f1:>8.4f}{iou:>8.4f}")

json.dump({"tag": TAG,
           "pooled_all": {"model": pooled_auc(pm, y),
                          "elevation": pooled_auc(pe, y),
                          "rain": pooled_auc(pr, y)},
           "flood_days_only": {"model": pooled_auc(P[fd][Mte[fd]], yf)},
           "pos_rate": float(y.mean())},
          (ROOT / "data/eval_pixel.json").open("w"), indent=1)
print("\nwrote data/eval_pixel.json")
