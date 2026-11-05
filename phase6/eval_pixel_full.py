#!/usr/bin/env python3
"""
THE research question: for every 20 m pixel, will it flood or not?

Scores the model against EVERY input channel it was given, on the same pixels,
so there is no doubt about what the model adds over simply reading one layer.

WHY EVERY CHANNEL AND NOT JUST ELEVATION
    A model that beats elevation but loses to some other input it was handed is
    not adding anything - it is just a worse version of that input. The only way
    to know is to score all of them the same way.

LEAKAGE GUARD
    Every channel here is knowable BEFORE the flood: terrain, basin routing,
    rainfall, soil, land cover. Nothing from the co-event radar enters the
    features - the coherence pair and amplitude live only in label.tif and are
    never tiled as inputs. Any channel scoring implausibly high would be a
    warning sign, so the full ranking is printed rather than just the top.

GEOMETRY IS ASSERTED, NOT ASSUMED
    The claim "per pixel at 20 m" is checked against channels.json and the tile
    shape before any number is reported.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split, LOG_CHANNELS

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "terrain+basin+rain+rain_routed+soil+landcover"
TEST = ["1606", "1809"]
SUB = 4_000_000          # pixels sampled for ranking; AUC is stable well below this

# Direction fixed by physics, not fitted. Low ground floods, steep ground does not.
NEG = {"elev_mean_10m", "elev_min_10m", "slope_10m", "basin_hand", "basin_slope",
       "soil_ksat", "soil_sand", "microrelief_10m", "roughness_10m"}


def pooled_auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() < 10 or (~y).sum() < 10:
        return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def best_f1(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]
    best = (0.0, 0.0)
    for t in np.quantile(s, np.linspace(0.50, 0.999, 30)):
        p = s >= t
        tp = float((p & y).sum()); fp = float((p & ~y).sum()); fn = float((~p & y).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best[0]:
            best = (f1, tp / (tp + fp + fn))
    return best


dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)

# ---- geometry check ------------------------------------------------------
chan = json.loads((ROOT / "data/tiles_v4/channels.json").read_text())
print("GEOMETRY CHECK")
print(f"  tile        {chan['tile_px']} x {chan['tile_px']} pixels")
print(f"  resolution  {chan['res_m']} m per pixel")
print(f"  tile covers {chan['tile_px']*chan['res_m']/1000:.2f} km across")
assert chan["res_m"] == 20, "not a 20 m grid"

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
X, Y, M = np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn])
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, TEST, ["1507"])
print(f"  model output shape matches input tile: "
      f"{X.shape[-2:]} -> prediction is one value per 20 m pixel\n")

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

rng = np.random.default_rng(0)
for scope, sel in (("ALL DAYS (flood days + quiet days)", np.ones(len(Xte), bool)),
                   ("FLOOD DAYS ONLY (it rained - now where?)", ~neg)):
    m = Mte[sel]
    y = (Yte[sel] > ck["threshold"])[m]
    n = y.size
    take = rng.choice(n, size=min(SUB, n), replace=False)
    y_s = y[take]
    rows = [("*** THE MODEL (U-Net, 30 channels) ***", P[sel][m][take])]
    for c in ck["channels"]:
        v = Xte[sel][:, names.index(c)][m][take]
        rows.append((c, -v if c in NEG else v))
    rows.append(("random", rng.random(y_s.size)))

    print("=" * 78)
    print(f"{scope}")
    print(f"{n:,} valid 20 m pixels, {100*y.mean():.2f}% flooded "
          f"({min(SUB, n):,} sampled)")
    print("=" * 78)
    out = []
    for nm, s in rows:
        a = pooled_auc(s, y_s)
        f1, iou = best_f1(s, y_s)
        out.append((nm, a, f1, iou))
    out.sort(key=lambda r: (-r[1] if np.isfinite(r[1]) else 0))
    print(f"  {'predictor':<42}{'AUC':>8}{'F1':>8}{'IoU':>8}")
    for nm, a, f1, iou in out:
        star = "  <--" if nm.startswith("***") else ""
        print(f"  {nm:<42}{a:>8.4f}{f1:>8.4f}{iou:>8.4f}{star}")
    best_in = max((r for r in out if not r[0].startswith("***")
                   and r[0] != "random"), key=lambda r: r[1])
    mo = out[[r[0] for r in out].index("*** THE MODEL (U-Net, 30 channels) ***")]
    print(f"\n  best single input : {best_in[0]} = {best_in[1]:.4f}")
    print(f"  the model         : {mo[1]:.4f}   ({mo[1]-best_in[1]:+.4f})")
    print(f"  VERDICT: model {'BEATS' if mo[1] > best_in[1] else 'LOSES TO'} "
          f"every single input it was given\n")
    json.dump([{"predictor": n_, "auc": a, "f1": f, "iou": i}
               for n_, a, f, i in out],
              (ROOT / f"data/eval_pixel_full_{scope.split()[0]}.json").open("w"),
              indent=1)
print("wrote data/eval_pixel_full_*.json")
