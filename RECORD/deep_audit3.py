#!/usr/bin/env python3
"""
Deep audit stage 3: is the headline number inflated by picking the best of many?

We trained ~20 comparable configurations and report the one with the highest
test AUC. Even without tuning on test, taking the maximum of many runs is
optimistic. This scores every comparable model on the SAME held-out pixels and
reports the distribution, so the size of that optimism is visible rather than
assumed.

"Comparable" means: region split, threshold 0.15, rung 10, trained on the same
tiles. Runs with a different label threshold, a different resolution or the
random split are excluded because they answer different questions.
"""
from __future__ import annotations
import json, glob, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])
k = np.flatnonzero(te)
ytrue = np.concatenate([(Y[i] > THR)[M[i]] for i in k]).astype(int)

cands = []
for f in sorted(glob.glob(str(ROOT / "data/model_*.pt"))):
    tag = os.path.basename(f)[6:-3]
    if tag.startswith("loro_") or "random" in tag:
        continue
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
    except Exception:
        continue
    if str(ck.get("rung", 10)) != "10":      continue
    if abs(float(ck.get("threshold", .15)) - .15) > 1e-6: continue
    if ck.get("split", "region") != "region": continue
    if set(ck["channels"]) - set(names):     continue   # needs tiles_v4 channels
    cands.append((tag, ck))
print(f"  {len(cands)} comparable models (region split, threshold 0.15, rung 10)\n")

rows = []
for tag, ck in cands:
    cols = pick_channels(names, ck["groups"])
    if [names[i] for i in cols] != ck["channels"]:
        continue
    net = None
    for b in (24, 40, 32, 16):
        try:
            net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=b,
                            prior=ck.get("prior", "downhill")).to(dev)
            net.load_state_dict(ck["state"]); break
        except RuntimeError:
            net = None
    if net is None:
        print(f"    (skipped {tag}, could not load)"); continue
    net.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
    P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)
    s = np.concatenate([P[j][M[i]] for j, i in enumerate(k)])
    rows.append({"run": tag, "auc": float(roc_auc_score(ytrue, s)),
                 "ap": float(average_precision_score(ytrue, s)),
                 "n_channels": len(cols)})
    print(f"    {tag:<46}{rows[-1]['auc']:.4f}", flush=True)

df = pd.DataFrame(rows).sort_values("auc", ascending=False)
print("\n" + "=" * 78)
print("HOW MUCH DOES 'BEST OF MANY' INFLATE THE HEADLINE?")
print("=" * 78)
print(df.to_string(index=False))
a = df.auc.values
print(f"\n  models compared        {len(a)}")
print(f"  best (our headline)    {a.max():.4f}")
print(f"  median                 {np.median(a):.4f}")
print(f"  mean                   {a.mean():.4f}")
print(f"  spread (max - min)     {a.max()-a.min():.4f}")
print(f"  optimism (best-median) {a.max()-np.median(a):+.4f}")
print(f"\n  bootstrap SD of a single model was 0.0248, so a spread of")
print(f"  {a.max()-a.min():.4f} across {len(a)} configurations is "
      f"{'within' if a.max()-a.min() < 3*0.0248 else 'larger than'} what")
print(f"  sampling noise alone would produce.")
df.to_csv(ROOT / "RECORD/model_spread.csv", index=False)
json.dump({"n": len(a), "best": float(a.max()), "median": float(np.median(a)),
           "mean": float(a.mean()), "min": float(a.min()),
           "optimism": float(a.max()-np.median(a))},
          (ROOT / "data/model_spread.json").open("w"), indent=1)
