#!/usr/bin/env python3
"""
AUDIT: is the per-pixel flood prediction actually sound?

One question only - for each 20 m pixel, flood or not - and whether the number
we report for it can be trusted. Seven independent checks. Any FAIL invalidates
the headline result.

  1  geometry            is a pixel really 20 m, and does the output line up
  2  label               is the target coh_drop, in range, correctly thresholded
  3  spatial alignment   does prediction[i,j] correspond to label[i,j]
  4  leakage             can any input see the flood it is predicting
  5  held-out split      are the test regions genuinely absent from training
  6  AUC arithmetic      does our AUC match an independent implementation
  7  IS IT PER PIXEL     does the prediction vary INSIDE a tile, or is it one
                         value per tile dressed up as a map
Check 7 is the one that matters most. A model that outputs a near-constant value
per tile can score well pooled purely by knowing which DAYS are wet, while
predicting nothing about individual pixels. That would make the headline false
even with every other check passing.
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
results = {}


def report(n, name, ok, detail):
    results[name] = {"pass": bool(ok), "detail": detail}
    print(f"  [{n}] {'PASS' if ok else '*** FAIL ***'}  {name}")
    for line in detail.splitlines():
        print(f"          {line}")


print("=" * 78)
print("AUDIT: per-pixel flood prediction")
print("=" * 78)

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
chan = json.loads((ROOT / "data/tiles_v4/channels.json").read_text())

# ---- 1 geometry ---------------------------------------------------------
report(1, "geometry", chan["res_m"] == 20 and chan["tile_px"] == 112,
       f"{chan['tile_px']}x{chan['tile_px']} px at {chan['res_m']} m = "
       f"{chan['tile_px']*chan['res_m']/1000:.2f} km per tile")

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", ck["rung"])
is_neg = np.concatenate([np.zeros(len(X), bool), np.ones(len(Xn), bool)])
X, Y, M = np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn])
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, TEST, ["1507"])

# ---- 2 label ------------------------------------------------------------
v = Y[M]
thr = ck["threshold"]
pos = float((v > thr).mean())
report(2, "label", -1.001 < v.min() and v.max() < 1.001 and 0.001 < pos < 0.5,
       f"target = coh_drop > {thr}\n"
       f"range {v.min():.3f} .. {v.max():.3f} (must be inside -1..1)\n"
       f"positive rate {100*pos:.2f}% of valid pixels")

# ---- 5 held-out split ---------------------------------------------------
rtr = set(idx.huc4.astype(str)[tr]); rte = set(idx.huc4.astype(str)[te])
ev_tr = set(idx.event.astype(str)[tr]); ev_te = set(idx.event.astype(str)[te])
report(5, "held-out split", not (rtr & rte) and not (ev_tr & ev_te),
       f"train regions {sorted(rtr)}\ntest regions {sorted(rte)}\n"
       f"overlapping regions {sorted(rtr & rte) or 'none'}\n"
       f"overlapping events  {len(ev_tr & ev_te)}")

# ---- 4 leakage ----------------------------------------------------------
post = [c for c in ck["channels"]
        if any(k in c.lower() for k in ("coh", "amp", "coevent", "post", "sar"))]
report(4, "leakage", not post,
       f"{len(ck['channels'])} inputs, all pre-event by construction\n"
       f"channels mentioning coherence/amplitude/post-event: {post or 'none'}\n"
       "the co-event radar lives only in label.tif and is never tiled as input")

# ---- predict ------------------------------------------------------------
cols = pick_channels(names, ck["groups"])
model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
model.load_state_dict(ck["state"]); model.eval()
mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
Xte, Yte, Mte, negte = X[te], Y[te], M[te], is_neg[te]
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                 .cpu().numpy())
P = np.concatenate(P)

# ---- 3 spatial alignment ------------------------------------------------
# If prediction and label were misaligned (flipped or transposed), the true
# pairing would score no better than a deliberately broken one.
lab = (Yte > thr)
def quick_auc(s, y):
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if y.sum() < 10 or (~y).sum() < 10: return np.nan
    r = pd.Series(s).rank().values
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((r[y].sum() - n1*(n1+1)/2)/(n1*n0))
sel = np.linspace(0, len(P)-1, min(400, len(P))).astype(int)
a_true = quick_auc(np.concatenate([P[i][Mte[i]] for i in sel]),
                   np.concatenate([lab[i][Mte[i]] for i in sel]))
a_flip = quick_auc(np.concatenate([P[i][::-1, ::-1][Mte[i]] for i in sel]),
                   np.concatenate([lab[i][Mte[i]] for i in sel]))
a_roll = quick_auc(np.concatenate([np.roll(P[i], 1, axis=0)[Mte[i]] for i in sel]),
                   np.concatenate([lab[i][Mte[i]] for i in sel]))
report(3, "spatial alignment", a_true > a_flip + 0.01,
       f"correct pairing      AUC {a_true:.4f}\n"
       f"flipped 180 degrees  AUC {a_flip:.4f}  (must be clearly worse)\n"
       f"shifted by one pixel AUC {a_roll:.4f}  (near-identical is expected)")

# ---- 6 AUC arithmetic ---------------------------------------------------
s_all = np.concatenate([P[i][Mte[i]] for i in sel])
y_all = np.concatenate([lab[i][Mte[i]] for i in sel])
ours = quick_auc(s_all, y_all)
try:
    from sklearn.metrics import roc_auc_score
    theirs = float(roc_auc_score(y_all, s_all))
    ok6 = abs(ours - theirs) < 1e-6
    d6 = f"ours {ours:.6f}\nsklearn {theirs:.6f}\ndifference {abs(ours-theirs):.2e}"
except Exception as e:
    ok6, d6 = False, f"sklearn unavailable: {e}"
report(6, "AUC arithmetic", ok6, d6)

# ---- 7 IS IT REALLY PER PIXEL ------------------------------------------
wi, be = [], []
for i in range(len(P)):
    m = Mte[i]
    if m.sum() < 500: continue
    wi.append(P[i][m].std()); be.append(P[i][m].mean())
wi, be = np.array(wi), np.array(be)
ratio = float(wi.mean() / (be.std() + 1e-12))
# how much of the pooled score survives if the day-level signal is removed?
per_tile_centred = np.concatenate([(P[i][Mte[i]] - P[i][Mte[i]].mean())
                                  for i in range(len(P)) if Mte[i].sum() >= 500])
y_c = np.concatenate([lab[i][Mte[i]] for i in range(len(P)) if Mte[i].sum() >= 500])
a_pool = quick_auc(np.concatenate([P[i][Mte[i]] for i in range(len(P))
                                   if Mte[i].sum() >= 500]), y_c)
a_centred = quick_auc(per_tile_centred, y_c)
# The per-tile average is the fair measure of spatial skill: it asks the
# ranking question inside each tile separately, so tiles with large prediction
# variance cannot dominate the way they do in a single pooled ranking.
per_tile = []
for i in range(len(P)):
    m = Mte[i]
    if m.sum() < 500: continue
    y_ = lab[i][m]
    if y_.sum() < 5 or (~y_).sum() < 5: continue
    per_tile.append(quick_auc(P[i][m], y_))
per_tile = np.array([x for x in per_tile if np.isfinite(x)])
report(7, "IS IT PER PIXEL", ratio > 0.3 and per_tile.mean() > 0.55,
       f"mean within-tile std of prediction   {wi.mean():.4f}\n"
       f"std of tile means (between tiles)    {be.std():.4f}\n"
       f"within / between ratio               {ratio:.3f}   (>0.3 = genuinely spatial)\n"
       f"pooled AUC                           {a_pool:.4f}\n"
       f"pooled AUC after removing each tile's own mean  {a_centred:.4f}\n"
       f"within-tile AUC, averaged per tile   {per_tile.mean():.4f} "
       f"(n={len(per_tile)} tiles)\n"
       f"  ^ the fair measure of spatial skill. The centred-pooled number above\n"
       f"    is harsher because one pooled ranking lets high-variance tiles\n"
       f"    dominate; per-tile averaging does not.")

print("\n" + "=" * 78)
n_fail = sum(1 for v in results.values() if not v["pass"])
print(f"{len(results) - n_fail}/{len(results)} checks passed"
      + ("" if n_fail == 0 else f"   *** {n_fail} FAILED ***"))
print("=" * 78)
json.dump(results, (ROOT / f"data/audit_pixel_{TAG}.json").open("w"), indent=1)
print(f"wrote data/audit_pixel_{TAG}.json")
