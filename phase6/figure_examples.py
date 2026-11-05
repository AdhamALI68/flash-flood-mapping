#!/usr/bin/env python3
"""
Phase 6 - the paper figure: what the model predicts, at a stated recall.

    ./.venv/bin/python -u phase6/figure_examples.py --recall 0.80

WHY THE EARLIER MAPS WERE CONFUSING
    They used the cutoff that maximises F1 (0.176), where overall recall is
    0.53, and they picked examples by AUC. AUC measures ranking, so a tile could
    score 0.91 while the model never crossed the cutoff there - showing a map
    with recall 0.11 and no detections. Useless as an illustration.

    This sets the cutoff to reach a STATED recall over the whole test set, then
    shows tiles closest to the median performance at that cutoff. What you see
    is what the model does in normal operation, not its best or worst day.

THREE PANELS, NOT FIVE
    truth | prediction | overlay. The inputs are described in the text; putting
    them in the figure was crowding out the comparison that matters.

Output: data/report/figure_examples.png
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "terrain+basin+rain+rain_routed+soil+landcover"

ap = argparse.ArgumentParser()
ap.add_argument("--recall", type=float, default=0.0,
                help="cutoff reaching this recall. 0 = use the F1-optimal "
                     "cutoff, which is the honest default operating point")
ap.add_argument("--block", type=int, default=1,
                help="aggregate to k x k blocks before scoring. k=8 gives the "
                     "160 m cells the paper recommends, where averaging 64 "
                     "label pixels cuts the coherence noise 8x")
ap.add_argument("--select", default="match", choices=["match", "median", "best"],
                help="median = typical performance, the honest default. "
                     "'best' is retained only as a diagnostic and must NOT be "
                     "used for a figure: the top-F1 tiles turn out to be the "
                     "~10 of 1918 that are >50% flooded, where a constant "
                     "'flood everywhere' prediction scores 0.96 precision. "
                     "Median tile flood fraction is 0.002; those tiles are 0.571")
ap.add_argument("--n", type=int, default=4)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
_, _, te = region_split(idx, ["1606", "1809"], ["1507"])

cols = pick_channels(names, ck["groups"])
m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
              prior=ck.get("prior", "downhill")).to(dev)
m.load_state_dict(ck["state"]); m.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
Xte, Yte, Mte = X[te], Y[te], M[te]
ixte = idx[te].reset_index(drop=True)
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i+32][:, cols]).to(dev)
        P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0].cpu().numpy())
P = np.concatenate(P)
lab = Yte > ck["threshold"]

if args.block > 1:
    k_ = args.block
    def blk(a, how="mean"):
        h, w = (a.shape[1] // k_) * k_, (a.shape[2] // k_) * k_
        r = a[:, :h, :w].reshape(a.shape[0], h // k_, k_, w // k_, k_)
        return r.mean(axis=(2, 4))
    Mb = blk(Mte.astype("float32")) > 0.999      # fully observed blocks only
    Pb = blk(P)
    Lb = blk(lab.astype("float32")) >= 0.30      # 30% of the block flooded
    P, lab, Mte = Pb, Lb, Mb
    print(f"aggregated to {k_}x{k_} blocks = {k_*20} m cells")

# cutoff that reaches the requested recall over the WHOLE test set
s = np.concatenate([P[i][Mte[i]] for i in range(len(P))])
y = np.concatenate([lab[i][Mte[i]] for i in range(len(P))])
if args.recall > 0:
    o = np.argsort(-s)
    need = int(np.ceil(args.recall * y.sum()))
    cut = float(s[o][np.searchsorted(np.cumsum(y[o]), need)])
else:
    cut, bf = 0.5, 0.0
    for t_ in np.quantile(s, np.linspace(0.50, 0.999, 120)):
        pd_ = s >= t_
        tp_ = float((pd_ & y).sum()); fp_ = float((pd_ & ~y).sum())
        fn_ = float((~pd_ & y).sum())
        if tp_ == 0: continue
        pr_, rc_ = tp_/(tp_+fp_), tp_/(tp_+fn_)
        f_ = 2*pr_*rc_/(pr_+rc_)
        if f_ > bf: cut, bf = float(t_), f_
pred = s >= cut
tp = float((pred & y).sum()); fp = float((pred & ~y).sum()); fn = float((~pred & y).sum())
tn = float((~pred & ~y).sum())
R, Pr = tp/(tp+fn), tp/(tp+fp)
ACC = (tp+tn)/len(y)
print(f"cutoff {cut:.4f}  ->  recall {R:.4f}  precision {Pr:.4f}  accuracy {ACC:.4f}")

# tiles closest to the median recall at this cutoff
rec = np.full(len(P), np.nan)
f1t = np.full(len(P), np.nan)
prc = np.full(len(P), np.nan)
acc = np.full(len(P), np.nan)
for i in range(len(P)):
    mk = Mte[i]
    if mk.sum() < (60 if args.block > 1 else 500): continue
    t, p_ = lab[i][mk], (P[i] >= cut)[mk]
    if t.sum() < (8 if args.block > 1 else 50): continue
    rec[i] = (t & p_).sum() / t.sum()
    tp_ = float((t & p_).sum()); fp_ = float((~t & p_).sum())
    tn_ = float((~t & ~p_).sum())
    acc[i] = (tp_ + tn_) / t.size
    if tp_ + fp_ > 0:
        prc[i] = tp_ / (tp_ + fp_)
        pr_ = tp_/(tp_+fp_)
        f1t[i] = 2*pr_*rec[i]/(pr_+rec[i]) if pr_+rec[i] > 0 else 0.0
ok = np.flatnonzero(np.isfinite(f1t))
med = np.median(f1t[ok])
if args.select == "match":
    # Tiles whose OWN recall/precision/accuracy are closest to the aggregate
    # numbers we report. A reported metric is an average over 23.5 M pixels;
    # no single tile equals it, so "typical" has to mean "closest to the
    # reported figures", not "median F1" - those are different tiles.
    d = (np.abs(rec - R) / max(R, 1e-6)
         + np.abs(prc - Pr) / max(Pr, 1e-6)
         + np.abs(acc - ACC) / max(ACC, 1e-6))
    ok2 = np.flatnonzero(np.isfinite(d))
    pick = ok2[np.argsort(d[ok2])][:args.n]
    print(f"showing {len(pick)} tiles closest to the reported "
          f"recall {R:.2f} / precision {Pr:.2f} / accuracy {ACC:.2f}")
elif args.select == "best":
    pick = ok[np.argsort(-f1t[ok])][:args.n]
    print(f"median per-tile F1 {med:.3f}; showing the {len(pick)} BEST tiles "
          f"- these are selected examples, not typical performance")
else:
    pick = ok[np.argsort(np.abs(f1t[ok] - med))][:args.n]
    print(f"median per-tile F1 {med:.3f}; showing {len(pick)} typical tiles")

cm = ListedColormap(["#eef0f2", "#d94a3d", "#e9a13b", "#2f8f5b"])
fig, axes = plt.subplots(len(pick), 3, figsize=(9.4, 3.05*len(pick)))
if len(pick) == 1: axes = axes[None, :]
for r, i in enumerate(pick):
    mk = Mte[i]
    t, pb = lab[i], P[i] >= cut
    cls = np.zeros(t.shape, int); cls[t & ~pb] = 1; cls[~t & pb] = 2; cls[t & pb] = 3
    tp_ = float((t & pb)[mk].sum()); fn_ = float((t & ~pb)[mk].sum())
    fp_ = float((~t & pb)[mk].sum()); tn_ = float((~t & ~pb)[mk].sum())
    rr, pp = tp_/(tp_+fn_), tp_/max(tp_+fp_, 1)
    aa = (tp_+tn_)/mk.sum()
    for c, (arr, ttl, cmap, vr) in enumerate((
            (t.astype(float), "Observed flood (InSAR)", "Reds", (0, 1)),
            (P[i], "Predicted probability", "Reds", (0, float(np.nanpercentile(P[i][mk], 99.5)))),
            (cls, "green hit · red missed · orange false alarm", cm, (0, 3)))):
        a_ = axes[r, c]
        a_.imshow(np.where(mk, arr, np.nan), cmap=cmap, vmin=vr[0], vmax=vr[1],
                  interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        if r == 0: a_.set_title(ttl, fontsize=9.5)
    axes[r, 0].set_ylabel(f"event {ixte.event.iloc[i]}\nregion {ixte.huc4.iloc[i]}",
                          fontsize=8.5)
    axes[r, 2].set_xlabel(f"recall {rr:.2f} · precision {pp:.2f} · accuracy {aa:.2f}",
                          fontsize=8.5)
note = " — SELECTED EXAMPLES, not typical" if args.select == "best" else ""
fig.suptitle(f"Flood prediction at {20*args.block} m, held-out regions{note}"
             f" · operating point: recall {R:.2f}, precision {Pr:.2f}, "
             f"accuracy {ACC:.2f}", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.97])
out = ROOT / f"data/report/figure_{20*args.block}m_{args.select}.png"
fig.savefig(out, dpi=140); plt.close(fig)
json.dump({"cutoff": cut, "recall": R, "precision": Pr, "accuracy": ACC,
           "median_tile_recall": float(med)},
          (ROOT / "data/figure_examples.json").open("w"), indent=1)
print(f"wrote {out.relative_to(ROOT)}")
