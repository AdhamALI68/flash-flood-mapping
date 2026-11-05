#!/usr/bin/env python3
"""
Figure for the RANDOM-split model.

WHAT THIS SHOWS, AND WHAT IT DOES NOT
    The random split tests on ground that also appears in training: 80% of our
    events lie within 5 km of another event, so most test terrain was seen. This
    measures "how well does it work where we already have data", which is a real
    and useful question, but it is NOT transfer to new ground.

    Reported honestly it is fine. Reported as a transfer result it is not, and
    the caption here says so.

Tiles are chosen to MATCH the reported operating point, so what you see is what
the numbers mean - not the best tiles.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, random_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / "data/model_rec_random.pt", map_location=dev, weights_only=False)

X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = random_split(idx, seed=ck.get("seed", 0))

cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                prior=ck.get("prior", "downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

def pred(sel):
    k = np.flatnonzero(sel); P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                     .cpu().numpy())
    return np.concatenate(P), k

Pva, kva = pred(va)
Pte, kte = pred(te)
sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
rate, bf = 0.2, 0.0
for q in np.linspace(0.02, 0.60, 120):
    p = sv >= np.quantile(sv, 1-q)
    tp = float((p & yv).sum()); fp = float((p & ~yv).sum()); fn = float((~p & yv).sum())
    if tp == 0: continue
    pr, rc = tp/(tp+fp), tp/(tp+fn); f = 2*pr*rc/(pr+rc)
    if f > bf: rate, bf = float(q), f
st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])
yt = np.concatenate([(Y[i] > THR)[M[i]] for i in kte])
cut = float(np.quantile(st, 1-rate))
p = st >= cut
tp = float((p & yt).sum()); fp = float((p & ~yt).sum())
fn = float((~p & yt).sum()); tn = float((~p & ~yt).sum())
R, P_, A = tp/(tp+fn), tp/(tp+fp), (tp+tn)/len(p)
print(f"operating point: recall {R:.4f} precision {P_:.4f} accuracy {A:.4f} "
      f"(flagging {100*rate:.0f}%)")

rec = np.full(len(Pte), np.nan); prc = np.full(len(Pte), np.nan)
acc = np.full(len(Pte), np.nan)
for j, i in enumerate(kte):
    mk = M[i]
    if mk.sum() < 800: continue
    t = (Y[i] > THR)[mk]; pb = (Pte[j] >= cut)[mk]
    if t.sum() < 60: continue
    tp_ = float((t & pb).sum()); fp_ = float((~t & pb).sum())
    fn_ = float((t & ~pb).sum()); tn_ = float((~t & ~pb).sum())
    rec[j] = tp_/(tp_+fn_); prc[j] = tp_/max(tp_+fp_, 1)
    acc[j] = (tp_+tn_)/t.size
d = (np.abs(rec-R)/R + np.abs(prc-P_)/P_ + np.abs(acc-A)/A)
pick = np.flatnonzero(np.isfinite(d))[np.argsort(d[np.isfinite(d)])][:3]

cmap = ListedColormap(["#eef0f2", "#d94a3d", "#e9a13b", "#2f8f5b"])
fig, ax = plt.subplots(3, 3, figsize=(9.4, 9.6))
for r_i, j in enumerate(pick):
    i = kte[j]; mk = M[i]
    t = (Y[i] > THR); pb = Pte[j] >= cut
    cls = np.zeros(t.shape, int); cls[t & ~pb] = 1; cls[~t & pb] = 2; cls[t & pb] = 3
    for c, (arr, ttl, cm, vr) in enumerate((
            (t.astype(float), "Observed flood (InSAR)", "Reds", (0, 1)),
            (Pte[j], "Model prediction", "Reds",
             (0, max(0.25, float(np.nanpercentile(Pte[j][mk], 99))))),
            (cls, "green hit · red missed · orange false alarm", cmap, (0, 3)))):
        a_ = ax[r_i, c]
        a_.imshow(np.where(mk, arr, np.nan), cmap=cm, vmin=vr[0], vmax=vr[1],
                  interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        if r_i == 0: a_.set_title(ttl, fontsize=9.5)
    ax[r_i, 0].set_ylabel(f"event {idx.event.iloc[i]}\nregion {idx.huc4.iloc[i]}",
                          fontsize=8.5)
    ax[r_i, 2].set_xlabel(f"recall {rec[j]:.2f} · precision {prc[j]:.2f} · "
                          f"accuracy {acc[j]:.2f}", fontsize=8.5)
fig.suptitle("Flood prediction, 20 m pixels — random split (test ground also "
             "appears in training)\n"
             f"recall {R:.2f} · precision {P_:.2f} · accuracy {A:.2f} · "
             f"AUC 0.826 · cutoff fitted on validation", fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "data/report/figure_random_matched.png", dpi=140)
print("wrote data/report/figure_random_matched.png")
