#!/usr/bin/env python3
"""
Same figure style as before, but the truth is the CLEANED label (salt removed
<15px 8-conn, holes filled <20px 4-conn) instead of the raw label.

Four panels: observed flood (cleaned), predicted probability, predicted
(binary at the fitted cutoff), and hit/miss/false-alarm.

This is on the held-out test regions (1606, 1809), the headline model, so
numbers are directly comparable to the raw-label figures already made.
"""
from __future__ import annotations
import sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, pandas as pd, torch
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
THR = 0.15
MIN_RED, MIN_HOLE = 15, 20
dev = "cuda" if torch.cuda.is_available() else "cpu"


def clean(mask):
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= MIN_RED; keep[0] = False
        mask = keep[lab]
    dry = ~mask
    lab2, n2 = ndimage.label(dry, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n2 > 0:
        border = set(lab2[0,:]) | set(lab2[-1,:]) | set(lab2[:,0]) | set(lab2[:,-1])
        sizes = np.bincount(lab2.ravel())
        fill = np.zeros(sizes.size, bool)
        for lb in range(1, sizes.size):
            if lb not in border and sizes[lb] < MIN_HOLE:
                fill[lb] = True
        mask = mask | fill[lab2]
    return mask


ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                prior=ck.get("prior", "downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

def predict(sel):
    k = np.flatnonzero(sel); P = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                     .cpu().numpy())
    return np.concatenate(P), k

Pva, kva = predict(va)
Pte, kte = predict(te)

# clean the truth for EVERY test tile, then build cutoff + pixel arrays
lab_clean = {}
for j, i in enumerate(kte):
    lab_clean[j] = clean(np.where(M[i], Y[i] > THR, False))

sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
op, bf = 0.5, 0.0
for t in np.quantile(sv, np.linspace(0.5, 0.999, 120)):
    p = sv >= t
    tp=float((p&yv).sum()); fp=float((p&~yv).sum()); fn=float((~p&yv).sum())
    if tp==0: continue
    pr,rc = tp/(tp+fp), tp/(tp+fn); f=2*pr*rc/(pr+rc)
    if f>bf: op,bf = float(t), f

st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])
yt_clean = np.concatenate([lab_clean[j][M[i]] for j, i in enumerate(kte)])
pred_t = st >= op
tp=float((pred_t&yt_clean).sum()); fp=float((pred_t&~yt_clean).sum())
fn=float((~pred_t&yt_clean).sum()); tn=float((~pred_t&~yt_clean).sum())
R = tp/(tp+fn); Pr = tp/(tp+fp); ACC=(tp+tn)/len(yt_clean)
from sklearn.metrics import roc_auc_score
AUC = roc_auc_score(yt_clean.astype(int), st)
print(f"CLEANED-label scoring: AUC {AUC:.4f} recall {R:.4f} precision {Pr:.4f} "
      f"accuracy {ACC:.4f}  (cutoff {op:.4f}, fit on validation, raw label)")

# pick 3 tiles close to these numbers
# Pick tiles that MATCH the reported overall recall/precision AND actually
# have predicted pixels - over half the test tiles predict nothing at this
# cutoff, and a naive "closest to median F1" search kept landing on those,
# giving three blank, uninformative panels.
rect = np.full(len(Pte), np.nan); prct = np.full(len(Pte), np.nan)
haspred = np.zeros(len(Pte), bool)
for j, i in enumerate(kte):
    mk = M[i]
    if mk.sum() < 500: continue
    t = lab_clean[j][mk]; pb = (Pte[j] >= op)[mk]
    if t.sum() < 50: continue
    haspred[j] = pb.any()
    tp_=float((t&pb).sum()); fp_=float((~t&pb).sum()); fn_=float((t&~pb).sum())
    rect[j]=tp_/(tp_+fn_) if tp_+fn_ else 0
    prct[j]=tp_/(tp_+fp_) if tp_+fp_ else 0
ok = np.flatnonzero(np.isfinite(rect) & haspred)
d = np.abs(rect[ok]-R)/max(R,1e-6) + np.abs(prct[ok]-Pr)/max(Pr,1e-6)
pick = ok[np.argsort(d)][:3]

cmap = ListedColormap(["#eef0f2", "#d94a3d", "#e9a13b", "#2f8f5b"])
fig, ax = plt.subplots(len(pick), 4, figsize=(12.6, 3.1*len(pick)))
for r, j in enumerate(pick):
    i = kte[j]; mk = M[i]
    t = lab_clean[j]; pr_prob = Pte[j]; pb = pr_prob >= op
    cls = np.zeros(t.shape, int); cls[t & ~pb] = 1; cls[~t & pb] = 2; cls[t & pb] = 3
    tp_=float((t&pb)[mk].sum()); fp_=float((~t&pb)[mk].sum())
    fn_=float((t&~pb)[mk].sum()); tn_=float((~t&~pb)[mk].sum())
    rr = tp_/(tp_+fn_) if tp_+fn_ else 0
    pp = tp_/(tp_+fp_) if tp_+fp_ else 0
    aa = (tp_+tn_)/mk.sum()
    panels = [
        (t.astype(float), "Observed flood\n(cleaned label)", "Reds", (0, 1)),
        (pr_prob, "Predicted probability", "Reds",
         (0, max(0.25, float(np.nanpercentile(pr_prob[mk], 99))))),
        (pb.astype(float), "Predicted (at cutoff)", "Reds", (0, 1)),
        (cls, "green hit · red missed\norange false alarm", cmap, (0, 3)),
    ]
    for c, (arr, ttl, cm, vr) in enumerate(panels):
        a_ = ax[r, c]
        a_.imshow(np.where(mk, arr, np.nan), cmap=cm, vmin=vr[0], vmax=vr[1],
                  interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        if r == 0: a_.set_title(ttl, fontsize=9.5)
    ax[r, 0].set_ylabel(f"event {idx.event.iloc[i]}\nregion {idx.huc4.iloc[i]}",
                        fontsize=8.5)
    ax[r, 3].set_xlabel(f"recall {rr:.2f} · precision {pp:.2f} · accuracy {aa:.2f}",
                        fontsize=8.5)
fig.suptitle(f"Model vs the CLEANED label, held-out regions · overall AUC "
             f"{AUC:.3f} · recall {R:.2f} · precision {Pr:.2f} · cutoff on validation",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = ROOT / "data/report/figure_cleaned_label.png"
fig.savefig(out, dpi=140)
print(f"wrote {out}")
