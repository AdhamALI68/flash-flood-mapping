#!/usr/bin/env python3
"""
The 'after' folder: score the clean+random model properly (pixel-level AUC,
cutoff fit on validation, applied to test), and make result images matching
the earlier figure style - observed(cleaned) / predicted probability /
predicted at cutoff / hit-miss-falsealarm.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, random_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "clean_random"
THR = 0.15   # the cleaned tiles store y = +1.0 flood / -1.0 dry, so >0.15 still works
OUT = ROOT / "handoff/after"
OUT.mkdir(parents=True, exist_ok=True)
dev = "cuda" if torch.cuda.is_available() else "cpu"

ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_clean", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_clean_neg", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = random_split(idx, seed=ck.get("seed", 0))

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
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0].cpu().numpy())
    return np.concatenate(P), k

Pva, kva = predict(va)
Pte, kte = predict(te)
sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
op, bf = 0.5, 0.0
for t in np.quantile(sv, np.linspace(0.5, 0.999, 150)):
    p = sv >= t
    tp=float((p&yv).sum()); fp=float((p&~yv).sum()); fn=float((~p&yv).sum())
    if tp==0: continue
    pr,rc = tp/(tp+fp), tp/(tp+fn); f=2*pr*rc/(pr+rc)
    if f>bf: op,bf = float(t), f

st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])
yt = np.concatenate([(Y[i] > THR)[M[i]] for i in kte]).astype(int)
AUC = roc_auc_score(yt, st); AP = average_precision_score(yt, st)
pred_t = st >= op
tp=float((pred_t&yt).sum()); fp=float((pred_t&~yt).sum())
fn=float((~pred_t&yt).sum()); tn=float((~pred_t&~yt).sum())
R=tp/(tp+fn); Pr=tp/(tp+fp); ACC=(tp+tn)/len(yt)
print(f"FINAL (clean labels, random split): AUC {AUC:.4f}  AP {AP:.4f}  "
      f"recall {R:.4f}  precision {Pr:.4f}  accuracy {ACC:.4f}  "
      f"(cutoff {op:.4f} fit on validation)  n={len(yt):,}  base={yt.mean():.4f}")

rec = np.full(len(Pte), np.nan); prc = np.full(len(Pte), np.nan)
for j, i in enumerate(kte):
    mk = M[i]
    if mk.sum() < 500: continue
    t = (Y[i] > THR)[mk]; pb = (Pte[j] >= op)[mk]
    if t.sum() < 30 or not pb.any(): continue
    tp_=float((t&pb).sum()); fp_=float((~t&pb).sum()); fn_=float((t&~pb).sum())
    rec[j]=tp_/(tp_+fn_) if tp_+fn_ else 0; prc[j]=tp_/(tp_+fp_) if tp_+fp_ else 0
ok = np.flatnonzero(np.isfinite(rec))
d = np.abs(rec[ok]-R)/max(R,1e-6) + np.abs(prc[ok]-Pr)/max(Pr,1e-6)
pick = ok[np.argsort(d)][:8]

cmap = ListedColormap(["#eef0f2", "#d94a3d", "#e9a13b", "#2f8f5b"])
for n_img, j in enumerate(pick, 1):
    i = kte[j]; mk = M[i]
    t = (Y[i] > THR); pr_prob = Pte[j]; pb = pr_prob >= op
    cls = np.zeros(t.shape, int); cls[t & ~pb] = 1; cls[~t & pb] = 2; cls[t & pb] = 3
    tp_=float((t&pb)[mk].sum()); fp_=float((~t&pb)[mk].sum())
    fn_=float((t&~pb)[mk].sum()); tn_=float((~t&~pb)[mk].sum())
    rr=tp_/(tp_+fn_) if tp_+fn_ else 0; pp=tp_/(tp_+fp_) if tp_+fp_ else 0
    aa=(tp_+tn_)/mk.sum()
    fig, ax = plt.subplots(1, 4, figsize=(12.6, 3.15))
    panels = [(t.astype(float), "Observed flood\n(cleaned label)", "Reds", (0,1)),
              (pr_prob, "Predicted probability", "Reds",
               (0, max(0.25, float(np.nanpercentile(pr_prob[mk], 99))))),
              (pb.astype(float), "Predicted (at cutoff)", "Reds", (0,1)),
              (cls, "green hit · red missed\norange false alarm", cmap, (0,3))]
    for c, (arr, ttl, cm, vr) in enumerate(panels):
        a_ = ax[c]
        a_.imshow(np.where(mk, arr, np.nan), cmap=cm, vmin=vr[0], vmax=vr[1],
                  interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([]); a_.set_title(ttl, fontsize=9.5)
    fig.suptitle(f"event {idx.event.iloc[i]}  region {idx.huc4.iloc[i]}  "
                 f"recall {rr:.2f} precision {pp:.2f} accuracy {aa:.2f}", fontsize=10)
    fig.tight_layout(rect=[0,0,1,0.90])
    fig.savefig(OUT / f"result_{n_img:02d}_event{idx.event.iloc[i]}.png", dpi=130)
    plt.close(fig)
    print(f"  wrote result_{n_img:02d}", flush=True)

json.dump({"auc": float(AUC), "ap": float(AP), "recall": float(R),
           "precision": float(Pr), "accuracy": float(ACC), "cutoff": op,
           "n_pixels": int(len(yt)), "base_rate": float(yt.mean())},
          (ROOT / "data/clean_random_final.json").open("w"), indent=1)
print(f"\nwrote {len(pick)} images to {OUT}")
