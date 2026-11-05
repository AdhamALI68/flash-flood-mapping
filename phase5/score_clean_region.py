#!/usr/bin/env python3
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "clean_region"; THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_clean", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_clean_neg", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])
cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24, prior=ck.get("prior","downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1,-1,1,1).to(dev); sd = torch.from_numpy(ck["std"]).view(1,-1,1,1).to(dev)
def predict(sel):
    k = np.flatnonzero(sel); P=[]
    with torch.no_grad():
        for i in range(0,len(k),32):
            xb = torch.from_numpy(X[k[i:i+32]][:,cols]).to(dev)
            P.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:,0].cpu().numpy())
    return np.concatenate(P), k
Pva, kva = predict(va); Pte, kte = predict(te)
sv = np.concatenate([Pva[j][M[i]] for j,i in enumerate(kva)])
yv = np.concatenate([(Y[i]>THR)[M[i]] for i in kva])
op, bf = 0.5, 0.0
for t in np.quantile(sv, np.linspace(0.5,0.999,150)):
    p = sv>=t
    tp=float((p&yv).sum()); fp=float((p&~yv).sum()); fn=float((~p&yv).sum())
    if tp==0: continue
    pr,rc = tp/(tp+fp), tp/(tp+fn); f=2*pr*rc/(pr+rc)
    if f>bf: op,bf = float(t), f
st = np.concatenate([Pte[j][M[i]] for j,i in enumerate(kte)])
yt = np.concatenate([(Y[i]>THR)[M[i]] for i in kte]).astype(int)
AUC = roc_auc_score(yt, st); AP = average_precision_score(yt, st)
pred = st>=op
tp=float((pred&yt).sum()); fp=float((pred&~yt).sum()); fn=float((~pred&yt).sum()); tn=float((~pred&~yt).sum())
R=tp/(tp+fn); Pr=tp/(tp+fp); ACC=(tp+tn)/len(yt)
print(f"CLEAN + REGION (honest split): AUC {AUC:.4f}  AP {AP:.4f}  recall {R:.4f}  "
      f"precision {Pr:.4f}  accuracy {ACC:.4f}  cutoff {op:.4f}  n={len(yt):,}  base={yt.mean():.4f}")
json.dump({"auc":float(AUC),"ap":float(AP),"recall":float(R),"precision":float(Pr),
           "accuracy":float(ACC),"cutoff":op,"n_pixels":int(len(yt)),"base_rate":float(yt.mean())},
          open(ROOT/"data/clean_region_final.json","w"), indent=1)
