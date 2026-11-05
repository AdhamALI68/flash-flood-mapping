#!/usr/bin/env python3
"""
3-way ensemble (average softmax probability across seeds), scored the honest
way (cutoff fit on validation, applied to test), plus a result-picture folder.

    ./.venv/bin/python -u phase5/make_ensemble_folder.py \
        --tags clean_region,clean_region_seed1,clean_region_seed2 \
        --tiles data/tiles_clean --neg data/tiles_clean_neg \
        --out handoff/results_clean_region_ensemble --base 24 --n 8
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True, help="comma-separated model tags to average")
    ap.add_argument("--tiles", default="data/tiles_clean")
    ap.add_argument("--neg", default="data/tiles_clean_neg")
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", type=int, default=24)
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()
    tags = args.tags.split(",")

    OUT = ROOT / args.out; OUT.mkdir(parents=True, exist_ok=True)
    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, "10")
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / args.neg, "10")
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)
    tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])

    nets, muS, sdS, colsS = [], [], [], []
    for tag in tags:
        ck = torch.load(ROOT / f"data/model_{tag}.pt", map_location=dev, weights_only=False)
        cols = pick_channels(names, ck["groups"])
        net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=args.base,
                        prior=ck.get("prior", "downhill")).to(dev)
        net.load_state_dict(ck["state"]); net.eval()
        nets.append(net); colsS.append(cols)
        muS.append(torch.from_numpy(ck["mean"]).view(1,-1,1,1).to(dev))
        sdS.append(torch.from_numpy(ck["std"]).view(1,-1,1,1).to(dev))
    print(f"ensemble of {len(nets)} models: {tags}")

    def predict(sel):
        k = np.flatnonzero(sel); P = []
        with torch.no_grad():
            for i in range(0, len(k), 32):
                probs = []
                for net, cols, mu, sd in zip(nets, colsS, muS, sdS):
                    xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
                    probs.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0].cpu().numpy())
                P.append(np.mean(probs, axis=0))
        return np.concatenate(P), k

    Pva, kva = predict(va); Pte, kte = predict(te)
    sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
    yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
    op, bf = 0.5, 0.0
    for t in np.quantile(sv, np.linspace(0.5, 0.999, 150)):
        p = sv >= t
        tp=float((p&yv).sum()); fp=float((p&~yv).sum()); fn=float((~p&yv).sum())
        if tp == 0: continue
        pr, rc = tp/(tp+fp), tp/(tp+fn); f = 2*pr*rc/(pr+rc)
        if f > bf: op, bf = float(t), f

    st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])
    yt = np.concatenate([(Y[i] > THR)[M[i]] for i in kte]).astype(int)
    AUC = roc_auc_score(yt, st); AP = average_precision_score(yt, st)
    pred = st >= op
    tp=float((pred&yt).sum()); fp=float((pred&~yt).sum())
    fn=float((~pred&yt).sum()); tn=float((~pred&~yt).sum())
    R=tp/(tp+fn); Pr=tp/(tp+fp); ACC=(tp+tn)/len(yt)
    print(f"ensemble: AUC {AUC:.4f}  AP {AP:.4f}  recall {R:.4f}  "
          f"precision {Pr:.4f}  accuracy {ACC:.4f}  cutoff {op:.4f}  n={len(yt):,}")

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
    pick = ok[np.argsort(d)][:args.n]

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
                  (pr_prob, "Predicted probability\n(3-model ensemble)", "Reds",
                   (0, max(0.25, float(np.nanpercentile(pr_prob[mk], 99))))),
                  (pb.astype(float), "Predicted (at cutoff)", "Reds", (0,1)),
                  (cls, "green hit · red missed\norange false alarm", cmap, (0,3))]
        for c, (arr, ttl, cm, vr) in enumerate(panels):
            a_ = ax[c]
            a_.imshow(np.where(mk, arr, np.nan), cmap=cm, vmin=vr[0], vmax=vr[1],
                      interpolation="nearest")
            a_.set_xticks([]); a_.set_yticks([]); a_.set_title(ttl, fontsize=9.5)
        fig.suptitle(f"[ensemble]  event {idx.event.iloc[i]}  region {idx.huc4.iloc[i]}  "
                     f"recall {rr:.2f} precision {pp:.2f} accuracy {aa:.2f}", fontsize=10)
        fig.tight_layout(rect=[0,0,1,0.90])
        fig.savefig(OUT / f"result_{n_img:02d}_event{idx.event.iloc[i]}.png", dpi=130)
        plt.close(fig)
    print(f"wrote {len(pick)} images to {OUT}")
    json.dump({"tags": tags, "auc": float(AUC), "ap": float(AP), "recall": float(R),
               "precision": float(Pr), "accuracy": float(ACC), "cutoff": op},
              open(ROOT / "data/clean_region_ensemble_final.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
