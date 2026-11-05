#!/usr/bin/env python3
"""
Fair head-to-head: every model graded by the SAME answer key (Sentinel-2 photo
label on flood days, no-flood on quiet days), on the same held-out test pixels,
with the cutoff fit on validation the same way for every model.

    ./.venv/bin/python -u phase5/compare_on_photo_ruler.py \
        --tags terrain+basin+rain+rain_routed+soil+landcover,direct_photo_v2

Precision is also shown as lift (precision / base rate), because the photo
label marks well under 1% of pixels as flood and raw precision alone hides that.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
dev = "cuda" if torch.cuda.is_available() else "cpu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", required=True)
    ap.add_argument("--tiles", default="data/tiles_direct")
    ap.add_argument("--neg", default="data/tiles_direct_neg")
    ap.add_argument("--out", default="data/compare_photo_ruler.json")
    args = ap.parse_args()

    X, Y, M, idx, names = load_tiles(ROOT / args.tiles, "10")
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / args.neg, "10")
    n_flood = len(X)
    X, Y, M = np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn])
    del Xn, Yn, Mn
    idx = pd.concat([idx, idxn], ignore_index=True)
    quiet = np.arange(len(X)) >= n_flood
    _, va, te = region_split(idx, ["1606", "1809"], ["1507"])
    kva, kte = np.flatnonzero(va), np.flatnonzero(te)

    yv = np.concatenate([(Y[i] > THR)[M[i]] for i in kva])
    yt = np.concatenate([(Y[i] > THR)[M[i]] for i in kte])
    base = float(yt.mean())
    print(f"test: {len(kte)} tiles ({int(quiet[kte].sum())} quiet), "
          f"{len(yt):,} pixels, photo flood rate {100*base:.3f}%\n")

    results = {}
    for tag in args.tags.split(","):
        ck = torch.load(ROOT / f"data/model_{tag}.pt", map_location=dev, weights_only=False)
        cols = pick_channels(names, ck["groups"])
        net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"],
                        base=ck.get("base", 24), prior=ck.get("prior", "downhill")).to(dev)
        net.load_state_dict(ck["state"]); net.eval()
        mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
        sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

        def predict(k):
            P = []
            with torch.no_grad():
                for s in range(0, len(k), 32):
                    xb = torch.from_numpy(X[k[s:s+32]][:, cols]).to(dev)
                    P.append(torch.sigmoid(net(torch.nan_to_num((xb - mu) / sd)))[:, 0].cpu().numpy())
            return np.concatenate(P)

        Pva, Pte = predict(kva), predict(kte)
        sv = np.concatenate([Pva[j][M[i]] for j, i in enumerate(kva)])
        st = np.concatenate([Pte[j][M[i]] for j, i in enumerate(kte)])

        # rate-based cutoff search on validation (same rule for every model)
        op, bf = float(np.quantile(sv, 0.999)), 0.0
        for q in np.linspace(0.00005, 0.40, 300):
            t = float(np.quantile(sv, 1 - q)); p = sv >= t
            tp = float((p & yv).sum()); fp = float((p & ~yv).sum()); fn = float((~p & yv).sum())
            if tp == 0:
                continue
            pr, rc = tp / (tp + fp), tp / (tp + fn); f = 2 * pr * rc / (pr + rc)
            if f > bf:
                op, bf = t, f

        pred = st >= op
        tp = float((pred & yt).sum()); fp = float((pred & ~yt).sum())
        fn = float((~pred & yt).sum()); tn = float((~pred & ~yt).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        tile_mean = np.array([Pte[j][M[i]].mean() for j, i in enumerate(kte)])
        r = {"auc": roc_auc_score(yt, st), "ap": average_precision_score(yt, st),
             "recall": tp / (tp + fn), "precision": prec, "lift": prec / base,
             "accuracy": (tp + tn) / len(yt), "cutoff": op,
             "flood_vs_quiet_auc": roc_auc_score(~quiet[kte], tile_mean)}
        # quiet days are all "no flood" and easy; score flood days alone too
        kf = [j for j, i in enumerate(kte) if not quiet[i]]
        yf = np.concatenate([(Y[kte[j]] > THR)[M[kte[j]]] for j in kf])
        sf = np.concatenate([Pte[j][M[kte[j]]] for j in kf])
        r["auc_flood_days_only"] = roc_auc_score(yf, sf)
        r["ap_flood_days_only"] = average_precision_score(yf, sf)
        # precision each model reaches at the same recall: no cutoff choice involved
        pc, rc, _ = precision_recall_curve(yt, st)
        for target in (0.25, 0.50):
            r[f"precision_at_recall_{target:.2f}"] = float(pc[rc >= target].max())
        results[tag] = {k: round(float(v), 4) for k, v in r.items()}
        print(f"{tag}")
        print("   " + "  ".join(f"{k} {v:.4f}" for k, v in results[tag].items()))

    results["_test"] = {"tiles": int(len(kte)), "pixels": int(len(yt)), "flood_rate": base}
    json.dump(results, open(ROOT / args.out, "w"), indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
