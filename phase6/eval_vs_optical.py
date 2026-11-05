#!/usr/bin/env python3
"""
Phase 6 - THE experiment: is the InSAR label the ceiling?

    ./.venv/bin/python -u phase6/eval_vs_optical.py

THE CLAIM TO TEST
    Every result in this project is graded against InSAR coherence drop, and we
    measured that label's limit: 5 looks per pixel gives about +/-0.09 noise on
    the drop against a 0.15 threshold, and a 3-sigma cut fires on 0.00% of
    pixels.

    If that noise is really what caps performance, then scoring the SAME model,
    on the SAME pixels, against a cleaner label must give a higher number. If it
    does not, the model is simply weak and we say so.

THE COMPARISON IS STRICTLY CONTROLLED
    Same model weights. Same events. Same pixels - only pixels where BOTH labels
    are defined are used, so the two scores cannot differ because one label
    covered more ground. Only the answer key changes.

WHAT A POSITIVE RESULT MEANS
    Not "our model is better than we thought". It means the measurement
    instrument was the limit, which is a finding about how floods should be
    labelled from satellites - useful to anyone doing this, and a stronger
    contribution than another flood model.

Output: data/eval_vs_optical.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet                                    # noqa: E402
from train import load_tiles, pick_channels, region_split       # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


def metrics(score, truth):
    truth = truth.astype(bool)
    if truth.sum() < 30 or (~truth).sum() < 30:
        return None
    auc = roc_auc_score(truth.astype(int), score)
    ap = average_precision_score(truth.astype(int), score)
    best = (0.0, 0.0, 0.0)
    for t in np.quantile(score, np.linspace(0.50, 0.999, 60)):
        p = score >= t
        tp = float((p & truth).sum()); fp = float((p & ~truth).sum())
        fn = float((~p & truth).sum())
        if tp == 0:
            continue
        pr, rc = tp / (tp + fp), tp / (tp + fn)
        f1 = 2 * pr * rc / (pr + rc)
        if f1 > best[0]:
            best = (f1, pr, rc)
    return {"auc": float(auc), "ap": float(ap), "f1": best[0],
            "precision": best[1], "recall": best[2],
            "n": int(truth.size), "base": float(truth.mean())}


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--tag", default="terrain+basin+rain+rain_routed+soil+landcover")
    ap_.add_argument("--threshold", type=float, default=0.15)
    args = ap_.parse_args()

    opt_dir = ROOT / "data/optical_labels"
    have = {p.stem for p in opt_dir.glob("*.tif")}
    if not have:
        raise SystemExit("no optical labels yet - run phase4/optical_labels.py")
    print(f"{len(have)} events have an optical label", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(ROOT / f"data/model_{args.tag}.pt", map_location=dev,
                    weights_only=False)
    X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
    idx["event"] = idx.event.astype(str)
    sel = idx.event.isin(have).values
    if sel.sum() == 0:
        raise SystemExit("no tiles from those events")
    print(f"{int(sel.sum())} tiles from {idx.event[sel].nunique()} of them",
          flush=True)

    cols = pick_channels(names, ck["groups"])
    m = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
    m.load_state_dict(ck["state"]); m.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

    Xs, Ys, Ms = X[sel], Y[sel], M[sel]
    ixs = idx[sel].reset_index(drop=True)
    P = []
    with torch.no_grad():
        for i in range(0, len(Xs), 32):
            xb = torch.from_numpy(Xs[i:i + 32][:, cols]).to(dev)
            P.append(torch.sigmoid(m(torch.nan_to_num((xb - mu) / sd)))[:, 0]
                     .cpu().numpy())
    P = np.concatenate(P)

    # The optical label lives on each event's full grid; cut the same window
    # each tile came from so the pixels line up exactly.
    cache: dict[str, tuple] = {}
    s_all, insar_all, opt_all, ev_all = [], [], [], []
    for i in range(len(P)):
        e = ixs.event.iloc[i]
        if e not in cache:
            with rasterio.open(opt_dir / f"{e}.tif") as s:
                nm = [s.descriptions[k] for k in range(s.count)]
                a = s.read()
            cache[e] = (a[nm.index("flood_optical")], a[nm.index("valid")] > 0.5)
        of, ov = cache[e]
        r0, c0 = int(ixs.row.iloc[i]), int(ixs.col.iloc[i])
        t = 112
        if r0 + t > of.shape[0] or c0 + t > of.shape[1]:
            continue
        ofw, ovw = of[r0:r0 + t, c0:c0 + t], ov[r0:r0 + t, c0:c0 + t]
        both = Ms[i] & ovw & np.isfinite(Ys[i])
        if both.sum() < 200:
            continue
        s_all.append(P[i][both])
        insar_all.append((Ys[i] > args.threshold)[both])
        opt_all.append(ofw[both] > 0.5)
        ev_all.append(np.full(int(both.sum()), e))

    if not s_all:
        raise SystemExit("no overlapping pixels between the two labels")
    s = np.concatenate(s_all)
    y_insar = np.concatenate(insar_all)
    y_opt = np.concatenate(opt_all)
    ev = np.concatenate(ev_all)

    r_insar = metrics(s, y_insar)
    r_opt = metrics(s, y_opt)

    print("\n" + "=" * 76)
    print("SAME MODEL, SAME PIXELS, TWO DIFFERENT ANSWER KEYS")
    print("=" * 76)
    print(f"  pixels compared      {s.size:,}   from "
          f"{len(set(ev.tolist()))} events")
    print(f"  {'':<26}{'AUC':>9}{'AP':>9}{'recall':>9}{'prec':>9}{'flooded':>10}")
    for nm, r in (("InSAR coherence label", r_insar),
                  ("Sentinel-2 optical label", r_opt)):
        if r is None:
            print(f"  {nm:<26}   too few positives to score")
            continue
        print(f"  {nm:<26}{r['auc']:>9.4f}{r['ap']:>9.4f}{r['recall']:>9.4f}"
              f"{r['precision']:>9.4f}{100*r['base']:>9.2f}%")

    if r_insar and r_opt:
        d = r_opt["auc"] - r_insar["auc"]
        print(f"\n  difference in AUC: {d:+.4f}")
        if d > 0.05:
            print("  VERDICT: the model scores clearly HIGHER against the")
            print("           independent optical label. The InSAR coherence")
            print("           label was the ceiling, not the model.")
        elif d > 0.0:
            print("  VERDICT: higher against optical, but by a small margin.")
        else:
            print("  VERDICT: NO better against the cleaner label. The limit is")
            print("           the model or the features, not the answer key.")
        print("\n  agreement between the two labels themselves:")
        agree = float((y_insar == y_opt).mean())
        both = float((y_insar & y_opt).sum())
        print(f"    pixels where they agree      {100*agree:.1f}%")
        print(f"    pixels both call flood       {int(both):,}")
        print(f"    InSAR says flood             {int(y_insar.sum()):,}")
        print(f"    optical says flood           {int(y_opt.sum()):,}")

    json.dump({"tag": args.tag, "n_pixels": int(s.size),
               "n_events": len(set(ev.tolist())),
               "insar": r_insar, "optical": r_opt},
              (ROOT / "data/eval_vs_optical.json").open("w"), indent=1)
    print("\nwrote data/eval_vs_optical.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
