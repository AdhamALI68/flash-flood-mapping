#!/usr/bin/env python3
"""
The area question: will a flash flood be reported in this 10 km square today?

Positive: a flood square on its report day (NOAA Storm Events, written by people
who saw the flood). Negative: the same place on a quiet day, same season, no
flood reported in the river basin within 14 days (phase4/select_negatives.py).
Train regions 1501, 1503, 1810; validation 1507; test 1606, 1809, never seen.

Two models, both fit or tuned on train and validation only:
  unet  : the headline pixel model, tile scores averaged or maxed per square
          (aggregation chosen on validation)
  trees : gradient-boosted trees on per-square summaries of the same inputs
          (rain, upstream routed rain and runoff, soil, terrain)
Two baselines: rain alone (24 h rain, square mean), and the shuffled label.

Cutoff chosen on validation (largest recall minus false-alarm rate), applied to
test once. Precision depends on how common flood days are, so it is reported
three ways: at the test mix (flood squares outnumber quiet ones, inflated), at
50/50, and at 1 flood day in 5.

The "rainy quiet days" rows keep only quiet days with at least 5 mm of rain in
24 h over the square, so a high score cannot come from "it rained" alone.

    ./.venv/bin/python -u phase7/area_day_test.py

Output: data/area_day_test.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from model import FloodUNet  # noqa: E402
from train import load_tiles, pick_channels  # noqa: E402

HEAD = "terrain+basin+rain+rain_routed+soil+landcover"
TRAIN, VAL, TEST = {"1501", "1503", "1810"}, {"1507"}, {"1606", "1809"}
RAINY_MM = 5.0
SUMMARY = ["rain_1h_peak", "rain_24h", "rain_ante48h", "rain_ante72h", "rain_cover_frac",
           "rain_1h_upstream_depth_mm", "rain_24h_upstream_depth_mm", "runoff_1h_upstream_depth_mm",
           "rain_1h_over_ksat_upstream_mean", "soil_ksat", "soil_clay", "slope_10m", "basin_hand", "basin_twi"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def tile_table(tiles, neg):
    X, Y, M, idx, names = load_tiles(ROOT / tiles, "10")
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / neg, "10")
    X = np.concatenate([X, Xn]); M = np.concatenate([M, Mn]); del Xn
    idx = pd.concat([idx.assign(flood=1), idxn.assign(flood=0)], ignore_index=True)
    idx["huc4"] = idx.huc4.astype(str); idx["event"] = idx.event.astype(str)
    return X, M, idx, names


def unet_tile_scores(X, M, names):
    ck = torch.load(ROOT / f"data/model_{HEAD}.pt", map_location=dev, weights_only=False)
    cols = pick_channels(names, ck["groups"])
    net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24, prior=ck.get("prior", "downhill")).to(dev)
    net.load_state_dict(ck["state"]); net.eval()
    mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
    sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)
    mean_p, top_p = np.zeros(len(X)), np.zeros(len(X))
    with torch.no_grad():
        for s in range(0, len(X), 32):
            xb = torch.from_numpy(X[s:s + 32][:, cols]).to(dev)
            P = torch.sigmoid(net(torch.nan_to_num((xb - mu) / sd)))[:, 0].cpu().numpy()
            for j, p in enumerate(P):
                v = p[M[s + j]]
                mean_p[s + j] = v.mean() if v.size else 0.0
                top_p[s + j] = np.quantile(v, 0.9) if v.size else 0.0
    return mean_p, top_p


def square_table(X, M, idx, names, mean_p, top_p):
    rows = []
    ci = {n: names.index(n) for n in SUMMARY}
    for ev, g in idx.groupby("event"):
        k = g.index.values
        r = {"event": ev, "huc4": g.huc4.iloc[0], "flood": int(g.flood.iloc[0]), "tiles": len(k),
             "unet_mean": float(mean_p[k].mean()), "unet_max": float(mean_p[k].max()),
             "unet_top": float(top_p[k].max())}
        for n, c in ci.items():
            vals = np.concatenate([X[t, c][M[t]] for t in k])
            vals = vals[np.isfinite(vals)]
            r[f"{n}_mean"] = float(vals.mean()) if vals.size else np.nan
            r[f"{n}_p90"] = float(np.quantile(vals, 0.9)) if vals.size else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def youden_cutoff(y, s):
    best, cut = -1.0, float(np.median(s))
    for t in np.unique(s):
        p = s >= t
        j = p[y == 1].mean() - p[y == 0].mean()
        if j > best:
            best, cut = j, float(t)
    return cut


def report(y, s, cut):
    y = np.asarray(y); s = np.asarray(s)
    p = s >= cut
    tpr = float(p[y == 1].mean()); fpr = float(p[y == 0].mean())
    prec_mix = float((p & (y == 1)).sum() / max(p.sum(), 1))
    def prec_at(share):
        return tpr * share / max(tpr * share + fpr * (1 - share), 1e-9)
    return {"auc": round(float(roc_auc_score(y, s)), 3), "recall": round(tpr, 3), "false_alarm_rate": round(fpr, 3),
            "precision_test_mix": round(prec_mix, 3), "precision_50_50": round(prec_at(0.5), 3),
            "precision_1_in_5": round(prec_at(0.2), 3), "n_flood": int((y == 1).sum()), "n_quiet": int((y == 0).sum())}


def main() -> int:
    X, M, idx, names = tile_table("data/tiles_v4", "data/tiles_neg_v4")
    print(f"{len(X)} tiles loaded, scoring with the headline model on {dev}", flush=True)
    mean_p, top_p = unet_tile_scores(X, M, names)
    sq = square_table(X, M, idx, names, mean_p, top_p)
    del X
    sq.to_csv(ROOT / "data/area_day_squares.csv", index=False)
    tr, va, te = sq[sq.huc4.isin(TRAIN)], sq[sq.huc4.isin(VAL)], sq[sq.huc4.isin(TEST)]
    print(f"squares: train {len(tr)} ({tr.flood.sum()} flood), val {len(va)} ({va.flood.sum()} flood), "
          f"test {len(te)} ({te.flood.sum()} flood)\n", flush=True)

    feats = [c for c in sq.columns if c.endswith("_mean") or c.endswith("_p90")]
    feats = [c for c in feats if not c.startswith("unet")]
    out = {}

    agg = max(("unet_mean", "unet_max", "unet_top"), key=lambda c: roc_auc_score(va.flood, va[c]))
    trees = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                           min_samples_leaf=10, class_weight="balanced", random_state=0)
    trees.fit(tr[feats], tr.flood)
    rng = np.random.default_rng(0)
    shuf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                          min_samples_leaf=10, class_weight="balanced", random_state=0)
    shuf.fit(tr[feats], rng.permutation(tr.flood.values))
    scores = {
        f"unet ({agg})": (va[agg].values, te[agg].values),
        "trees on square summaries": (trees.predict_proba(va[feats])[:, 1], trees.predict_proba(te[feats])[:, 1]),
        "rain alone (24 h, square mean)": (va["rain_24h_mean"].values, te["rain_24h_mean"].values),
        "shuffled-label trees": (shuf.predict_proba(va[feats])[:, 1], shuf.predict_proba(te[feats])[:, 1]),
    }
    rainy = (te.flood.values == 1) | (te.rain_24h_mean.values >= RAINY_MM)
    for name, (sv, st) in scores.items():
        cut = youden_cutoff(va.flood.values, sv)
        out[name] = {"all quiet days": report(te.flood.values, st, cut),
                     f"rainy quiet days only (>= {RAINY_MM:g} mm)": (report(te.flood.values[rainy], st[rainy], cut)
                                                                    if (te.flood.values[rainy] == 0).sum() >= 5 else None)}
        print(name)
        for k, v in out[name].items():
            print(f"   {k:34s} {v}")
    out["_info"] = {"aggregation_chosen_on_validation": agg, "rainy_mm": RAINY_MM,
                    "test_quiet_days_rainy": int(((te.flood == 0) & (te.rain_24h_mean >= RAINY_MM)).sum()),
                    "test_quiet_days": int((te.flood == 0).sum())}
    print("\n", out["_info"])
    json.dump(out, open(ROOT / "data/area_day_test.json", "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
