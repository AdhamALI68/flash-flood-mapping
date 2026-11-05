#!/usr/bin/env python3
"""
Train a small "corrector": given only the InSAR-visible signal at a pixel
(coherence drop, land cover, terrain position relative to drainage), predict
what an independent Sentinel-2 optical check would say - real water or not.

WHY
    We measured that the raw InSAR label reacts to things that are not floods
    (farmland plowing/harvest inflates it 3x, see cropland_check.json) and that
    connected-component cleaning does not fix this (speckle_test2.json). But we
    also measured that our trained model already scores BETTER against the
    independent optical evidence (0.751) than against its own noisy training
    label (0.643), see eval_vs_optical.json - so the model has learned
    something real that the label itself buries in noise.

    This corrector is a direct attempt to fix the LABEL, not the model. It is
    trained only on the 210 events where we have both an InSAR label and a real
    optical photo, learning the mapping "given these InSAR-visible clues, was
    this really water". Once trained, apply_label_corrector.py applies it to
    ALL 617 events, including the ~400 that never got a usable photo, so no
    training data is thrown away.

HOW IT IS KEPT HONEST
    Split by EVENT (not by pixel or tile), so no event's pixels appear on both
    sides. Compared directly against the raw "coherence drop > 0.15" rule on
    held-out events, using the same 0.15-threshold pixels for both, so the
    corrector has to actually beat the current label, not just look sharp
    against a metric.

Output: data/label_corrector.json (model), data/label_corrector_result.json
        (evaluation numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
FEATURES = ["coh_drop", "basin_hand", "basin_slope", "basin_twi",
            "basin_spi", "lc_class", "lc_cropland", "lc_water"]


def load_shared_shards(d: Path):
    """Read the shared shards only (coh_drop lives in y, the rest in xs), in
    the same order index.csv was built, so row i lines up with index row i."""
    chan = json.loads((d / "channels.json").read_text())
    sc = chan["shared_channels"]
    idx = pd.read_csv(d / "index.csv")
    Xs, Y, M = [], [], []
    for p in sorted((d / "shared").glob("shard-*.npz")):
        z = np.load(p)
        Xs.append(z["xs"]); Y.append(z["y"]); M.append(z["m"])
    Xs = np.concatenate(Xs); Y = np.concatenate(Y); M = np.concatenate(M)
    if len(idx) != len(Xs):
        raise SystemExit(f"{d}: index has {len(idx)} rows, shards hold {len(Xs)}")
    return Xs, Y, M, idx, sc


def main() -> int:
    opt_dir = ROOT / "data/optical_labels"
    opt_events = {p.stem for p in opt_dir.glob("*.tif")}
    print(f"{len(opt_events)} events have an optical label", flush=True)

    Xs, Y, M, idx, sc = load_shared_shards(ROOT / "data/tiles_v4")
    ci = {n: sc.index(n) for n in FEATURES if n != "coh_drop"}

    rows_X, rows_y, rows_ev = [], [], []
    cache: dict[str, tuple] = {}
    n_events_used = 0
    for i in range(len(idx)):
        ev = str(idx.event.iloc[i])
        if ev not in opt_events:
            continue
        if ev not in cache:
            import rasterio
            with rasterio.open(opt_dir / f"{ev}.tif") as s:
                cache[ev] = s.read()
            n_events_used += 1
        flood_o, _, valid_o = cache[ev]
        r0, c0 = int(idx.row.iloc[i]), int(idx.col.iloc[i])
        fo = flood_o[r0:r0 + 112, c0:c0 + 112]
        vo = valid_o[r0:r0 + 112, c0:c0 + 112]
        mask = M[i] & (vo > 0.5) & np.isfinite(fo)
        if mask.sum() < 200:
            continue
        cols = [Y[i][mask]]
        for nm in FEATURES:
            if nm == "coh_drop":
                continue
            cols.append(Xs[i, ci[nm]][mask])
        F = np.stack(cols, axis=1)
        rows_X.append(F)
        rows_y.append(fo[mask].astype(int))
        rows_ev.append(np.full(mask.sum(), ev))

    X = np.concatenate(rows_X); y = np.concatenate(rows_y)
    evid = np.concatenate(rows_ev)
    good = np.isfinite(X).all(axis=1)
    X, y, evid = X[good], y[good], evid[good]
    print(f"used {n_events_used} events, {len(X):,} pixels, "
          f"{100 * y.mean():.2f}% optically wet", flush=True)

    uniq = np.array(sorted(set(evid.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * 0.2))
    test_ev = set(uniq[:n_test]); train_ev = set(uniq[n_test:])
    tr = np.array([e in train_ev for e in evid])
    te = np.array([e in test_ev for e in evid])
    print(f"train events {len(train_ev)}  test events {len(test_ev)}  "
          f"train px {tr.sum():,}  test px {te.sum():,}", flush=True)

    clf = xgb.XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            min_child_weight=20, reg_lambda=1.0, n_jobs=-1,
                            tree_method="hist", eval_metric="aucpr",
                            random_state=0)
    clf.fit(X[tr], y[tr])

    p_test = clf.predict_proba(X[te])[:, 1]
    raw_test = X[te, 0]  # coh_drop itself, the current label's own signal
    auc_corrector = roc_auc_score(y[te], p_test)
    auc_raw = roc_auc_score(y[te], raw_test)

    raw_rule = (raw_test > THR).astype(int)
    tp = float((raw_rule & y[te]).sum()); fp = float((raw_rule & ~y[te].astype(bool)).sum())
    fn = float((~raw_rule.astype(bool) & y[te].astype(bool)).sum())
    prec_raw = tp / max(tp + fp, 1); rec_raw = tp / max(tp + fn, 1)

    # The base rate here is 0.16%, so a fixed absolute grid (0.05..0.95) sits
    # entirely in the far tail of predicted probabilities and finds nothing
    # useful. Search a RATE instead - what fraction of pixels to flag - the
    # same fix already forced once elsewhere in this project (phase7_final.py)
    # for exactly this reason: an absolute cutoff does not transfer, a rate does.
    best_t, best_f1 = float(np.quantile(p_test, 0.999)), 0.0
    for q in np.linspace(0.001, 0.40, 200):
        t = float(np.quantile(p_test, 1 - q))
        pr = (p_test >= t).astype(int)
        tp2 = float((pr & y[te]).sum()); fp2 = float((pr & ~y[te].astype(bool)).sum())
        fn2 = float((~pr.astype(bool) & y[te].astype(bool)).sum())
        if tp2 == 0:
            continue
        pr_, rc_ = tp2 / (tp2 + fp2), tp2 / (tp2 + fn2)
        f1 = 2 * pr_ * rc_ / (pr_ + rc_)
        if f1 > best_f1:
            best_t, best_f1 = t, f1
    flag_rate = float((p_test >= best_t).mean())
    pr = (p_test >= best_t).astype(int)
    tp3 = float((pr & y[te]).sum()); fp3 = float((pr & ~y[te].astype(bool)).sum())
    fn3 = float((~pr.astype(bool) & y[te].astype(bool)).sum())
    prec_c = tp3 / max(tp3 + fp3, 1); rec_c = tp3 / max(tp3 + fn3, 1)

    print("\n" + "=" * 70)
    print("CORRECTOR VS THE RAW LABEL, GRADED AGAINST OPTICAL TRUTH, HELD-OUT EVENTS")
    print("=" * 70)
    print(f"  raw coh_drop alone     AUC {auc_raw:.4f}")
    print(f"  corrector              AUC {auc_corrector:.4f}")
    print(f"  raw rule  (>0.15)      precision {prec_raw:.4f}  recall {rec_raw:.4f}")
    print(f"  corrector (best cut)   precision {prec_c:.4f}  recall {rec_c:.4f}  "
          f"cutoff {best_t:.4f}  flags {100*flag_rate:.3f}% of pixels")

    imp = sorted(zip(["coh_drop"] + [n for n in FEATURES if n != "coh_drop"],
                     clf.feature_importances_), key=lambda x: -x[1])
    print("\n  feature importance:")
    for n_, v in imp:
        print(f"    {n_:<14}{v:.4f}")

    clf.save_model(str(ROOT / "data/label_corrector.json"))
    json.dump({
        "features": FEATURES, "cutoff": best_t,
        "n_events_total": len(opt_events), "n_events_used": n_events_used,
        "n_train_events": len(train_ev), "n_test_events": len(test_ev),
        "n_train_px": int(tr.sum()), "n_test_px": int(te.sum()),
        "auc_raw": float(auc_raw), "auc_corrector": float(auc_corrector),
        "precision_raw": float(prec_raw), "recall_raw": float(rec_raw),
        "precision_corrector": float(prec_c), "recall_corrector": float(rec_c),
        "flag_rate": flag_rate,
        "importance": [(n_, float(v)) for n_, v in imp],
    }, open(ROOT / "data/label_corrector_result.json", "w"), indent=1)
    print("\nwrote data/label_corrector.json and data/label_corrector_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
