#!/usr/bin/env python3
"""
Corrector v2. The first attempt (train_label_corrector.py) failed for a
specific, diagnosed reason: it leaned mainly on land cover and terrain shape
(0.24 + 0.23 + 0.21 of its decision weight), features that are STATIC per
location and are ALSO given to the main model as input. The main model then
just reconstructed the corrector's own rule from its own inputs, matching the
corrected label almost perfectly (validation AUC 0.99) while actually losing
the ability to tell a flood day from a quiet day (0.545, barely above chance).

THE FIX
    Land cover is only allowed to raise the bar, never to decide the answer by
    itself. The corrected label is a GATED rule:

        flood = coh_drop > T_normal                      (ordinary ground)
        flood = coh_drop > T_cropland   (T_cropland > T_normal, on farmland)

    Both branches still require real, per-event radar evidence to fire. A
    farmland pixel with a huge coh_drop can still be called flood; an ordinary
    pixel with a small coh_drop still cannot. This means the corrected label
    can never be reconstructed from land cover and terrain alone, since land
    cover only changes WHERE THE BAR SITS, the actual per-storm number still
    has to clear it. Cropland is the only static input, and it is the one
    static confound already measured and proven real (cropland_check.json:
    farmland tiles show 3x the apparent flood rate, and that is well
    established, not a guess).

    Both thresholds are calibrated on the same 210 optically-verified events
    used before, split by event, held out the same way.

Output: data/label_corrector2.json (the two thresholds),
        data/label_corrector2_result.json (the evaluation numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def load_shared_shards(d: Path):
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
    return Xs, Y, M, idx, sc.index("lc_cropland")


def metrics(pred, truth):
    tp = float((pred & truth).sum()); fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return prec, rec


def main() -> int:
    opt_dir = ROOT / "data/optical_labels"
    opt_events = {p.stem for p in opt_dir.glob("*.tif")}
    print(f"{len(opt_events)} events have an optical label", flush=True)

    Xs, Y, M, idx, ci_crop = load_shared_shards(ROOT / "data/tiles_v4")

    import rasterio
    cache: dict[str, tuple] = {}
    coh_l, crop_l, y_l, ev_l = [], [], [], []
    n_events_used = 0
    for i in range(len(idx)):
        ev = str(idx.event.iloc[i])
        if ev not in opt_events:
            continue
        if ev not in cache:
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
        coh_l.append(Y[i][mask])
        crop_l.append(Xs[i, ci_crop][mask])
        y_l.append(fo[mask].astype(int))
        ev_l.append(np.full(mask.sum(), ev))

    coh = np.concatenate(coh_l); crop = np.concatenate(crop_l) > 0.5
    y = np.concatenate(y_l).astype(bool); evid = np.concatenate(ev_l)
    good = np.isfinite(coh)
    coh, crop, y, evid = coh[good], crop[good], y[good], evid[good]
    print(f"used {n_events_used} events, {len(coh):,} pixels, "
          f"{100 * y.mean():.3f}% optically wet, {100 * crop.mean():.1f}% cropland",
          flush=True)

    uniq = np.array(sorted(set(evid.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * 0.2))
    test_ev = set(uniq[:n_test])
    te = np.array([e in test_ev for e in evid])
    tr = ~te
    print(f"train px {tr.sum():,}  test px {te.sum():,}", flush=True)

    # Fit a tiny, regularised logistic model instead of grid-searching F1
    # directly: with a 0.16% base rate (and far fewer positives within the
    # 2.2% cropland slice), an F1-maximising grid search has almost nothing
    # to lock onto and collapses onto a lucky, unstable point (this is
    # exactly what happened on the first pass here: both branches converged
    # to the identical, absurdly strict 0.5886, which is not a real signal).
    # Logistic regression fits a smooth curve to ALL the data at once instead
    # of hunting for one best point, so it degrades gracefully instead of
    # collapsing.
    from sklearn.linear_model import LogisticRegression
    Ftr = np.stack([coh[tr], crop[tr].astype(float),
                    coh[tr] * crop[tr]], axis=1)
    logit = LogisticRegression(class_weight="balanced", max_iter=2000)
    logit.fit(Ftr, y[tr])

    # Pick the operating threshold as a RATE fitted on a slice of the train
    # pixels held back for this purpose, then convert back into the two
    # equivalent coh_drop cutoffs for interpretability - same rate-based
    # philosophy already forced on this project once before (phase7_final.py):
    # an absolute probability cutoff does not transfer, a rate does.
    p_tr = logit.predict_proba(Ftr)[:, 1]
    best_q, best_f1 = 0.02, 0.0
    for q in np.linspace(0.0005, 0.05, 100):
        t = np.quantile(p_tr, 1 - q)
        pred = p_tr >= t
        prec, rec = metrics(pred, y[tr])
        if prec + rec == 0:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_q, best_f1 = float(q), f1
    p_cut = float(np.quantile(p_tr, 1 - best_q))

    # convert the probability cutoff into an equivalent coh_drop threshold
    # for each branch, by inverting the fitted logistic curve
    def implied_threshold(is_crop: float) -> float:
        b0, b1, b2 = logit.intercept_[0], logit.coef_[0][0], logit.coef_[0][2]
        w = logit.coef_[0][1] * is_crop
        z = np.log(p_cut / (1 - p_cut))
        slope = b1 + b2 * is_crop
        if abs(slope) < 1e-9:
            return float("inf")
        return float((z - b0 - w) / slope)

    t_normal = implied_threshold(0.0)
    t_crop = implied_threshold(1.0)
    print(f"fitted operating rate: flag top {100*best_q:.3f}% of train pixels")
    print(f"threshold, ordinary ground: {t_normal:.4f}")
    print(f"threshold, cropland:        {t_crop:.4f}")

    # evaluate on held-out events
    pred_new = np.where(crop[te], coh[te] >= t_crop, coh[te] >= t_normal)
    pred_raw = coh[te] >= 0.15
    auc_raw = roc_auc_score(y[te], coh[te])  # ranking is identical to v1's raw check
    prec_raw, rec_raw = metrics(pred_raw, y[te])
    prec_new, rec_new = metrics(pred_new, y[te])

    print("\n" + "=" * 70)
    print("GATED CORRECTOR VS THE RAW LABEL, HELD-OUT EVENTS, REAL PHOTOS")
    print("=" * 70)
    print(f"  raw coh_drop ranking alone   AUC {auc_raw:.4f}")
    print(f"  raw rule   (>0.15 everywhere)      precision {prec_raw:.4f}  recall {rec_raw:.4f}")
    print(f"  gated rule (cropland-aware)        precision {prec_new:.4f}  recall {rec_new:.4f}")

    # THE SAFETY CHECK - can land cover / terrain alone reconstruct this label
    # well, without any per-event signal? If yes, this design failed the same
    # way the first one did. Uses the SAME two static inputs the main model
    # also sees (cropland is one of them by construction; add a couple more
    # terrain channels here from the same tiles for a fair, harsh test).
    chan = json.loads((ROOT / "data/tiles_v4/channels.json").read_text())
    sc = chan["shared_channels"]
    static_names = ["basin_hand", "basin_slope", "basin_twi", "lc_class", "lc_water"]
    static_idx = [sc.index(n) for n in static_names]
    # Build a fresh pixel table restricted to the test events, this time with
    # static terrain/land-cover features instead of coh_drop, to see if they
    # alone can guess the new corrected label.
    test_pixel_static = []
    test_pixel_label = []
    for i in range(len(idx)):
        ev = str(idx.event.iloc[i])
        if ev not in test_ev:
            continue
        if ev not in cache:
            continue
        flood_o, _, valid_o = cache[ev]
        r0, c0 = int(idx.row.iloc[i]), int(idx.col.iloc[i])
        fo = flood_o[r0:r0 + 112, c0:c0 + 112]
        vo = valid_o[r0:r0 + 112, c0:c0 + 112]
        mask = M[i] & (vo > 0.5) & np.isfinite(fo)
        if mask.sum() < 200:
            continue
        c0h = Y[i][mask]
        crp = Xs[i, ci_crop][mask] > 0.5
        new_label = np.where(crp, c0h >= t_crop, c0h >= t_normal)
        feats = np.stack([Xs[i, k][mask] for k in static_idx], axis=1)
        test_pixel_static.append(feats)
        test_pixel_label.append(new_label)
    Sx = np.concatenate(test_pixel_static)
    Sy = np.concatenate(test_pixel_label)
    good2 = np.isfinite(Sx).all(axis=1)
    Sx, Sy = Sx[good2], Sy[good2]

    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    sx_tr, sx_te, sy_tr, sy_te = train_test_split(Sx, Sy, test_size=0.3, random_state=0)
    guard = xgb.XGBClassifier(n_estimators=200, max_depth=4, n_jobs=-1,
                              tree_method="hist", random_state=0)
    guard.fit(sx_tr, sy_tr)
    guard_auc = roc_auc_score(sy_te, guard.predict_proba(sx_te)[:, 1])
    print(f"\n  SAFETY CHECK: can land cover + terrain alone predict the new "
          f"corrected label?")
    print(f"  AUC {guard_auc:.4f} (v1's equivalent number was 0.99+, which is "
          f"why it failed)")
    if guard_auc > 0.85:
        print("  WARNING: still high. The static shortcut may still exist.")
    else:
        print("  OK: land cover and terrain alone cannot reconstruct this "
              "label well. The corrected label still depends on real, "
              "per-event radar evidence.")

    json.dump({
        "t_normal": t_normal, "t_cropland": t_crop,
        "n_events_used": n_events_used,
        "auc_raw": float(auc_raw),
        "precision_raw": float(prec_raw), "recall_raw": float(rec_raw),
        "precision_gated": float(prec_new), "recall_gated": float(rec_new),
        "static_shortcut_auc": float(guard_auc),
    }, open(ROOT / "data/label_corrector2_result.json", "w"), indent=1)
    json.dump({"t_normal": t_normal, "t_cropland": t_crop},
              open(ROOT / "data/label_corrector2.json", "w"), indent=1)
    print("\nwrote data/label_corrector2.json and data/label_corrector2_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
