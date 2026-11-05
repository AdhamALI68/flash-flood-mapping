#!/usr/bin/env python3
"""
Corrector v4, the safest kind possible: recalibrate the label using ONLY the
two radar-derived numbers already involved, coherence drop and amplitude
darkening. Nothing else.

WHY THIS CANNOT REPEAT THE EARLIER FAILURES
    Attempt 1 (train_label_corrector.py) leaned on land cover and terrain,
    features ALSO given to the main model as input, so the model just copied
    the corrector's own rule instead of learning anything (validation AUC
    0.99, real skill near chance). Attempt 2 narrowed to cropland alone and
    lost too much signal to be useful. Attempt 3 found no single land-cover
    class stands out (99.2% of raw positives are wrong almost everywhere).

    This attempt uses NEITHER land cover NOR terrain. coh_drop and
    amp_change_db are never handed to the main model as an input feature
    anywhere in this project, they only ever exist as the label. There is
    therefore no shared information for the main model to reverse-engineer
    a shortcut from. If this label predicts better, it is because it is a
    genuinely better label, not because it leaked.

WHAT WAS JUST MEASURED (test_amplitude_label.py)
    On the fullest, least-filtered population: coh_drop alone already reaches
    AUC 0.7642 against real photos - a real, decent signal, not the 0.41 seen
    earlier in a narrower slice. amp_change_db alone only reaches 0.5617. The
    real lever left is calibration: turning that decent ranking into a good
    yes/no decision, given real flooding is only ~0.3% of pixels, not
    inventing a new feature.

Output: data/signal_corrector.json (the fitted model + cutoff),
        data/signal_corrector_result.json (the evaluation numbers)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def metrics(pred, truth):
    tp = float((pred & truth).sum()); fp = float((pred & ~truth).sum())
    fn = float((~pred & truth).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return prec, rec


def main() -> int:
    opt_dir = ROOT / "data/optical_labels"
    feat_dir = ROOT / "data/features"
    opt_events = sorted(p.stem for p in opt_dir.glob("*.tif"))

    amp_l, coh_l, y_l, ev_l = [], [], [], []
    for ev in opt_events:
        lab_p = feat_dir / ev / "label.tif"
        if not lab_p.exists():
            continue
        with rasterio.open(lab_p) as s:
            names = list(s.descriptions)
            if "amp_change_db" not in names or "coh_drop" not in names:
                continue
            lab = s.read()
        amp = lab[names.index("amp_change_db")]
        coh = lab[names.index("coh_drop")]
        with rasterio.open(opt_dir / f"{ev}.tif") as s:
            flood_o, _, valid_o = s.read()
        mask = (np.isfinite(amp) & np.isfinite(coh) & (valid_o > 0.5)
                & np.isfinite(flood_o))
        if mask.sum() < 200:
            continue
        amp_l.append(amp[mask]); coh_l.append(coh[mask])
        y_l.append(flood_o[mask].astype(int))
        ev_l.append(np.full(int(mask.sum()), ev))

    amp = np.concatenate(amp_l); coh = np.concatenate(coh_l)
    y = np.concatenate(y_l).astype(bool); evid = np.concatenate(ev_l)
    print(f"{len(coh):,} pixels from {len(set(evid.tolist()))} events, "
          f"{100 * y.mean():.3f}% optically wet", flush=True)

    uniq = np.array(sorted(set(evid.tolist())))
    rng = np.random.default_rng(0)
    rng.shuffle(uniq)
    n_test = max(1, int(len(uniq) * 0.2))
    test_ev = set(uniq[:n_test])
    te = np.array([e in test_ev for e in evid])
    tr = ~te
    print(f"train events {len(uniq) - n_test}  test events {n_test}  "
          f"train px {tr.sum():,}  test px {te.sum():,}", flush=True)

    F = np.stack([coh, amp], axis=1)
    logit = LogisticRegression(class_weight="balanced", max_iter=2000)
    logit.fit(F[tr], y[tr])

    p_te = logit.predict_proba(F[te])[:, 1]
    p_tr = logit.predict_proba(F[tr])[:, 1]
    auc_new = roc_auc_score(y[te], p_te)
    auc_coh_only = roc_auc_score(y[te], coh[te])
    raw_rule = coh[te] > 0.15
    prec_raw, rec_raw = metrics(raw_rule, y[te])

    # rate-based cutoff, fitted on TRAIN, same discipline as the rest of the
    # project (an absolute cutoff does not transfer, a rate does)
    best_q, best_f1 = 0.003, 0.0
    for q in np.linspace(0.0005, 0.05, 150):
        t = np.quantile(p_tr, 1 - q)
        pred = p_tr >= t
        prec, rec = metrics(pred, y[tr])
        if prec + rec == 0:
            continue
        f1 = 2 * prec * rec / (prec + rec)
        if f1 > best_f1:
            best_q, best_f1 = float(q), f1
    p_cut = float(np.quantile(p_tr, 1 - best_q))
    pred_new = p_te >= p_cut
    prec_new, rec_new = metrics(pred_new, y[te])

    print("\n" + "=" * 70)
    print("SIGNAL-ONLY CORRECTOR (coh_drop + amp_change_db, nothing else)")
    print("=" * 70)
    print(f"  coh_drop alone            AUC {auc_coh_only:.4f}")
    print(f"  corrector (both signals)  AUC {auc_new:.4f}")
    print(f"  raw rule   (coh > 0.15)        precision {prec_raw:.4f}  recall {rec_raw:.4f}")
    print(f"  corrector  (fitted rate)       precision {prec_new:.4f}  recall {rec_new:.4f}  "
          f"flags top {100*best_q:.3f}%")
    print(f"\n  coefficients: coh_drop {logit.coef_[0][0]:.3f}  "
          f"amp_change_db {logit.coef_[0][1]:.3f}  intercept {logit.intercept_[0]:.3f}")

    json.dump({
        "coef_coh": float(logit.coef_[0][0]), "coef_amp": float(logit.coef_[0][1]),
        "intercept": float(logit.intercept_[0]), "prob_cutoff": p_cut,
        "flag_rate": best_q,
        "auc_coh_only": float(auc_coh_only), "auc_corrector": float(auc_new),
        "precision_raw": float(prec_raw), "recall_raw": float(rec_raw),
        "precision_corrector": float(prec_new), "recall_corrector": float(rec_new),
    }, open(ROOT / "data/signal_corrector_result.json", "w"), indent=1)
    print("\nwrote data/signal_corrector_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
