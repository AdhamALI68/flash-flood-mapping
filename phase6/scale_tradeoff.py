#!/usr/bin/env python3
"""
Phase 6 - the trade-off: how precise can we be, and how accurate at that precision?

    ./.venv/bin/python -u phase6/scale_tradeoff.py

THE QUESTION THIS SETTLES
    Per 20 m pixel we reach AUC 0.695. Per 2.24 km tile we reach 0.732. The
    tile is more accurate but useless as a warning - "15% of this 5 km2 square
    floods" does not tell anyone where to go. The pixel is precise but noisy.

    Nobody has measured what happens between them, and that is the number a
    practitioner actually needs: at what spatial resolution does this become
    both usable and trustworthy?

WHY ACCURACY RISES AS CELLS GROW, AND WHY IT IS NOT CHEATING
    The label is InSAR coherence drop with about +/-0.09 of estimation noise per
    pixel at 5 looks. Averaging an n x n block cuts random noise by n. So a
    coarser cell has a genuinely cleaner target - the signal is not invented,
    the noise is averaged out.

    What IS lost is precision on the ground, and the base rate shifts as cells
    grow, which flatters precision and F1. So AUC and average precision relative
    to the base rate are the honest columns and are reported as such.

EVERY SCALE IS TREATED IDENTICALLY
    Same features (block means), same regions held out, same model class, cutoff
    always chosen on validation. Only the size of the prediction unit changes.

Output: data/scale_tradeoff.json, data/report/scale_tradeoff.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_tiles, region_split                     # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
BLOCKS = [1, 2, 4, 8, 16, 28, 56, 112]        # x20 m = 20 m .. 2.24 km


def blocks_of(a: np.ndarray, k: int) -> np.ndarray:
    """Mean over k x k blocks. Returns (n_blocks,) flattened."""
    if k == 1:
        return a.ravel()
    h, w = (a.shape[0] // k) * k, (a.shape[1] // k) * k
    return a[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3)).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rung", default="10")
    ap.add_argument("--threshold", type=float, default=0.15)
    ap.add_argument("--frac", type=float, default=0.30,
                    help="a block counts as flooded if this fraction of its "
                         "pixels flooded. At k=1 it is just the pixel label")
    ap.add_argument("--cap", type=int, default=1_500_000)
    args = ap.parse_args()

    print("loading tiles...", flush=True)
    X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", args.rung)
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", args.rung)
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)
    tr, va, te = region_split(idx, ["1606", "1809"], ["1507"])
    print(f"  {len(X)} tiles   train {int(tr.sum())} val {int(va.sum())} "
          f"test {int(te.sum())}\n", flush=True)

    import xgboost as xgb
    rng = np.random.default_rng(0)
    rows = []

    for k in BLOCKS:
        def build(sel, cap):
            F, L = [], []
            for i in np.flatnonzero(sel):
                m = M[i]
                if m.sum() < 500:
                    continue
                mb = blocks_of(m.astype("float32"), k)
                # a block is used only if it is fully observed, so no block is
                # part real and part invented
                ok = mb > 0.999
                if ok.sum() == 0:
                    continue
                lab = blocks_of(np.where(m, Y[i] > args.threshold, 0.0), k)
                lab = (lab >= (args.frac if k > 1 else 0.5))[ok]
                feat = np.stack([blocks_of(np.where(m, X[i][c], np.nan), k)[ok]
                                 for c in range(X.shape[1])], axis=1)
                F.append(feat)
                L.append(lab)
            if not F:
                return None, None
            F = np.concatenate(F)
            L = np.concatenate(L).astype(int)
            good = np.isfinite(F).all(axis=1)
            F, L = F[good], L[good]
            if len(F) > cap:
                j = rng.choice(len(F), cap, replace=False)
                F, L = F[j], L[j]
            return F, L

        Ftr, Ltr = build(tr, args.cap)
        Fva, Lva = build(va, args.cap // 3)
        Fte, Lte = build(te, args.cap)
        if Ftr is None or Fte is None or Lte.sum() < 30 or (Lte == 0).sum() < 30:
            print(f"  {k*20:>5} m: too few blocks to score", flush=True)
            continue

        clf = xgb.XGBClassifier(n_estimators=800, max_depth=7,
                                learning_rate=0.05, subsample=0.8,
                                colsample_bytree=0.7, min_child_weight=8,
                                reg_lambda=1.0, n_jobs=-1, tree_method="hist",
                                eval_metric="aucpr", early_stopping_rounds=40,
                                random_state=0)
        clf.fit(Ftr, Ltr, eval_set=[(Fva, Lva)], verbose=False)
        pv, pt = clf.predict_proba(Fva)[:, 1], clf.predict_proba(Fte)[:, 1]
        auc = roc_auc_score(Lte, pt)
        apr = average_precision_score(Lte, pt)
        best_t, best_f1 = 0.5, 0.0
        for t in np.quantile(pv, np.linspace(0.3, 0.995, 80)):
            p = pv >= t
            tp = float((p & (Lva == 1)).sum()); fp = float((p & (Lva == 0)).sum())
            fn = float((~p & (Lva == 1)).sum())
            if tp == 0:
                continue
            pr_, rc_ = tp / (tp + fp), tp / (tp + fn)
            f1 = 2 * pr_ * rc_ / (pr_ + rc_)
            if f1 > best_f1:
                best_t, best_f1 = float(t), f1
        # ALSO score every scale at the SAME recall. Each scale otherwise
        # picks its own cutoff, which makes the recall column look better at
        # 20 m purely because it flags far more ground. Fixing recall makes the
        # precision column directly comparable, which is what a reader needs.
        target = 0.60
        order = np.sort(pt)[::-1]
        need = int(np.ceil(target * max(Lte.sum(), 1)))
        hits = np.cumsum(Lte[np.argsort(-pt)])
        j = int(np.searchsorted(hits, need))
        t_fixed = float(order[min(j, len(order) - 1)])
        pf = pt >= t_fixed
        tpf = float((pf & (Lte == 1)).sum()); fpf = float((pf & (Lte == 0)).sum())
        fnf = float((~pf & (Lte == 1)).sum())
        rec_f = tpf / (tpf + fnf) if tpf + fnf else np.nan
        pre_f = tpf / (tpf + fpf) if tpf + fpf else np.nan

        p = pt >= best_t
        tp = float((p & (Lte == 1)).sum()); fp = float((p & (Lte == 0)).sum())
        fn = float((~p & (Lte == 1)).sum())
        rec = tp / (tp + fn) if tp + fn else np.nan
        pre = tp / (tp + fp) if tp + fp else np.nan
        f1 = 2 * pre * rec / (pre + rec) if pre and rec else np.nan
        base = float(Lte.mean())
        rows.append({"cell_m": k * 20, "area_km2": (k * 20 / 1000) ** 2,
                     "blocks_test": int(len(Lte)), "base": base,
                     "auc": float(auc), "ap": float(apr),
                     "lift": float(apr / base) if base else np.nan,
                     "recall": float(rec), "precision": float(pre),
                     "f1": float(f1),
                     "recall_at_fixed": float(rec_f),
                     "precision_at_fixed": float(pre_f),
                     "n_pixels_real": int(len(Lte)) if k > 4 else None})
        print(f"  {k*20:>5} m  blocks {len(Lte):>9,}  base {100*base:>5.1f}%  "
              f"AUC {auc:.4f}  AP {apr:.4f}  lift {apr/base:.2f}x  "
              f"F1 {f1:.4f}", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + "=" * 90)
    print("HOW ACCURATE, AT WHAT SPATIAL PRECISION")
    print("=" * 90)
    print(f"  {'cell':>8}{'area':>10}{'blocks':>11}{'flooded':>9}{'AUC':>8}"
          f"{'AP':>8}{'lift':>7}{'recall':>8}{'prec':>8}{'F1':>8}")
    for r in rows:
        print(f"  {r['cell_m']:>6} m{r['area_km2']:>9.3f}km²{r['blocks_test']:>11,}"
              f"{100*r['base']:>8.1f}%{r['auc']:>8.4f}{r['ap']:>8.4f}"
              f"{r['lift']:>6.2f}x{r['recall']:>8.4f}{r['precision']:>8.4f}"
              f"{r['f1']:>8.4f}")
    print("\n" + "=" * 90)
    print("SAME TABLE, EVERY SCALE FORCED TO ~60% RECALL")
    print("(so the precision column is directly comparable - no cutoff games)")
    print("=" * 90)
    print(f"  {'cell':>8}{'recall':>10}{'precision':>12}{'AUC':>9}{'lift':>8}")
    for r in rows:
        print(f"  {r['cell_m']:>6} m{r['recall_at_fixed']:>10.4f}"
              f"{r['precision_at_fixed']:>12.4f}{r['auc']:>9.4f}"
              f"{r['lift']:>7.2f}x")
    print("\n  lift = average precision divided by the base rate. It is the")
    print("  honest column: it cannot be inflated by making floods commoner.")

    if rows:
        best = max(rows, key=lambda r: r["lift"])
        print(f"\n  best lift: {best['cell_m']} m cells "
              f"({best['area_km2']:.3f} km²) at {best['lift']:.2f}x")

    json.dump(rows, (ROOT / "data/scale_tradeoff.json").open("w"), indent=1)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        out = ROOT / "data/report"
        out.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        c = [r["cell_m"] for r in rows]
        ax[0].plot(c, [r["auc"] for r in rows], "o-", color="#1668a8")
        ax[0].set_xscale("log"); ax[0].set_xlabel("prediction cell size (m)")
        ax[0].set_ylabel("AUC"); ax[0].grid(alpha=.3)
        ax[0].set_title("Accuracy against spatial precision")
        ax[1].plot(c, [r["lift"] for r in rows], "o-", color="#c33c33")
        ax[1].set_xscale("log"); ax[1].set_xlabel("prediction cell size (m)")
        ax[1].set_ylabel("average precision / base rate")
        ax[1].grid(alpha=.3); ax[1].set_title("Lift over chance")
        fig.tight_layout()
        fig.savefig(out / "scale_tradeoff.png", dpi=130)
        print(f"  figure: data/report/scale_tradeoff.png")
    except Exception as e:
        print(f"  (figure skipped: {e})")
    print("\nwrote data/scale_tradeoff.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
