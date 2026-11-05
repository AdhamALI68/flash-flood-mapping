#!/usr/bin/env python3
"""
Phase 6 - can we hit recall >= 0.70 AND accuracy >= 0.75, per 20 m pixel?

    ./.venv/bin/python -u phase6/target_search.py

THE ARITHMETIC, STATED FIRST
    At a 5.24% base rate, recall 0.70 with accuracy 0.75 needs precision ~0.135.
    We measure ~0.09. So the targets are unreachable without changing the scope
    of the data - no model tweak closes that gap, because it is fixed by how
    rare floods are, not by how good the model is.

    Restricting to events with substantial flooding raises the base rate, which
    moves both targets into reach. That is a legitimate STUDY SCOPE if it is
    declared and the unfiltered number is reported next to it. It is not
    legitimate if it is quietly applied and the reader is left thinking the
    numbers describe all floods.

    So every row prints the base rate it was achieved at, and the unfiltered
    row is always shown.

THE SEARCH
    event flood filter x label threshold x model, per 20 m pixel throughout,
    because the point is a map someone can look at.

    The filter is applied to train AND test - it defines which events the study
    covers, not which ones the model is graded on after the fact. The regional
    split is untouched: 1606 and 1809 are never trained on.

    The cutoff is chosen on VALIDATION as the point that reaches the recall
    target, then applied unchanged to test.

Output: data/target_search.csv
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import load_tiles, region_split

ROOT = Path(__file__).resolve().parent.parent
TARGET_RECALL, TARGET_ACC = 0.70, 0.75


def flat(X, Y, M, sel, thr, cap, rng, names):
    k = np.flatnonzero(sel)
    if not len(k):
        return None, None
    xs = np.concatenate([X[i][:, M[i]].T for i in k])
    ys = np.concatenate([(Y[i] > thr)[M[i]] for i in k]).astype(int)
    good = np.isfinite(xs).all(axis=1)
    xs, ys = xs[good], ys[good]
    if len(xs) > cap:
        j = rng.choice(len(xs), cap, replace=False)
        xs, ys = xs[j], ys[j]
    return xs, ys


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, default=2_000_000)
    args = ap.parse_args()
    rng = np.random.default_rng(0)

    print("loading...", flush=True)
    X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
    Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
    X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]),
               np.concatenate([M, Mn]))
    idx = pd.concat([idx, idxn], ignore_index=True)
    idx["event"] = idx.event.astype(str)
    tr0, va0, te0 = region_split(idx, ["1606", "1809"], ["1507"])

    import xgboost as xgb
    rows = []
    for thr in (0.10, 0.15, 0.20, 0.25):
        # flood fraction of each EVENT at this label threshold
        ff = pd.Series({e: np.nanmean([ (Y[i] > thr)[M[i]].mean()
                                        for i in g.index if M[i].sum() > 200 ])
                        for e, g in idx.groupby("event")})
        for keep in (0.0, 0.05, 0.10, 0.15, 0.20):
            good_ev = set(ff[ff >= keep].index)
            sel = idx.event.isin(good_ev).values
            tr, va, te = tr0 & sel, va0 & sel, te0 & sel
            if te.sum() < 40 or tr.sum() < 100:
                continue
            Xtr, ytr = flat(X, Y, M, tr, thr, args.cap, rng, names)
            Xva, yva = flat(X, Y, M, va, thr, args.cap // 3, rng, names)
            Xte, yte = flat(X, Y, M, te, thr, args.cap, rng, names)
            if Xtr is None or Xte is None or yte.sum() < 500:
                continue
            clf = xgb.XGBClassifier(n_estimators=700, max_depth=8,
                                    learning_rate=0.05, subsample=0.8,
                                    colsample_bytree=0.7, min_child_weight=10,
                                    reg_lambda=1.0, n_jobs=-1,
                                    tree_method="hist", eval_metric="aucpr",
                                    early_stopping_rounds=40, random_state=0)
            clf.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            pv, pt = clf.predict_proba(Xva)[:, 1], clf.predict_proba(Xte)[:, 1]
            auc = roc_auc_score(yte, pt)
            # cutoff = lowest that still reaches the recall target ON VALIDATION
            qs = np.quantile(pv, np.linspace(0.01, 0.99, 200))
            cut = None
            for t in qs[::-1]:
                r = ((pv >= t) & (yva == 1)).sum() / max(yva.sum(), 1)
                if r >= TARGET_RECALL:
                    cut = float(t)
                    break
            if cut is None:
                cut = float(qs[0])
            p = pt >= cut
            tp = float((p & (yte == 1)).sum()); fp = float((p & (yte == 0)).sum())
            fn = float((~p & (yte == 1)).sum()); tn = float((~p & (yte == 0)).sum())
            rec = tp / (tp + fn) if tp + fn else 0
            pre = tp / (tp + fp) if tp + fp else 0
            acc = (tp + tn) / len(p)
            rows.append({"threshold": thr, "keep_events_ge": keep,
                         "events": len(good_ev), "test_px": int(len(yte)),
                         "base": float(yte.mean()), "auc": float(auc),
                         "recall": rec, "precision": pre, "accuracy": acc,
                         "f1": 2*pre*rec/(pre+rec) if pre+rec else 0,
                         "hits_target": bool(rec >= TARGET_RECALL
                                             and acc >= TARGET_ACC)})
            print(f"  thr {thr:.2f}  keep>={100*keep:>4.0f}%  "
                  f"base {100*yte.mean():>5.1f}%  AUC {auc:.4f}  "
                  f"recall {rec:.4f}  acc {acc:.4f}  prec {pre:.4f}"
                  + ("   *** TARGET MET ***" if rec >= TARGET_RECALL
                     and acc >= TARGET_ACC else ""), flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/target_search.csv", index=False)
    print("\n" + "=" * 92)
    print(f"CONFIGURATIONS REACHING recall >= {TARGET_RECALL} "
          f"AND accuracy >= {TARGET_ACC}")
    print("=" * 92)
    hit = df[df.hits_target]
    if len(hit):
        print(hit.sort_values("auc", ascending=False).to_string(index=False))
    else:
        print("  none. Closest by accuracy at recall >= 0.70:")
        near = df[df.recall >= TARGET_RECALL].nlargest(5, "accuracy")
        print(near.to_string(index=False) if len(near) else "  none reached the recall target")
    print("\nwrote data/target_search.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
