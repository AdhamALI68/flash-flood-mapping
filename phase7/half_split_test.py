#!/usr/bin/env python3
"""
Labels or inputs? Train on one half of a clean flood's 10 km square and test on
the other half, with a 1 km gap between the halves. The label is clean there,
so a low score cannot be blamed on label noise.

Decision lines written before the run:
  held-out AUC >= 0.85  the inputs can describe where a clean flood went;
                        the bottleneck is labels and transfer
  held-out AUC <= 0.70  the inputs cannot describe even a clean flood;
                        the bottleneck is the inputs
  in between            both

Model: gradient-boosted trees on the same 30 per-pixel inputs as the U-Net
(a per-pixel tree scored 0.680 against the U-Net's 0.695 in Section 11b, so it
is a fair probe of how much the inputs hold). Four directions per flood.

Controls on the same pixels:
  radar      : the old coherence-drop label (coh_drop > 0.15)
  shuffled   : photo labels shuffled in the training half; must score about 0.5
  elevation  : the free rule, lower ground more likely flooded

    ./.venv/bin/python -u phase7/half_split_test.py
    ./.venv/bin/python -u phase7/half_split_test.py --rungs 1,2,5,10,30 --out data/half_split_test_allrungs.json

Output: data/half_split_test.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import numpy as np
import rasterio
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "phase6"))
from train import load_tiles  # noqa: E402

FLOODS = {"983830": "data/tiles_trace_painted_rock", "1142175": "data/tiles_trace_checked_v1",
          "1293229": "data/tiles_trace_checked_v1", "779149": "data/tiles_trace_checked_v1"}
MID, GAP = 250, 25
MAX_TRAIN = 150_000
rng = np.random.default_rng(0)


def pixels(eid, tiles_dir, cache, rungs):
    if tiles_dir not in cache:
        cache[tiles_dir] = load_tiles(ROOT / tiles_dir, rungs)
    X, Y, M, idx, names = cache[tiles_dir]
    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
        coh = s.read(4)
    feats, photo, radar, rows, cols = [], [], [], [], []
    for t in np.flatnonzero(idx.event.astype(str).values == eid):
        r0, c0 = int(idx.row.iloc[t]), int(idx.col.iloc[t])
        ii, jj = np.nonzero(M[t])
        feats.append(X[t][:, ii, jj].T)
        photo.append(Y[t][ii, jj] > 0)
        radar.append(coh[r0 + ii, c0 + jj])
        rows.append(r0 + ii); cols.append(c0 + jj)
    return (np.concatenate(feats).astype("float32"), np.concatenate(photo), np.concatenate(radar),
            np.concatenate(rows), np.concatenate(cols), names)


def fit_score(Xtr, ytr, Xte, yte):
    if ytr.sum() < 50 or yte.sum() < 50 or (~yte).sum() < 50:
        return None
    if len(ytr) > MAX_TRAIN:
        k = rng.choice(len(ytr), MAX_TRAIN, replace=False)
        Xtr, ytr = Xtr[k], ytr[k]
    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                         class_weight="balanced", early_stopping=True,
                                         validation_fraction=0.1, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    pc, rc, _ = precision_recall_curve(yte, p)
    return {"auc": round(float(roc_auc_score(yte, p)), 3), "ap": round(float(average_precision_score(yte, p)), 3),
            "base_rate": round(float(yte.mean()), 3), "precision_at_recall_0.5": round(float(pc[rc >= 0.5].max()), 3),
            "test_flood_pixels": int(yte.sum())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rungs", default="10", help="terrain resolutions to stack, e.g. 1,2,5,10,30")
    ap.add_argument("--out", default="data/half_split_test.json")
    args = ap.parse_args()
    cache, out = {}, {}
    print(f"terrain rungs {args.rungs}\n", flush=True)
    for eid, tdir in FLOODS.items():
        F, photo, coh, rows, cols, names = pixels(eid, tdir, cache, args.rungs)
        elev = F[:, names.index("elev_mean_10m")]
        radar_ok = np.isfinite(coh)
        halves = {"west": cols < MID - GAP, "east": cols >= MID + GAP,
                  "north": rows < MID - GAP, "south": rows >= MID + GAP}
        res = {}
        for tr, te in (("west", "east"), ("east", "west"), ("north", "south"), ("south", "north")):
            a, b = halves[tr], halves[te]
            shuffled = photo[a].copy(); rng.shuffle(shuffled)
            r = {"photo": fit_score(F[a], photo[a], F[b], photo[b]),
                 "radar": fit_score(F[a & radar_ok], coh[a & radar_ok] > 0.15, F[b & radar_ok], coh[b & radar_ok] > 0.15),
                 "shuffled": fit_score(F[a], shuffled, F[b], photo[b])}
            r["elevation"] = (round(float(roc_auc_score(photo[b], -elev[b])), 3)
                              if 50 <= photo[b].sum() < b.sum() - 50 else None)
            res[f"train {tr}, test {te}"] = r
            print(eid, f"train {tr:5s} test {te:5s}",
                  {k: (v["auc"] if isinstance(v, dict) else v) for k, v in r.items()}, flush=True)
        photo_aucs = [v["photo"]["auc"] for v in res.values() if v["photo"]]
        out[eid] = {"splits": res, "photo_auc_mean": round(float(np.mean(photo_aucs)), 3) if photo_aucs else None,
                    "flood_share": round(float(photo.mean()), 3), "pixels": int(len(photo))}
        print(f"  {eid}: mean held-out AUC with the clean label {out[eid]['photo_auc_mean']}\n", flush=True)
    json.dump({"rungs": args.rungs, "floods": out}, open(ROOT / args.out, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
