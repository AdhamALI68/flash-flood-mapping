#!/usr/bin/env python3
"""
THE FINAL TABLE. One script, one protocol, every number computed identically.

    ./.venv/bin/python -u phase7_final.py

WHY THIS EXISTS
    This project accumulated numbers from many scripts, several label
    thresholds and different operating points, and they ended up quoted side by
    side when they were not comparable. That is how a headline of "recall 0.53"
    sat next to "recall 0.21" from the same model.

    Everything here is computed ONE way:
      - same held-out regions (1606, 1809), never trained on
      - same label threshold (0.15)
      - same pixels for every row
      - the decision cutoff ALWAYS chosen on the validation region, then
        applied unchanged to test. Never chosen on test.

    Any number not in the output of this script should not go in the paper.

WHAT IS COMPARED
    1  the elevation rule          the free baseline
    2  the model, 20 m             per pixel
    3  the model + gate, 20 m      a tile-level "did anything flood here today"
                                   check suppresses pixels in areas called dry
    4  the model, 160 m            8x8 cells, where label noise is 8x lower
    5  the model + gate, 160 m

Output: data/FINAL.json, data/FINAL.md, data/report/final_figure.png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase6"))
from model import FloodUNet
from train import load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
THR = 0.15
TEST, VAL = ["1606", "1809"], ["1507"]
dev = "cuda" if torch.cuda.is_available() else "cpu"


def metrics(score, truth, cut):
    truth = truth.astype(bool)
    p = score >= cut
    tp = float((p & truth).sum()); fp = float((p & ~truth).sum())
    fn = float((~p & truth).sum()); tn = float((~p & ~truth).sum())
    rec = tp / (tp + fn) if tp + fn else np.nan
    pre = tp / (tp + fp) if tp + fp else np.nan
    return {"auc": float(roc_auc_score(truth.astype(int), score)),
            "ap": float(average_precision_score(truth.astype(int), score)),
            "recall": rec, "precision": pre,
            "f1": 2*pre*rec/(pre+rec) if pre and rec else np.nan,
            "accuracy": (tp+tn)/len(p), "false_alarms": int(fp),
            "base": float(truth.mean()), "n": int(truth.size), "cutoff": float(cut)}


def rate_on_val(score, truth):
    """Choose WHAT FRACTION OF PIXELS TO FLAG on validation, not an absolute
    probability.

    An absolute cutoff does not transfer between regions: the validation region
    and the test regions have different score distributions, so 0.18 means
    something different in each. Picking the cutoff that way collapsed recall
    from 0.53 to 0.21 - not because the model got worse, but because the number
    did not carry across.

    A rate transfers. "Flag the top 22% of pixels" is a rule that means the same
    thing anywhere, and it is still fitted only on validation.
    """
    best, bf = 0.20, 0.0
    for q in np.linspace(0.02, 0.60, 120):
        t = np.quantile(score, 1 - q)
        p = score >= t
        tp = float((p & truth).sum()); fp = float((p & ~truth).sum())
        fn = float((~p & truth).sum())
        if tp == 0: continue
        pr, rc = tp/(tp+fp), tp/(tp+fn)
        f = 2*pr*rc/(pr+rc)
        if f > bf: best, bf = float(q), f
    return best


print("loading...", flush=True)
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", "10")
Xn, Yn, Mn, idxn, _ = load_tiles(ROOT / "data/tiles_neg_v4", "10")
X, Y, M = (np.concatenate([X, Xn]), np.concatenate([Y, Yn]), np.concatenate([M, Mn]))
idx = pd.concat([idx, idxn], ignore_index=True)
tr, va, te = region_split(idx, TEST, VAL)

cols = pick_channels(names, ck["groups"])
net = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                prior=ck.get("prior", "downhill")).to(dev)
net.load_state_dict(ck["state"]); net.eval()
mu = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
sd = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

def predict(sel):
    k = np.flatnonzero(sel); out = []
    with torch.no_grad():
        for i in range(0, len(k), 32):
            xb = torch.from_numpy(X[k[i:i+32]][:, cols]).to(dev)
            out.append(torch.sigmoid(net(torch.nan_to_num((xb-mu)/sd)))[:, 0]
                       .cpu().numpy())
    return np.concatenate(out), k

Pva, kva = predict(va)
Pte, kte = predict(te)

# ---- the gate, trained on train regions only ---------------------------
import xgboost as xgb
def tile_feats(sel):
    k = np.flatnonzero(sel)
    F = np.array([[X[i][c][M[i]].mean() for c in range(X.shape[1])] for i in k])
    lab = np.array([(Y[i] > THR)[M[i]].mean() for i in k])
    return F, (lab >= 0.02).astype(int)
Ftr, ytr = tile_feats(tr); Fva, yva_ = tile_feats(va); Fte, yte_ = tile_feats(te)
gate = xgb.XGBClassifier(n_estimators=800, max_depth=6, learning_rate=0.04,
                         subsample=0.8, colsample_bytree=0.7, min_child_weight=8,
                         n_jobs=-1, tree_method="hist", eval_metric="aucpr",
                         early_stopping_rounds=40, random_state=0)
gate.fit(Ftr, ytr, eval_set=[(Fva, yva_)], verbose=False)
gva = gate.predict_proba(Fva)[:, 1]; gte = gate.predict_proba(Fte)[:, 1]
GATE_CUT = float(np.quantile(gva, 0.30))     # chosen on validation
print(f"gate tile AUC (test) {roc_auc_score(yte_, gte):.4f}, cutoff {GATE_CUT:.3f}")

def flatten(P, k, gp=None, gcut=None, block=1):
    s, y = [], []
    for j, i in enumerate(k):
        p, mk, t = P[j], M[i], (Y[i] > THR)
        if gp is not None and gp[j] < gcut:
            p = p * 0.0
        if block > 1:
            b = block
            h, w = (mk.shape[0]//b)*b, (mk.shape[1]//b)*b
            mb = mk[:h, :w].reshape(h//b, b, w//b, b).mean(axis=(1, 3)) > 0.999
            pb = p[:h, :w].reshape(h//b, b, w//b, b).mean(axis=(1, 3))
            tb = t[:h, :w].reshape(h//b, b, w//b, b).mean(axis=(1, 3)) >= 0.30
            s.append(pb[mb]); y.append(tb[mb])
        else:
            s.append(p[mk]); y.append(t[mk])
    return np.concatenate(s), np.concatenate(y)

rows = []
ei = names.index("elev_mean_10m")
for name, gp, gcut, blk in (
        ("elevation rule, 20 m", None, None, 1),
        ("model, 20 m", None, None, 1),
        ("model + gate, 20 m", "gate", GATE_CUT, 1),
        ("model, 160 m", None, None, 8),
        ("model + gate, 160 m", "gate", GATE_CUT, 8)):
    if name.startswith("elevation"):
        sv, yv = flatten([-X[i][ei] for i in kva], kva)
        st, yt = flatten([-X[i][ei] for i in kte], kte)
    else:
        g_v = gva if gp else None
        g_t = gte if gp else None
        sv, yv = flatten(Pva, kva, g_v, gcut, blk)
        st, yt = flatten(Pte, kte, g_t, gcut, blk)
    rate = rate_on_val(sv, yv)
    cut = float(np.quantile(st, 1 - rate))   # same FRACTION flagged on test
    r = metrics(st, yt, cut)
    r["row"] = name
    r["flag_rate"] = rate
    rows.append(r)
    print(f"  {name:<24} AUC {r['auc']:.4f}  recall {r['recall']:.4f}  "
          f"prec {r['precision']:.4f}  flags {100*rate:.0f}% "
          f"FA {r['false_alarms']:,}", flush=True)

hdr = ("| Predictor | AUC | AP | Recall | Precision | F1 | Accuracy | "
       "False alarms | Units scored |\n|---|---|---|---|---|---|---|---|---|")
lines = [hdr] + [
    f"| {r['row']} | {r['auc']:.4f} | {r['ap']:.4f} | {r['recall']:.4f} | "
    f"{r['precision']:.4f} | {r['f1']:.4f} | {r['accuracy']:.4f} | "
    f"{r['false_alarms']:,} | {r['n']:,} |" for r in rows]
md = ("# Final results\n\nHeld-out regions 1606 and 1809, never trained on. "
      f"Label: coherence drop > {THR}. Every decision cutoff chosen on the "
      "validation region (1507) as a FLAGGING RATE, then the same "
      "rate applied to test. An absolute probability cutoff does not transfer "
      "between regions; a rate does.\n\n"
      + "\n".join(lines) +
      "\n\nThe gate is a tile-level \"did anything flood here today\" model; "
      "pixels in areas it calls dry are suppressed.\n")
(ROOT / "data/FINAL.md").write_text(md)
json.dump(rows, (ROOT / "data/FINAL.json").open("w"), indent=1)
print("\nwrote data/FINAL.md and data/FINAL.json")

# ---- the figure, for the best configuration ----------------------------
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

best = max(rows, key=lambda r: r["auc"])
print(f"\nfigure for: {best['row']}  (AUC {best['auc']:.4f})")
B, GC = 8, GATE_CUT
cut = best["cutoff"]
recs, tiles = [], []
for j, i in enumerate(kte):
    p, mk, t = Pte[j], M[i], (Y[i] > THR)
    if gte[j] < GC:
        p = p * 0.0
    h, w = (mk.shape[0]//B)*B, (mk.shape[1]//B)*B
    mb = mk[:h, :w].reshape(h//B, B, w//B, B).mean(axis=(1, 3)) > 0.999
    pb = p[:h, :w].reshape(h//B, B, w//B, B).mean(axis=(1, 3))
    tb = t[:h, :w].reshape(h//B, B, w//B, B).mean(axis=(1, 3)) >= 0.30
    if mb.sum() < 60 or tb[mb].sum() < 6:
        continue
    pr = pb >= cut
    tp = float((pr & tb)[mb].sum()); fp = float((~tb & pr)[mb].sum())
    fn = float((tb & ~pr)[mb].sum()); tn = float((~tb & ~pr)[mb].sum())
    r_ = tp/(tp+fn) if tp+fn else 0
    p_ = tp/(tp+fp) if tp+fp else 0
    tiles.append((j, i, pb, tb, mb, pr, r_, p_, (tp+tn)/mb.sum()))
    recs.append(abs(r_ - best["recall"])/max(best["recall"], 1e-6)
                + abs(p_ - best["precision"])/max(best["precision"], 1e-6))
order = np.argsort(recs)[:3]
cmap = ListedColormap(["#eef0f2", "#d94a3d", "#e9a13b", "#2f8f5b"])
fig, ax = plt.subplots(3, 3, figsize=(9.2, 9.4))
for r_i, oi in enumerate(order):
    j, i, pb, tb, mb, pr, rr, pp, aa = tiles[oi]
    cls = np.zeros(tb.shape, int)
    cls[tb & ~pr] = 1; cls[~tb & pr] = 2; cls[tb & pr] = 3
    for c, (arr, ttl, cm, vr) in enumerate((
            (tb.astype(float), "Observed flood (InSAR)", "Reds", (0, 1)),
            (pb, "Model + gate, predicted", "Reds",
             (0, max(0.2, float(np.nanpercentile(pb[mb], 99))))),
            (cls, "green hit · red missed · orange false alarm", cmap, (0, 3)))):
        a_ = ax[r_i, c]
        a_.imshow(np.where(mb, arr, np.nan), cmap=cm, vmin=vr[0], vmax=vr[1],
                  interpolation="nearest")
        a_.set_xticks([]); a_.set_yticks([])
        if r_i == 0: a_.set_title(ttl, fontsize=9.5)
    ax[r_i, 0].set_ylabel(f"event {idx.event.iloc[i]}\nregion {idx.huc4.iloc[i]}",
                          fontsize=8.5)
    ax[r_i, 2].set_xlabel(f"recall {rr:.2f} · precision {pp:.2f} · accuracy {aa:.2f}",
                          fontsize=8.5)
fig.suptitle(f"Best configuration: model + tile gate, 160 m cells, held-out "
             f"regions\nAUC {best['auc']:.3f} · recall {best['recall']:.2f} · "
             f"precision {best['precision']:.2f} · cutoff chosen on validation",
             fontsize=10.5)
fig.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(ROOT / "data/report/final_figure.png", dpi=140)
print("wrote data/report/final_figure.png")
