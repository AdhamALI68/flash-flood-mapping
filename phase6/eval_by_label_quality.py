#!/usr/bin/env python3
"""
Does the model score better where the label is more trustworthy?

THE QUESTION
    Every number in this project is capped by the answer key. If the cap is
    real, then on events whose label two independent radar measurements agree
    about, the SAME model should score higher - with no retraining, no tuning,
    nothing changed but which tiles we grade on.

WHY THIS IS NOT CIRCULAR
    label_quality is built from baseline coherence, amplitude corroboration and
    amplitude contrast. None of those is a model input, and none of them is
    "how well the model did". The split is decided before the model is asked.

    It is not free of confound, and the honest caveat is stated in the output:
    events where amplitude and coherence agree are probably also the larger,
    more obvious floods. Cleaner label and stronger event are entangled here.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import FloodUNet
from train import auc_within, load_tiles, pick_channels, region_split

ROOT = Path(__file__).resolve().parent.parent
TAG = sys.argv[1] if len(sys.argv) > 1 else "terrain+basin+rain+rain_routed+soil+landcover"

dev = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location=dev, weights_only=False)
X, Y, M, idx, names = load_tiles(ROOT / "data/tiles_v4", ck["rung"])
_, _, te = region_split(idx, ["1606", "1809"], ["1507"])
cols = pick_channels(names, ck["groups"])

model = FloodUNet(in_ch=len(cols), elev_index=ck["elev_index"], base=24,
                  prior=ck.get("prior", "downhill")).to(dev)
model.load_state_dict(ck["state"]); model.eval()
mean = torch.from_numpy(ck["mean"]).view(1, -1, 1, 1).to(dev)
std = torch.from_numpy(ck["std"]).view(1, -1, 1, 1).to(dev)

Xte, Yte, Mte, ixte = X[te], Y[te], M[te], idx[te].reset_index(drop=True)
P = []
with torch.no_grad():
    for i in range(0, len(Xte), 32):
        xb = torch.from_numpy(Xte[i:i + 32][:, cols]).to(dev)
        P.append(torch.sigmoid(model(torch.nan_to_num((xb - mean) / std)))[:, 0]
                 .cpu().numpy())
P = np.concatenate(P)

ei = names.index(f"elev_mean_{ck['rung']}m")
lab = (Yte > ck["threshold"]).astype("float32")
am = np.array([auc_within(P[i], lab[i], Mte[i]) for i in range(len(P))])
ae = np.array([auc_within(-Xte[i, ei], lab[i], Mte[i]) for i in range(len(P))])

q = pd.read_csv(ROOT / "data/label_quality.csv", dtype={"EVENT_ID": str})
ixte["event"] = ixte.event.astype(str)
j = ixte.merge(q[["EVENT_ID", "label_quality", "amp_agreement",
                  "coh_baseline_mean"]],
               left_on="event", right_on="EVENT_ID", how="left")
j["auc_model"], j["auc_elev"] = am, ae
j = j[np.isfinite(j.auc_model) & np.isfinite(j.auc_elev) & j.label_quality.notna()]

print(f"model: {TAG}")
print(f"test tiles with a quality score: {len(j)} "
      f"from {j.event.nunique()} events\n")
print("=" * 74)
print("MODEL SCORE BY LABEL QUALITY  (same model, only the grading set changes)")
print("=" * 74)
print(f"  {'label quality':<22}{'tiles':>7}{'model':>9}{'elevation':>11}{'gain':>9}")
j["bucket"] = pd.qcut(j.label_quality, 4,
                      labels=["worst 25%", "2nd", "3rd", "best 25%"])
for b, g in j.groupby("bucket", observed=True):
    print(f"  {str(b):<22}{len(g):>7}{g.auc_model.mean():>9.4f}"
          f"{g.auc_elev.mean():>11.4f}{(g.auc_model-g.auc_elev).mean():>+9.4f}")
print(f"  {'ALL':<22}{len(j):>7}{j.auc_model.mean():>9.4f}"
      f"{j.auc_elev.mean():>11.4f}{(j.auc_model-j.auc_elev).mean():>+9.4f}")

print("\n" + "-" * 74)
print("KEEPING ONLY THE CLEANEST EVENTS")
print("-" * 74)
print(f"  {'keep top':<12}{'tiles':>8}{'events':>8}{'model':>9}{'elevation':>11}{'gain':>9}")
for frac in (1.0, 0.75, 0.5, 0.25, 0.10):
    cut = j.label_quality.quantile(1 - frac)
    g = j[j.label_quality >= cut]
    if len(g) < 30:
        continue
    print(f"  {100*frac:>9.0f}%{len(g):>8}{g.event.nunique():>8}"
          f"{g.auc_model.mean():>9.4f}{g.auc_elev.mean():>11.4f}"
          f"{(g.auc_model-g.auc_elev).mean():>+9.4f}")

print("\n  also, by amplitude agreement alone (the cleanest single signal):")
for name, g in (("amplitude AGREES (>0)", j[j.amp_agreement > 0]),
                ("amplitude disagrees (<=0)", j[j.amp_agreement <= 0])):
    if len(g) > 20:
        print(f"    {name:<28}{len(g):>6} tiles   model {g.auc_model.mean():.4f}"
              f"   elevation {g.auc_elev.mean():.4f}")

out = ROOT / f"data/eval_label_quality_{TAG}.json"
out.write_text(json.dumps(
    {"tag": TAG, "n_tiles": int(len(j)),
     "by_bucket": {str(b): {"tiles": int(len(g)),
                            "model": float(g.auc_model.mean()),
                            "elev": float(g.auc_elev.mean())}
                   for b, g in j.groupby("bucket", observed=True)}}, indent=1))
print(f"\nwrote {out.relative_to(ROOT)}")
