#!/usr/bin/env python3
"""
Phase 4 - is isolated coherence-drop speckle noise, and does removing it help?

THE PROPOSAL
    A flood pixel with no other flood pixel within a few cells is probably
    estimation noise: coherence noise is near-independent per pixel, while real
    flooding is spatially connected. Drop those isolated positives - in train
    AND test alike - and the label gets cleaner without rebuilding anything.

WHY IT CANNOT BE JUDGED BY THE MODEL SCORE
    Our model produces smooth, connected predictions, so isolated truth pixels
    are precisely the ones it always missed. Deleting them raises recall
    mechanically whether or not they were noise. Judging the idea by the score
    it produces would be circular.

THE NON-CIRCULAR TEST
    Sentinel-2 optical flood maps, from a different satellite with different
    physics, exist for 64 events. If isolated pixels are noise, removing them
    must make the coherence label agree BETTER with the optical evidence. If
    agreement does not improve, we deleted real data that was merely hard to
    predict. No model is involved in that decision.

Output: data/speckle_test.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_SIZES = [1, 2, 4, 8, 16, 32]     # smallest connected flood patch to keep


def clean(mask, min_size):
    """Drop connected flood patches smaller than min_size pixels."""
    if min_size <= 1:
        return mask
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = np.zeros(sizes.size, bool)
    keep[min_size <= sizes] = True
    keep[0] = False
    return keep[lab]


opt_dir = ROOT / "data/optical_labels"
events = sorted(p.stem for p in opt_dir.glob("*.tif"))
print(f"{len(events)} events with an independent optical label\n", flush=True)

frac_removed = {m: [] for m in MIN_SIZES}
agree = {m: [] for m in MIN_SIZES}
n_used = 0
for e in events:
    lab_p = ROOT / "data/features" / e / "label.tif"
    if not lab_p.exists():
        continue
    with rasterio.open(lab_p) as s:
        nm = [s.descriptions[i] for i in range(s.count)]
        drop = s.read(nm.index("coh_drop") + 1)
    with rasterio.open(opt_dir / f"{e}.tif") as s:
        onm = [s.descriptions[i] for i in range(s.count)]
        of = s.read(onm.index("flood_optical") + 1) > 0.5
        ov = s.read(onm.index("valid") + 1) > 0.5
    m = np.isfinite(drop) & ov
    if m.sum() < 20000 or of[m].sum() < 20:
        continue
    base = (drop > THR) & m
    if base.sum() < 100:
        continue
    n_used += 1
    for ms in MIN_SIZES:
        c = clean(base, ms)
        frac_removed[ms].append(1 - c.sum() / max(base.sum(), 1))
        # does the CLEANED label agree better with optical truth?
        try:
            agree[ms].append(roc_auc_score(of[m].astype(int), c[m].astype(int)))
        except Exception:
            pass

print("=" * 78)
print("DOES REMOVING ISOLATED FLOOD PIXELS MAKE THE LABEL MORE TRUE?")
print("=" * 78)
print(f"  {'keep patches >=':<18}{'flood px removed':>19}{'agreement with optical':>26}")
res = {}
for ms in MIN_SIZES:
    if not agree[ms]:
        continue
    fr = float(np.mean(frac_removed[ms]))
    ag = float(np.mean(agree[ms]))
    res[ms] = {"removed": fr, "optical_agreement": ag, "events": len(agree[ms])}
    tag = "   <- current (no cleaning)" if ms == 1 else ""
    print(f"  {ms:<18}{100*fr:>18.1f}%{ag:>26.4f}{tag}")

if res:
    b = max(res.items(), key=lambda kv: kv[1]["optical_agreement"])
    base = res[1]["optical_agreement"]
    print(f"\n  events used: {n_used}")
    print(f"  no cleaning        : agreement {base:.4f}")
    print(f"  best ({b[0]} px min)     : agreement {b[1]['optical_agreement']:.4f}"
          f"  ({100*b[1]['removed']:.0f}% of flood pixels removed)")
    d = b[1]["optical_agreement"] - base
    print(f"  change             : {d:+.4f}")
    print()
    if d > 0.01:
        print("  VERDICT: cleaning makes the label agree BETTER with an")
        print("           independent instrument. The removed pixels were noise.")
        print("           Worth applying to train and test.")
    elif d > 0.0:
        print("  VERDICT: a marginal improvement, within noise. Not evidence.")
    else:
        print("  VERDICT: cleaning does NOT improve agreement with optical")
        print("           truth. The isolated pixels are not distinguishable")
        print("           from real flood, so removing them would delete data")
        print("           and inflate recall without making the label truer.")
json.dump(res, (ROOT / "data/speckle_test.json").open("w"), indent=1)
print("\nwrote data/speckle_test.json")
