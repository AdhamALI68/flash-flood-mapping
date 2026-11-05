#!/usr/bin/env python3
"""
Test the EXACT method from the "20x20 grid" document: salt removal (8-conn,
area<15) AND pepper/hole filling (4-conn, area<20), checked against real
Sentinel-2 optical evidence, not model score.

This extends phase4/speckle_test.py, which only tested salt removal at sizes
1/2/4/8/16/32 and found agreement with optical truth FALLS as more is removed.
This adds the hole-filling step the document also specifies, using its exact
numbers (15, 20), to give that document's method a fair, complete test.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, rasterio
from scipy import ndimage
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
THR = 0.15
MIN_RED = 15     # document's salt threshold, 8-connectivity
MIN_HOLE = 20    # document's pepper threshold, 4-connectivity


def clean_salt_and_pepper(mask):
    # Step 1: salt removal, 8-connectivity, exactly as specified
    lab, n = ndimage.label(mask, structure=np.ones((3, 3)))
    if n > 0:
        sizes = np.bincount(lab.ravel())
        keep = sizes >= MIN_RED
        keep[0] = False
        mask = keep[lab]

    # Step 2: pepper removal, 4-connectivity, fill small enclosed white holes
    dry = ~mask
    lab2, n2 = ndimage.label(dry, structure=np.array([[0,1,0],[1,1,1],[0,1,0]]))
    if n2 > 0:
        # discard (never fill) components touching the outer border - those
        # are the real background, not an enclosed hole
        border_labels = set(lab2[0, :]) | set(lab2[-1, :]) | \
                        set(lab2[:, 0]) | set(lab2[:, -1])
        sizes = np.bincount(lab2.ravel())
        fill = np.zeros(sizes.size, bool)
        for lb in range(1, sizes.size):
            if lb not in border_labels and sizes[lb] < MIN_HOLE:
                fill[lb] = True
        mask = mask | fill[lab2]
    return mask


opt_dir = ROOT / "data/optical_labels"
events = sorted(p.stem for p in opt_dir.glob("*.tif"))
print(f"{len(events)} events with an independent optical label\n", flush=True)

frac_removed, frac_added, agree_before, agree_after = [], [], [], []
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
    base = np.where(m, drop > THR, False)
    if base.sum() < 100:
        continue
    n_used += 1
    cleaned = clean_salt_and_pepper(base)

    frac_removed.append(float(((base & ~cleaned) & m).sum()) / max(base[m].sum(), 1))
    frac_added.append(float(((cleaned & ~base) & m).sum()) / max(base[m].sum(), 1))
    try:
        agree_before.append(roc_auc_score(of[m].astype(int), base[m].astype(int)))
        agree_after.append(roc_auc_score(of[m].astype(int), cleaned[m].astype(int)))
    except Exception:
        pass

print("=" * 78)
print("EXACT DOCUMENT METHOD: salt<15px (8-conn) removed, holes<20px (4-conn) filled")
print("=" * 78)
print(f"  events used                          {n_used}")
print(f"  flood pixels removed (salt cleanup)  {100*np.mean(frac_removed):.2f}%")
print(f"  flood pixels added (hole fill)       {100*np.mean(frac_added):.2f}%")
print(f"  agreement with optical, BEFORE       {np.mean(agree_before):.4f}")
print(f"  agreement with optical, AFTER        {np.mean(agree_after):.4f}")
d = np.mean(agree_after) - np.mean(agree_before)
print(f"  change                                {d:+.4f}")
print()
if d > 0.01:
    print("  VERDICT: the document's exact method DOES improve agreement with")
    print("           independent ground truth. Worth adopting.")
elif d > 0.0:
    print("  VERDICT: marginal improvement, within noise. Not strong evidence.")
else:
    print("  VERDICT: does NOT improve agreement with independent ground truth,")
    print("           consistent with the earlier salt-only test. The document's")
    print("           claimed precision/recall gains are not supported here.")
json.dump({"n_events": n_used, "frac_removed": float(np.mean(frac_removed)),
           "frac_added": float(np.mean(frac_added)),
           "agreement_before": float(np.mean(agree_before)),
           "agreement_after": float(np.mean(agree_after))},
          (ROOT / "data/speckle_test2.json").open("w"), indent=1)
print("\nwrote data/speckle_test2.json")
