#!/usr/bin/env python3
"""
Tier 1 of the "how do we get a better label" plan: test a completely
different radar technique against the same 210 real photos already used
tonight, no model, no fitting, just a direct measurement.

WHAT THIS TESTS
    Everything tonight used coherence drop: compare two radar visits, see if
    the ground statistically "changed." That is an indirect trick and we have
    now proven, three separate ways, that its noise cannot be cleaned up after
    the fact (99.2% of everything it calls "flood" is wrong, uniformly across
    every land cover type).

    This is a different physical principle: water is smooth, so radar energy
    bounces off it away from the sensor, and it looks distinctly DARK.
    `amp_change_db` (already computed, in every event's label.tif, unused
    until now) is exactly 10*log10(after / before) of that brightness. A real
    flood should make this go strongly NEGATIVE (darker after the storm).

WHY THIS IS A FAIR TEST, NOT ANOTHER GUESS
    Same 210 events, same real photos, same "was this actually water" answer
    key already built and verified tonight. No fitting to the data at all,
    coh_drop and amp_change_db are just measured, directly, against the truth,
    so there is no way for this to overfit or hide a shortcut.

Output: data/amplitude_label_test.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    opt_dir = ROOT / "data/optical_labels"
    feat_dir = ROOT / "data/features"
    opt_events = sorted(p.stem for p in opt_dir.glob("*.tif"))
    print(f"{len(opt_events)} events have an optical label", flush=True)

    amp_l, coh_l, y_l, ev_l = [], [], [], []
    used = skipped = 0
    for ev in opt_events:
        lab_p = feat_dir / ev / "label.tif"
        if not lab_p.exists():
            skipped += 1
            continue
        with rasterio.open(lab_p) as s:
            names = list(s.descriptions)
            if "amp_change_db" not in names or "coh_drop" not in names:
                skipped += 1
                continue
            lab = s.read()
        amp = lab[names.index("amp_change_db")]
        coh = lab[names.index("coh_drop")]
        with rasterio.open(opt_dir / f"{ev}.tif") as s:
            flood_o, _, valid_o = s.read()
        mask = (np.isfinite(amp) & np.isfinite(coh) & (valid_o > 0.5)
                & np.isfinite(flood_o))
        if mask.sum() < 200:
            skipped += 1
            continue
        amp_l.append(amp[mask]); coh_l.append(coh[mask])
        y_l.append(flood_o[mask].astype(int))
        ev_l.append(np.full(int(mask.sum()), ev))
        used += 1

    amp = np.concatenate(amp_l); coh = np.concatenate(coh_l)
    y = np.concatenate(y_l).astype(bool); evid = np.concatenate(ev_l)
    print(f"used {used} events (skipped {skipped}), {len(amp):,} pixels, "
          f"{100*y.mean():.3f}% optically wet\n", flush=True)

    auc_amp_raw = roc_auc_score(y, amp)          # expect < 0.5 if darkening = flood
    auc_amp_dark = roc_auc_score(y, -amp)        # the "is it dark" score
    auc_coh = roc_auc_score(y, coh)              # same population, for a fair comparison

    print("=" * 70)
    print("BRIGHTNESS DARKENING vs COHERENCE DROP, GRADED AGAINST REAL PHOTOS")
    print("=" * 70)
    print(f"  coherence drop (this project's label all along)   AUC {auc_coh:.4f}")
    print(f"  amp_change_db, raw                                AUC {auc_amp_raw:.4f}")
    print(f"  amp_change_db, scored as darkness = flood         AUC {auc_amp_dark:.4f}")

    # combined: does knowing BOTH help beyond either alone? Simple, no fitting
    # - just rank by the sum of the two z-scored signals, a fixed, symmetric
    # combination rather than a trained model, so this cannot overfit either.
    def z(a):
        return (a - np.nanmean(a)) / (np.nanstd(a) + 1e-9)
    combo = z(-amp) + z(coh)
    auc_combo = roc_auc_score(y, combo)
    print(f"  simple sum of both signals (no fitting)           AUC {auc_combo:.4f}")

    best_single = max(auc_amp_dark, 1 - auc_amp_raw)
    print(f"\n  best single new signal vs the old label: "
          f"{best_single:.4f} vs {auc_coh:.4f}  "
          f"(delta {best_single - auc_coh:+.4f})")

    json.dump({
        "n_events_used": used, "n_events_skipped": skipped,
        "n_pixels": int(len(amp)), "base_rate": float(y.mean()),
        "auc_coh_drop": float(auc_coh),
        "auc_amp_change_raw": float(auc_amp_raw),
        "auc_amp_change_as_darkness": float(auc_amp_dark),
        "auc_combined_no_fit": float(auc_combo),
    }, open(ROOT / "data/amplitude_label_test.json", "w"), indent=1)
    print("\nwrote data/amplitude_label_test.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
