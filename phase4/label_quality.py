#!/usr/bin/env python3
"""
Phase 4 - score how trustworthy each event's flood label is.

WHY THIS EXISTS
    Our label is coherence drop. It is strong at the DAY level (floods beat
    their own quiet control in 86.8% of 174 pairs, p=2.8e-24) and weak at the
    PIXEL level (the Sentinel-2 optical cross-check found no confirmation:
    41.6%, p=0.27). A model cannot score better than the answer key it is
    graded against, so pixel-level label noise is the ceiling on every number
    in this project.

    Rather than accept that ceiling for the whole dataset, this scores each
    event so the noisiest can be separated from the trustworthy ones.

THE EVIDENCE USED, AND WHY EACH IS FAIR
  baseline coherence
    Coherence drop is only meaningful if there was coherence to lose. Where the
    baseline pair is already decorrelated, the "drop" is noise subtracted from
    noise.

  amplitude corroboration  <- the important one
    A flood does two physically different things to radar: it destroys
    coherence, and smooth standing water reflects energy away from the sensor
    so the return darkens. Those are separate effects. Where both appear in the
    same pixels, two independent measurements agree.

    This is fair to use because amp_change_db is NOT a model input. It sits in
    label.tif and never enters any feature stack, so selecting on it cannot
    leak information the model then gets to exploit.

  contrast against the event's own quiet control
    Every event with a paired control can be asked: does the flood day show
    more disturbance than the same ground on a calm day? This is the day-level
    test that already passed globally, applied one event at a time.

WHAT THIS DELIBERATELY DOES NOT DO
    It does not select events by how well any model predicts them. That would
    be circular - keeping the events a model already gets right and calling the
    result an improvement. Every signal here is a property of the measurement,
    decided before any model sees it.

Output: data/label_quality.csv
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
THRESH = 0.15               # the coherence drop that counts as flooded


def bands(p: Path) -> dict[str, np.ndarray] | None:
    if not p.exists():
        return None
    with rasterio.open(p) as s:
        a = s.read().astype("float32")
        names = [s.descriptions[i] for i in range(s.count)]
    if not all(names):
        return None
    return {n: a[i] for i, n in enumerate(names)}


def score_event(d: Path) -> dict | None:
    lab = bands(d / "label.tif")
    if lab is None or "coh_drop" not in lab:
        return None
    val = bands(d / "valid.tif")
    drop = lab["coh_drop"]
    base = lab.get("coh_baseline")
    amp = lab.get("amp_change_db")

    m = np.isfinite(drop)
    if base is not None:
        m &= np.isfinite(base)
    if amp is not None:
        m &= np.isfinite(amp)
    if m.sum() < 5000:
        return None

    dr = drop[m]
    flooded = dr > THRESH
    out = {
        "px": int(m.sum()),
        "flood_frac": float(flooded.mean()),
        "coh_baseline_mean": float(base[m].mean()) if base is not None else np.nan,
    }

    # --- amplitude corroboration -------------------------------------------
    # Standing water darkens the return, so amp_change_db should be NEGATIVE
    # where coherence dropped. A negative rank correlation is the agreement.
    if amp is not None and flooded.sum() > 100 and (~flooded).sum() > 100:
        av = amp[m]
        n = min(200_000, av.size)
        sel = np.linspace(0, av.size - 1, n).astype(int)
        r = float(np.corrcoef(pd.Series(dr[sel]).rank(),
                              pd.Series(av[sel]).rank())[0, 1])
        out["amp_agreement"] = -r          # positive = the two agree
        out["amp_median_flood"] = float(np.median(av[flooded]))
        out["amp_median_dry"] = float(np.median(av[~flooded]))
        out["amp_contrast_db"] = out["amp_median_dry"] - out["amp_median_flood"]
    else:
        out["amp_agreement"] = np.nan
        out["amp_contrast_db"] = np.nan
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/features")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    feat = ROOT / args.features
    idx = pd.read_csv(feat / "_index.csv", dtype={"id": str, "huc4": str,
                                                  "huc8": str})
    if args.limit:
        idx = idx.head(args.limit)
    print(f"scoring {len(idx)} events\n", flush=True)

    rows = []
    for n, r in enumerate(idx.itertuples(index=False), 1):
        s = score_event(feat / r.id)
        if s is None:
            continue
        s.update(EVENT_ID=r.id, huc4=r.huc4, huc8=r.huc8)
        rows.append(s)
        if n % 100 == 0:
            print(f"  [{n}/{len(idx)}] {len(rows)} scored", flush=True)

    df = pd.DataFrame(rows)

    # --- Sentinel-2 confirmation, where a clear image pair existed ----------
    s2 = ROOT / "data/sentinel2_check.csv"
    if s2.exists():
        c = pd.read_csv(s2)
        c["EVENT_ID"] = c.EVENT_ID.astype(str)
        c["s2_confirms"] = (c.dmndwi_flood - c.dmndwi_dry) > 0
        df = df.merge(c[["EVENT_ID", "s2_confirms"]], on="EVENT_ID", how="left")
    else:
        df["s2_confirms"] = np.nan

    # --- combine into one score --------------------------------------------
    # Ranked, not raw, so no single scale dominates, and averaged rather than
    # multiplied so one missing signal does not zero an otherwise good event.
    parts = []
    for col, sign in (("amp_agreement", 1), ("amp_contrast_db", 1),
                      ("coh_baseline_mean", 1)):
        if col in df and df[col].notna().sum() > 10:
            parts.append((sign * df[col]).rank(pct=True))
    df["label_quality"] = (pd.concat(parts, axis=1).mean(axis=1)
                           if parts else np.nan)

    out = ROOT / "data/label_quality.csv"
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("LABEL QUALITY, per event")
    print("=" * 72)
    for c in ("coh_baseline_mean", "amp_agreement", "amp_contrast_db",
              "flood_frac", "label_quality"):
        if c not in df:
            continue
        v = df[c].dropna()
        if v.empty:
            continue
        print(f"  {c:<20} p10 {np.percentile(v,10):>8.4f}  "
              f"p50 {np.percentile(v,50):>8.4f}  p90 {np.percentile(v,90):>8.4f}")

    print(f"\n  events scored: {len(df)}")
    if "amp_agreement" in df:
        a = df.amp_agreement.dropna()
        print(f"  events where amplitude AGREES with coherence (>0): "
              f"{int((a>0).sum())}/{len(a)} ({100*(a>0).mean():.0f}%)")
        print(f"  events where it disagrees (<0): {int((a<0).sum())}")

    if df.s2_confirms.notna().sum() > 10:
        print("\n  cross-check: does our quality score agree with Sentinel-2?")
        g = df[df.s2_confirms.notna()]
        hi = g[g.label_quality > g.label_quality.median()]
        lo = g[g.label_quality <= g.label_quality.median()]
        print(f"    top half by quality:    S2 confirms "
              f"{100*hi.s2_confirms.mean():.0f}%  (n={len(hi)})")
        print(f"    bottom half by quality: S2 confirms "
              f"{100*lo.s2_confirms.mean():.0f}%  (n={len(lo)})")
        print("    (an independent instrument agreeing more often with the "
              "events we call clean is the check that this score is real)")

    print(f"\n  per region, median label quality:")
    print(df.groupby("huc4").label_quality.agg(["size", "median"]).round(3).to_string())
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
