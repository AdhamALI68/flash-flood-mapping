#!/usr/bin/env python3
"""
Phase 4 - can the label be made less noisy? Tested, not argued.

THE CAUSE OF THE NOISE, MEASURED
    HyP3 gave us 5 range x 1 azimuth looks. Coherence estimated from N=5 looks
    carries noise of roughly (1 - g^2)/sqrt(2N) on each estimate, so the DROP,
    being a difference of two of them, carries sqrt(2) more:

        true coherence 0.90  ->  noise on the drop ~0.085   (0.15 = 1.8 sigma)
        true coherence 0.50  ->  noise on the drop ~0.335   (0.15 = 0.45 sigma)

    A single fixed threshold of 0.15 is therefore a weak test where coherence is
    high and no test at all where coherence is low. That is the whole problem.

THE CANDIDATES
    fixed       drop > 0.15                     what we use now
    z2, z3      drop / sigma(drop) > 2 or 3     statistical detection: the
                                                threshold tightens automatically
                                                exactly where the estimate is
                                                unreliable
    look9       average coherence over 3x3 first, then drop > 0.15
                9x more looks means sigma falls by 3. Costs resolution, not data
    look9_z2    both together
    fusion      drop > 0.15 AND the return darkened (amp_change_db < 0)
                two independent radar effects must agree

HOW THE WINNER IS CHOSEN, WITHOUT CIRCULARITY
    Not by which label a model predicts best - that would just pick whatever is
    easiest to fit. The 174 quiet control dates are the test. They are the same
    ground, same season, same orbit, on days with no flood reported.

    A good label fires on flood days and stays silent on quiet days. So for each
    candidate we measure the firing rate on both, and the winner is the one with
    the largest separation. No model is involved anywhere in this decision.

Output: data/label_variants.csv
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
LOOKS = 5                    # 5 range x 1 azimuth, from the HyP3 product metadata
FIXED = 0.15


def coh_sigma(g: np.ndarray, n_looks: int) -> np.ndarray:
    """Noise on a coherence magnitude estimated from n_looks samples."""
    g = np.clip(g, 0.0, 0.999)
    return (1.0 - g ** 2) / np.sqrt(2.0 * n_looks)


def boxcar(a: np.ndarray, k: int) -> np.ndarray:
    """k x k mean that keeps the grid, ignoring NaN. More looks, same pixels."""
    if k == 1:
        return a
    v = np.nan_to_num(a, nan=0.0)
    w = np.isfinite(a).astype("float32")
    ker = np.ones((k, k), dtype="float32")
    from scipy.signal import fftconvolve
    num = fftconvolve(v, ker, mode="same")
    den = fftconvolve(w, ker, mode="same")
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(den > 0.5, num / den, np.nan)
    return out.astype("float32")


def read_label(d: Path):
    p = d / "label.tif"
    if not p.exists():
        return None
    with rasterio.open(p) as s:
        a = s.read().astype("float32")
        nm = [s.descriptions[i] for i in range(s.count)]
    if not all(nm):
        return None
    b = {n: a[i] for i, n in enumerate(nm)}
    v = d / "valid.tif"
    if v.exists():
        with rasterio.open(v) as s:
            b["_valid"] = s.read(1) > 0.5
    return b


def variants(b: dict) -> dict[str, np.ndarray] | None:
    base, coev = b.get("coh_baseline"), b.get("coh_coevent")
    if base is None or coev is None:
        return None
    drop = b["coh_drop"]
    amp = b.get("amp_change_db")
    m = np.isfinite(drop) & np.isfinite(base) & np.isfinite(coev)
    if "_valid" in b:
        m &= b["_valid"]
    if m.sum() < 5000:
        return None

    sig1 = np.sqrt(coh_sigma(base, LOOKS) ** 2 + coh_sigma(coev, LOOKS) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z1 = drop / sig1

    b3, c3 = boxcar(base, 3), boxcar(coev, 3)
    d3 = b3 - c3
    sig9 = np.sqrt(coh_sigma(b3, LOOKS * 9) ** 2 + coh_sigma(c3, LOOKS * 9) ** 2)
    with np.errstate(invalid="ignore", divide="ignore"):
        z9 = d3 / sig9

    out = {
        "fixed": drop > FIXED,
        "z2": z1 > 2.0,
        "z3": z1 > 3.0,
        "look9": d3 > FIXED,
        "look9_z2": z9 > 2.0,
        "look9_z3": z9 > 3.0,
    }
    if amp is not None:
        out["fusion"] = (drop > FIXED) & (amp < 0) & np.isfinite(amp)
        out["look9_z2_fusion"] = (z9 > 2.0) & (amp < 0) & np.isfinite(amp)
    return {"_mask": m, **{k: np.where(m, v, False) for k, v in out.items()}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=150,
                    help="events and controls to sample (evenly spread)")
    args = ap.parse_args()

    rows = []
    for kind, sub in (("flood", "data/features"), ("quiet", "data/features_neg")):
        idx = pd.read_csv(ROOT / sub / "_index.csv", dtype={"id": str, "huc4": str})
        take = np.linspace(0, len(idx) - 1, min(args.n, len(idx))).astype(int)
        idx = idx.iloc[take]
        print(f"{kind}: {len(idx)} events", flush=True)
        for n, r in enumerate(idx.itertuples(index=False), 1):
            b = read_label(ROOT / sub / r.id)
            if b is None:
                continue
            v = variants(b)
            if v is None:
                continue
            m = v.pop("_mask")
            px = int(m.sum())
            row = {"kind": kind, "id": r.id, "huc4": r.huc4, "px": px}
            for k, arr in v.items():
                row[k] = float(arr.sum() / px)
            rows.append(row)
            if n % 50 == 0:
                print(f"  [{n}/{len(idx)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/label_variants.csv", index=False)
    cand = [c for c in df.columns if c not in ("kind", "id", "huc4", "px")]

    from scipy import stats
    print("\n" + "=" * 84)
    print("WHICH LABEL FIRES ON FLOOD DAYS AND STAYS QUIET ON CALM DAYS?")
    print("=" * 84)
    print(f"  {'label':<20}{'flood days':>12}{'quiet days':>12}{'ratio':>8}"
          f"{'separation':>12}{'p':>10}")
    out = []
    f = df[df.kind == "flood"]
    q = df[df.kind == "quiet"]
    for c in cand:
        a, b = f[c].dropna(), q[c].dropna()
        if len(a) < 20 or len(b) < 20:
            continue
        u = stats.mannwhitneyu(a, b, alternative="greater")
        auc = u.statistic / (len(a) * len(b))
        ratio = a.mean() / b.mean() if b.mean() > 1e-9 else np.inf
        out.append((c, a.mean(), b.mean(), ratio, auc, u.pvalue))
    out.sort(key=lambda r: -r[4])
    for c, fm, qm, ratio, auc, p in out:
        star = "  <-- current" if c == "fixed" else ""
        print(f"  {c:<20}{100*fm:>11.2f}%{100*qm:>11.2f}%{ratio:>8.1f}"
              f"{auc:>12.4f}{p:>10.2g}{star}")

    best = out[0]
    cur = next(r for r in out if r[0] == "fixed")
    print(f"\n  best      : {best[0]} separation {best[4]:.4f}")
    print(f"  current   : fixed separation {cur[4]:.4f}")
    print(f"  difference: {best[4] - cur[4]:+.4f}")
    if best[0] == "fixed":
        print("  VERDICT: the label we already use is the best of these. The "
              "noise is not fixable this way.")
    else:
        print(f"  VERDICT: {best[0]} is a cleaner label. Worth rebuilding on.")
    print("\n  (a label firing on 20% of quiet-day pixels is 20% wrong by "
          "construction, whatever its flood-day rate)")
    json.dump([{"label": c, "flood": fm, "quiet": qm, "ratio": float(ratio),
                "separation": auc, "p": float(p)} for c, fm, qm, ratio, auc, p in out],
              (ROOT / "data/label_variants.json").open("w"), indent=1)
    print(f"\nwrote data/label_variants.csv and .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
