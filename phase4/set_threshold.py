#!/usr/bin/env python3
"""
Set the flood threshold, using the validated flood/quiet controls.

This is what NEXT_STEPS section 4 always intended. The first attempt failed
because its negatives were reported floods with silent gauges - mostly real
floods. These negatives are quiet dates at the same locations, and they
separate from floods at p = 2.8e-24.

The cutoff is chosen to maximise separation between flood days and quiet days,
per region and then averaged, so no single region decides it (section 4c). Model
scores are never consulted.

Reports the result at several cutoffs, not one, so the paper can show the
conclusion does not hinge on the choice (section 3d).

Output: data/final_threshold.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

NEAR_KM = 10.0
LOW_PCT = 25
CUTS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30, 0.40]


def frac_over(gdir: Path, lon: float, lat: float) -> dict | None:
    """Fraction of channel pixels whose coherence drop exceeds each cutoff."""
    try:
        bc_p = sorted((gdir / "baseline").glob("*_corr.tif"))[0]
        cc_p = sorted((gdir / "coevent").glob("*_corr.tif"))[0]
        dem_p = sorted((gdir / "coevent").glob("*_dem.tif"))[0]
        (bc, cc, dem), transform, crs = read_aligned(
            [str(bc_p), str(cc_p), str(dem_p)])
    except Exception:
        return None
    valid = np.isfinite(bc) & np.isfinite(cc)
    drop = np.where(valid, bc - cc, np.nan)
    rr, cix = np.indices(bc.shape)
    xs, ys = rasterio.transform.xy(transform, rr, cix)
    xs = np.array(xs).reshape(bc.shape); ys = np.array(ys).reshape(bc.shape)
    ex, ey = warp_transform("EPSG:4326", crs, [lon], [lat])
    near = valid & (np.hypot(xs - ex[0], ys - ey[0]) / 1000.0 <= NEAR_KM)
    if near.sum() < 5000:
        return None
    dem_ok = np.isfinite(dem) & (dem > -1e4)
    low = near
    if dem_ok.sum() > 1000:
        fill = float(np.nanmean(dem[dem_ok]))
        smooth = ndimage.uniform_filter(np.where(dem_ok, dem, fill), size=51)
        rel = np.where(dem_ok, dem - smooth, np.nan)
        thr = np.nanpercentile(rel[near], LOW_PCT)
        cand = near & (rel <= thr)
        if cand.sum() > 500:
            low = cand
    dl = drop[low]
    return {c: float(np.nanmean(dl > c)) for c in CUTS}


def main() -> int:
    # The pair table already holds one cutoff; recompute across the full sweep.
    negs = [json.loads(l) for l in
            (ROOT / "data/negative_bursts.jsonl").read_text().splitlines() if l.strip()]
    nst = json.loads((ROOT / "data/negatives_state.json").read_text())
    fst = json.loads((ROOT / "data/phase3_state.json").read_text())

    rows = []
    for i, n in enumerate(negs, 1):
        ngid = nst["events"].get(n["NEG_ID"], {}).get("group_id")
        fgid = fst["events"].get(str(n["paired_event"]), {}).get("group_id")
        if not ngid or not fgid:
            continue
        q = frac_over(ROOT / "data/negatives/groups" / ngid, n["lon"], n["lat"])
        f = frac_over(ROOT / "data/phase3/groups" / fgid, n["lon"], n["lat"])
        if not q or not f:
            continue
        for c in CUTS:
            rows.append({"huc4": n["huc4"], "cut": c,
                         "flood": f[c], "quiet": q[c]})
        if i % 25 == 0:
            print(f"  [{i}/{len(negs)}]", flush=True)

    df = pd.DataFrame(rows)
    out = []
    for c in CUTS:
        s = df[df["cut"] == c]
        overall = float((s["flood"] > s["quiet"]).mean())
        per = {}
        for reg, g in s.groupby("huc4"):
            if len(g) >= 5:
                per[reg] = float((g["flood"] > g["quiet"]).mean())
        bal = float(np.mean(list(per.values()))) if per else np.nan
        # Contrast: how many times more channel area a flood disturbs than its
        # own control. Separation says the ordering is right; contrast says the
        # gap is big enough to be worth thresholding on.
        mf, mq = float(s["flood"].median()), float(s["quiet"].median())
        out.append({"cut": c, "win_rate": round(overall, 4),
                    "win_rate_region_balanced": round(bal, 4),
                    "median_flood": round(mf, 5), "median_quiet": round(mq, 5),
                    "contrast_x": round(mf / mq, 2) if mq > 0 else np.inf,
                    **{f"r{k}": round(v, 3) for k, v in sorted(per.items())}})

    res = pd.DataFrame(out)
    print("\n" + "=" * 92)
    print("THRESHOLD SELECTION — flood vs its own quiet control, by cutoff")
    print("=" * 92)
    show = ["cut", "win_rate", "win_rate_region_balanced",
            "median_flood", "median_quiet", "contrast_x"]
    print(res[show].to_string(index=False))

    best = res.loc[res["win_rate_region_balanced"].idxmax()]
    print()
    print(f"CHOSEN: {best.cut}  "
          f"(region-balanced win rate {best.win_rate_region_balanced})")
    print(f"  flood disturbs {best.contrast_x}x more channel area than its control")
    print()
    print("per-region win rate at each cutoff:")
    rc = [c for c in res.columns if c.startswith("r1")]
    print(res[["cut"] + rc].to_string(index=False))

    json.dump({"chosen_threshold": float(best.cut),
               "win_rate_region_balanced": float(best.win_rate_region_balanced),
               "contrast_x": float(best.contrast_x),
               "sweep": out,
               "method": "coherence drop, channel-restricted, 10 km of the event; "
                         "chosen to maximise region-balanced separation of flood "
                         "days from quiet days at the same locations. No model "
                         "score consulted.",
               "report_at": [0.10, 0.15, 0.20]},
              (ROOT / "data/final_threshold.json").open("w"), indent=2)
    print("\nwrote data/final_threshold.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
