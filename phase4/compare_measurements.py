#!/usr/bin/env python3
"""
Which measurement actually tracks flood size? Needs no negatives.

The first sweep averaged coherence drop over a 10 km circle - 314 km2 - and got
a correlation of 0.04 with measured discharge. A flood affecting 2 km2 of that
circle is 0.6% of it, and would be invisible in a mean even if perfectly
detected. The measurement may have been the problem, not the labels.

This compares candidate measurements against gauge peak discharge, using only
data already on disk. Discharge is an independent physical quantity, so a
measurement that tracks it is measuring flooding rather than noise.

Variants:
  mean            what the first sweep used
  p95, p99, max   sensitive to a small area changing a lot
  frac_gt_X       fraction of pixels past a cutoff
  amp_*           AMPLITUDE change, never tested until now. Water is dark in
                  radar, and the 2024 RSE work the plan cites says amplitude +
                  coherence fusion is optimal for arid terrain
  low_*           restricted to locally LOW ground using the DEM shipped inside
                  every product - a cheap proxy for the channel network, where
                  floods actually are

Whichever variant correlates best with discharge is the one to use when the
negatives land.

Output: data/measurement_comparison.csv
"""

from __future__ import annotations

import argparse
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

GROUPS = ROOT / "data/phase3/groups"
NEAR_KM = 10.0
LOW_PCT = 25          # "low ground" = lowest quarter of local elevation


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation - discharge spans orders of magnitude and is skewed."""
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 8:
        return float("nan")
    ra = pd.Series(a[ok]).rank().values
    rb = pd.Series(b[ok]).rank().values
    return float(np.corrcoef(ra, rb)[0, 1])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    piles = pd.read_csv(ROOT / "data/gauge_piles.csv")
    pos = piles[piles["pile"] == "POSITIVE"]
    disch = {}
    for line in (ROOT / "data/gauge_discharge.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            if r.get("best_peak"):
                disch[r["EVENT_ID"]] = r["best_peak"]
    pos = pos[pos["EVENT_ID"].isin(disch)]
    if args.limit:
        pos = pos.head(args.limit)
    print(f"{len(pos)} gauge-confirmed floods with a discharge value\n", flush=True)

    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    recs = {}
    for line in (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[str(r["EVENT_ID"])] = r

    rows = []
    for i, r in enumerate(pos.itertuples(index=False), 1):
        eid = str(r.EVENT_ID)
        gid = st["events"].get(eid, {}).get("group_id")
        if not gid or eid not in recs:
            continue
        d = GROUPS / gid
        try:
            bc_p = sorted((d / "baseline").glob("*_corr.tif"))[0]
            cc_p = sorted((d / "coevent").glob("*_corr.tif"))[0]
            ba_p = sorted((d / "baseline").glob("*_amp.tif"))[0]
            ca_p = sorted((d / "coevent").glob("*_amp.tif"))[0]
            dem_p = sorted((d / "coevent").glob("*_dem.tif"))[0]
            (bc, cc, ba, ca, dem), transform, crs = read_aligned(
                [str(bc_p), str(cc_p), str(ba_p), str(ca_p), str(dem_p)])
        except Exception as e:
            print(f"  {eid}: skip ({type(e).__name__})", flush=True)
            continue

        valid = np.isfinite(bc) & np.isfinite(cc)
        drop = np.where(valid, bc - cc, np.nan)
        # Amplitude ratio in dB: negative means it got darker, which is what
        # standing water does.
        with np.errstate(divide="ignore", invalid="ignore"):
            amp_db = 10 * np.log10(np.where((ba > 0) & (ca > 0), ca / ba, np.nan))

        rr, cix = np.indices(bc.shape)
        xs, ys = rasterio.transform.xy(transform, rr, cix)
        xs = np.array(xs).reshape(bc.shape); ys = np.array(ys).reshape(bc.shape)
        rec = recs[eid]
        ex, ey = warp_transform("EPSG:4326", crs, [rec["lon"]], [rec["lat"]])
        near = valid & (np.hypot(xs - ex[0], ys - ey[0]) / 1000.0 <= NEAR_KM)
        if near.sum() < 5000:
            continue

        # Cheap channel proxy: ground well below its local surroundings. Uses
        # the DEM already inside every product, so it needs no lidar processing.
        dem_ok = np.isfinite(dem) & (dem > -1e4)
        low = np.zeros_like(near)
        if dem_ok.sum() > 1000:
            smooth = ndimage.uniform_filter(np.where(dem_ok, dem, np.nan_to_num(np.nanmean(dem[dem_ok]))), size=51)
            rel = np.where(dem_ok, dem - smooth, np.nan)
            thr = np.nanpercentile(rel[near], LOW_PCT)
            low = near & (rel <= thr)

        row = {"EVENT_ID": r.EVENT_ID, "huc4": r.huc4,
               "discharge": disch[r.EVENT_ID], "px": int(near.sum()),
               "low_px": int(low.sum())}
        dn = drop[near]
        row["coh_mean"] = float(np.nanmean(dn))
        row["coh_p95"] = float(np.nanpercentile(dn, 95))
        row["coh_p99"] = float(np.nanpercentile(dn, 99))
        row["coh_frac_0.15"] = float(np.nanmean(dn > 0.15))
        row["coh_frac_0.30"] = float(np.nanmean(dn > 0.30))
        an = amp_db[near]
        row["amp_mean_db"] = float(np.nanmean(an))
        row["amp_p05_db"] = float(np.nanpercentile(an[np.isfinite(an)], 5)) if np.isfinite(an).any() else np.nan
        row["amp_frac_dark"] = float(np.nanmean(an < -3))
        # fusion: darker AND decorrelated
        fus = near & (drop > 0.15) & (amp_db < -1)
        row["fusion_frac"] = float(fus.sum() / near.sum())
        if low.sum() > 500:
            dl = drop[low]
            row["low_coh_mean"] = float(np.nanmean(dl))
            row["low_coh_frac_0.15"] = float(np.nanmean(dl > 0.15))
            row["low_fusion_frac"] = float((low & (drop > 0.15) & (amp_db < -1)).sum() / low.sum())
            # The two strongest ideas combined - amplitude, restricted to the
            # channels where floods actually are. Untested until now.
            al = amp_db[low]
            al = al[np.isfinite(al)]
            if al.size > 100:
                row["low_amp_p05_db"] = float(np.percentile(al, 5))
                row["low_amp_mean_db"] = float(np.mean(al))
                row["low_amp_frac_dark"] = float(np.mean(al < -3))
                row["low_amp_p10_db"] = float(np.percentile(al, 10))
        rows.append(row)
        if i % 10 == 0:
            print(f"  [{i}/{len(pos)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/measurement_comparison.csv", index=False)
    print(f"\nprocessed {len(df)} events")

    logq = np.log10(df["discharge"].values)
    metrics = [c for c in df.columns
               if c not in ("EVENT_ID", "huc4", "discharge", "px", "low_px")]
    out = []
    for m in metrics:
        out.append({"measurement": m,
                    "spearman_vs_discharge": round(spearman(df[m].values, logq), 4),
                    "n": int(np.isfinite(df[m].values).sum())})
    res = pd.DataFrame(out).sort_values("spearman_vs_discharge",
                                        key=lambda s: s.abs(), ascending=False)
    print()
    print("=" * 66)
    print("CORRELATION WITH MEASURED DISCHARGE (rank, |higher| is better)")
    print("=" * 66)
    print(res.to_string(index=False))
    print()
    print(f"the first sweep used coh_mean, which scored "
          f"{res[res.measurement=='coh_mean'].spearman_vs_discharge.iloc[0]:+.4f}")
    best = res.iloc[0]
    print(f"best variant: {best.measurement} at {best.spearman_vs_discharge:+.4f}")
    print()
    print("wrote data/measurement_comparison.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
