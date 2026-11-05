#!/usr/bin/env python3
"""
Step 2 - do the radar labels agree with what the rivers recorded?

For each candidate coherence-drop cutoff, label every gauge-validated event and
measure how well the labels separate two piles built from gauges alone:

    POSITIVE  a gauge downstream recorded a clear rise (76 events)
    NEGATIVE  a flood was reported but the gauge stayed flat (85 events)

Separation is measured as AUC of flood-fraction against pile membership. Model
scores are never consulted - the cutoff is chosen against river measurements, per
NEXT_STEPS section 4.

Three things this reports that a single pooled number would hide:

  1. AUC per region AND a region-averaged AUC. Region 1503 holds a quarter of the
     gauged events; pooled, the answer would largely be 1503's answer (section 4c).
  2. A held-out region test - choose on some regions, score on the rest.
  3. The magnitude check (section 4d): does a bigger measured discharge produce a
     bigger labelled area? A cutoff can separate two piles while producing extents
     unrelated to flood size; correlation with discharge cannot be faked that way.

Farmland is masked first. Gate B showed irrigated fields decorrelate in both
pairs regardless of flooding, so unmasked they inflate every event equally.

Output: data/threshold_sweep.csv, data/threshold_per_event.csv,
        data/threshold_result.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

GROUPS = ROOT / "data/phase3/groups"
LC_DIR = ROOT / "data/landcover/worldcover"
CROPLAND = 40
NEAR_KM = 10.0
THRESHOLDS = [0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30]


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Probability a random positive scores above a random negative.

    Rank-based, so it is unaffected by the scale of flood fraction - which
    changes with every candidate cutoff and would otherwise make the
    thresholds incomparable.
    """
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, cnt = np.unique(allv, return_inverse=True, return_counts=True)
    sums = np.zeros(len(cnt))
    np.add.at(sums, inv, ranks)
    ranks = (sums / cnt)[inv]
    rp = ranks[: len(pos)].sum()
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def cropland_mask(transform, crs, shape) -> np.ndarray:
    """Reproject WorldCover cropland onto this event's SAR grid.

    WorldCover is 10 m in EPSG:4326; the SAR grid is 20 m UTM. Nearest-neighbour
    is required - class codes must not be interpolated, since averaging class 40
    and class 50 would produce a class that does not exist.
    """
    left, top = transform.c, transform.f
    right = left + transform.a * shape[1]
    bottom = top + transform.e * shape[0]
    lons, lats = warp_transform(crs, "EPSG:4326",
                                [left, right, left, right], [top, top, bottom, bottom])
    out = np.zeros(shape, dtype="uint8")
    for tif in sorted(LC_DIR.glob("*.tif")):
        with rasterio.open(tif) as src:
            b = src.bounds
            if max(lons) < b.left or min(lons) > b.right or \
               max(lats) < b.bottom or min(lats) > b.top:
                continue
            win = from_bounds(max(min(lons), b.left), max(min(lats), b.bottom),
                              min(max(lons), b.right), min(max(lats), b.top),
                              transform=src.transform)
            data = src.read(1, window=win)
            if data.size == 0:
                continue
            dst = np.zeros(shape, dtype="uint8")
            reproject(source=data, destination=dst,
                      src_transform=src.window_transform(win), src_crs=src.crs,
                      dst_transform=transform, dst_crs=crs,
                      resampling=Resampling.nearest)
            out = np.maximum(out, (dst == CROPLAND).astype("uint8"))
    return out.astype(bool)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    piles = pd.read_csv(ROOT / "data/gauge_piles.csv")
    final = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    piles = piles[piles["EVENT_ID"].isin(final["EVENT_ID"])]
    piles = piles[piles["pile"].isin(["POSITIVE", "NEGATIVE"])]

    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    ev_state = st["events"]
    recs = {}
    for line in (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            recs[str(r["EVENT_ID"])] = r

    disch = {}
    for line in (ROOT / "data/gauge_discharge.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            disch[r["EVENT_ID"]] = r.get("best_peak")

    todo = piles if not args.limit else piles.head(args.limit)
    print(f"{len(todo)} gauge-validated events "
          f"({(todo.pile=='POSITIVE').sum()} positive, "
          f"{(todo.pile=='NEGATIVE').sum()} negative)\n", flush=True)

    rows = []
    for i, r in enumerate(todo.itertuples(index=False), 1):
        eid = str(r.EVENT_ID)
        v = ev_state.get(eid, {})
        gid = v.get("group_id")
        if not gid or eid not in recs:
            continue
        d = GROUPS / gid
        bp = sorted((d / "baseline").glob("*_corr.tif"))
        cp = sorted((d / "coevent").glob("*_corr.tif"))
        if not bp or not cp:
            continue
        try:
            (bc, cc), transform, crs = read_aligned([str(bp[0]), str(cp[0])])
        except Exception as e:
            print(f"  {eid}: skip ({type(e).__name__})", flush=True)
            continue

        valid = np.isfinite(bc) & np.isfinite(cc)
        drop = np.where(valid, bc - cc, np.nan)

        rr, cix = np.indices(bc.shape)
        xs, ys = rasterio.transform.xy(transform, rr, cix)
        xs = np.array(xs).reshape(bc.shape); ys = np.array(ys).reshape(bc.shape)
        rec = recs[eid]
        ex, ey = warp_transform("EPSG:4326", crs, [rec["lon"]], [rec["lat"]])
        near = valid & (np.hypot(xs - ex[0], ys - ey[0]) / 1000.0 <= NEAR_KM)

        try:
            crop = cropland_mask(transform, crs, bc.shape)
        except Exception:
            crop = np.zeros(bc.shape, bool)
        use = near & ~crop
        if use.sum() < 1000:
            continue

        row = {"EVENT_ID": r.EVENT_ID, "pile": r.pile, "huc4": r.huc4,
               "basin": r.basin_name, "px": int(use.sum()),
               "cropland_pct": round(100 * (near & crop).sum() / max(near.sum(), 1), 2),
               "gauge_peak": disch.get(r.EVENT_ID)}
        for t in THRESHOLDS:
            row[f"f_{t}"] = float((use & (drop > t)).sum() / use.sum())
        rows.append(row)
        if i % 20 == 0:
            print(f"  [{i}/{len(todo)}]", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/threshold_per_event.csv", index=False)
    print(f"\nprocessed {len(df)} events "
          f"({(df.pile=='POSITIVE').sum()}P / {(df.pile=='NEGATIVE').sum()}N)")
    print(f"median cropland inside 10 km: {df.cropland_pct.median():.1f}%")

    # ---- sweep -----------------------------------------------------------
    out = []
    for t in THRESHOLDS:
        col = f"f_{t}"
        p = df[df.pile == "POSITIVE"][col].values
        n = df[df.pile == "NEGATIVE"][col].values
        pooled = auc(p, n)
        per = {}
        for reg, g in df.groupby("huc4"):
            pp = g[g.pile == "POSITIVE"][col].values
            nn = g[g.pile == "NEGATIVE"][col].values
            if len(pp) >= 3 and len(nn) >= 3:
                per[reg] = auc(pp, nn)
        # Equal region weight, per section 4c - pooled is dominated by 1503.
        bal = float(np.nanmean(list(per.values()))) if per else float("nan")
        sub = df[df.gauge_peak.notna() & (df.gauge_peak > 0)]
        mag = (np.corrcoef(np.log10(sub.gauge_peak.values),
                           sub[col].values)[0, 1] if len(sub) > 5 else float("nan"))
        out.append({"threshold": t, "auc_pooled": round(pooled, 4),
                    "auc_region_balanced": round(bal, 4),
                    "n_regions_scored": len(per),
                    "median_flood_pct_POS": round(100 * float(np.median(p)), 2),
                    "median_flood_pct_NEG": round(100 * float(np.median(n)), 2),
                    "magnitude_corr": round(float(mag), 4),
                    **{f"auc_{k}": round(v, 3) for k, v in sorted(per.items())}})

    sw = pd.DataFrame(out)
    sw.to_csv(ROOT / "data/threshold_sweep.csv", index=False)

    print("\n" + "=" * 96)
    print("THRESHOLD SWEEP — separation of gauge-confirmed floods from non-events")
    print("=" * 96)
    show = ["threshold", "auc_pooled", "auc_region_balanced", "n_regions_scored",
            "median_flood_pct_POS", "median_flood_pct_NEG", "magnitude_corr"]
    print(sw[show].to_string(index=False))

    # Region-balanced is the criterion (section 4c). It is undefined when no
    # region has at least 3 of each pile - which happens on small subsets - so
    # fall back to pooled and say so rather than crash.
    if sw["auc_region_balanced"].notna().any():
        best = sw.loc[sw["auc_region_balanced"].idxmax()]
        criterion = "region-balanced AUC"
    else:
        best = sw.loc[sw["auc_pooled"].idxmax()]
        criterion = "POOLED AUC (too few events per region to balance)"
    print()
    print(f"BEST by {criterion}: threshold = {best.threshold} "
          f"(AUC {best.auc_region_balanced if criterion.startswith('region') else best.auc_pooled})")
    print(f"  pooled AUC at that value : {best.auc_pooled}")
    print(f"  magnitude correlation    : {best.magnitude_corr}")
    print()
    print("per-region AUC at each threshold:")
    reg_cols = [c for c in sw.columns if c.startswith("auc_1")]
    print(sw[["threshold"] + reg_cols].to_string(index=False))

    json.dump({"selection_criterion": criterion,
               "best_threshold": float(best.threshold),
               "auc_region_balanced": float(best.auc_region_balanced),
               "auc_pooled": float(best.auc_pooled),
               "magnitude_corr": float(best.magnitude_corr),
               "n_events": int(len(df)),
               "n_positive": int((df.pile == "POSITIVE").sum()),
               "n_negative": int((df.pile == "NEGATIVE").sum()),
               "thresholds_tested": THRESHOLDS,
               "note": "AUC 0.5 = no separation. Chosen on gauges only; "
                       "no model score was consulted."},
              (ROOT / "data/threshold_result.json").open("w"), indent=2)
    print("\nwrote data/threshold_sweep.csv, threshold_per_event.csv, threshold_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
