#!/usr/bin/env python3
"""
THE TEST. Do flood days look different from ordinary days at the same place?

Every negative was built as the same location as a real flood event, same
season, same relative orbit, no flood reported within 14 days, and put through
an identical pipeline. So each flood/quiet pair differs in one thing only:
whether a flood happened.

That supports a PAIRED comparison, which is much stronger than pooling two
groups. Terrain, land cover, incidence angle and processing are held constant
within a pair, so a difference cannot be explained by any of them.

Reported for every candidate measurement, not just the best one, because a
result that only appears under one hand-picked statistic is not a result:

  paired win rate   how often the flood scores above its own quiet control.
                    0.5 = no difference. This is the headline.
  sign test p       probability of that many wins if floods and quiet days were
                    identical.
  AUC               unpaired separation, for comparison with published work.

Output: data/flood_vs_quiet.csv, data/flood_vs_quiet_result.json
"""

from __future__ import annotations

import argparse
import json
import math
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


def measure(gdir: Path, lon: float, lat: float) -> dict | None:
    """All candidate measurements for one scene triple. Identical for both groups."""
    try:
        bc_p = sorted((gdir / "baseline").glob("*_corr.tif"))[0]
        cc_p = sorted((gdir / "coevent").glob("*_corr.tif"))[0]
        ba_p = sorted((gdir / "baseline").glob("*_amp.tif"))[0]
        ca_p = sorted((gdir / "coevent").glob("*_amp.tif"))[0]
        dem_p = sorted((gdir / "coevent").glob("*_dem.tif"))[0]
        (bc, cc, ba, ca, dem), transform, crs = read_aligned(
            [str(bc_p), str(cc_p), str(ba_p), str(ca_p), str(dem_p)])
    except Exception:
        return None

    valid = np.isfinite(bc) & np.isfinite(cc)
    drop = np.where(valid, bc - cc, np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        amp_db = 10 * np.log10(np.where((ba > 0) & (ca > 0), ca / ba, np.nan))

    rr, cix = np.indices(bc.shape)
    xs, ys = rasterio.transform.xy(transform, rr, cix)
    xs = np.array(xs).reshape(bc.shape); ys = np.array(ys).reshape(bc.shape)
    ex, ey = warp_transform("EPSG:4326", crs, [lon], [lat])
    near = valid & (np.hypot(xs - ex[0], ys - ey[0]) / 1000.0 <= NEAR_KM)
    if near.sum() < 5000:
        return None

    dem_ok = np.isfinite(dem) & (dem > -1e4)
    low = np.zeros_like(near)
    if dem_ok.sum() > 1000:
        fill = float(np.nanmean(dem[dem_ok]))
        smooth = ndimage.uniform_filter(np.where(dem_ok, dem, fill), size=51)
        rel = np.where(dem_ok, dem - smooth, np.nan)
        thr = np.nanpercentile(rel[near], LOW_PCT)
        low = near & (rel <= thr)

    dn, an = drop[near], amp_db[near]
    anf = an[np.isfinite(an)]
    out = {
        "coh_mean": float(np.nanmean(dn)),
        "coh_frac_0.15": float(np.nanmean(dn > 0.15)),
        "coh_p95": float(np.nanpercentile(dn, 95)),
        "amp_p05_db": float(np.percentile(anf, 5)) if anf.size else np.nan,
        "amp_frac_dark": float(np.nanmean(an < -3)),
        "fusion_frac": float((near & (drop > 0.15) & (amp_db < -1)).sum() / near.sum()),
    }
    if low.sum() > 500:
        dl, al = drop[low], amp_db[low]
        alf = al[np.isfinite(al)]
        out["low_coh_frac_0.15"] = float(np.nanmean(dl > 0.15))
        out["low_amp_p05_db"] = float(np.percentile(alf, 5)) if alf.size else np.nan
        out["low_amp_frac_dark"] = float(np.nanmean(al < -3))
        out["low_fusion_frac"] = float((low & (drop > 0.15) & (amp_db < -1)).sum() / low.sum())
    return out


def sign_test_p(wins: int, n: int) -> float:
    """Two-sided exact binomial p for wins out of n at p=0.5."""
    if n == 0:
        return float("nan")
    k = min(wins, n - wins)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def auc(pos, neg) -> float:
    pos = np.asarray(pos, float); neg = np.asarray(neg, float)
    pos = pos[np.isfinite(pos)]; neg = neg[np.isfinite(neg)]
    if len(pos) < 3 or len(neg) < 3:
        return float("nan")
    allv = np.concatenate([pos, neg])
    r = pd.Series(allv).rank().values
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2)
                 / (len(pos) * len(neg)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    negs = [json.loads(l) for l in
            (ROOT / "data/negative_bursts.jsonl").read_text().splitlines() if l.strip()]
    nst = json.loads((ROOT / "data/negatives_state.json").read_text())
    fst = json.loads((ROOT / "data/phase3_state.json").read_text())
    if args.limit:
        negs = negs[: args.limit]
    print(f"{len(negs)} flood/quiet pairs\n", flush=True)

    rows = []
    for i, n in enumerate(negs, 1):
        ngid = nst["events"].get(n["NEG_ID"], {}).get("group_id")
        fgid = fst["events"].get(str(n["paired_event"]), {}).get("group_id")
        if not ngid or not fgid:
            continue
        q = measure(ROOT / "data/negatives/groups" / ngid, n["lon"], n["lat"])
        f = measure(ROOT / "data/phase3/groups" / fgid, n["lon"], n["lat"])
        if not q or not f:
            continue
        row = {"NEG_ID": n["NEG_ID"], "paired_event": n["paired_event"],
               "huc4": n["huc4"], "basin": n["basin_name"]}
        for k in set(q) & set(f):
            row[f"flood_{k}"] = f[k]
            row[f"quiet_{k}"] = q[k]
        rows.append(row)
        if i % 20 == 0:
            print(f"  [{i}/{len(negs)}] {len(rows)} usable pairs", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "data/flood_vs_quiet.csv", index=False)
    print(f"\n{len(df)} complete pairs\n")

    metrics = sorted({c[6:] for c in df.columns if c.startswith("flood_")})
    # For amplitude, flooding makes the value MORE NEGATIVE, so a "win" is the
    # flood scoring lower. Direction is fixed by physics, not by the data.
    lower_is_flood = {"amp_p05_db", "low_amp_p05_db"}

    res = []
    for m in metrics:
        fv = df[f"flood_{m}"].values
        qv = df[f"quiet_{m}"].values
        ok = np.isfinite(fv) & np.isfinite(qv)
        f_, q_ = fv[ok], qv[ok]
        if len(f_) < 10:
            continue
        wins = int((f_ < q_).sum() if m in lower_is_flood else (f_ > q_).sum())
        n = len(f_)
        a = auc(q_, f_) if m in lower_is_flood else auc(f_, q_)
        res.append({"measurement": m, "n_pairs": n,
                    "paired_win_rate": round(wins / n, 4),
                    "sign_test_p": float(f"{sign_test_p(wins, n):.3g}"),
                    "auc": round(a, 4),
                    "median_flood": round(float(np.median(f_)), 5),
                    "median_quiet": round(float(np.median(q_)), 5)})

    rdf = pd.DataFrame(res).sort_values("paired_win_rate",
                                        key=lambda s: (s - 0.5).abs(), ascending=False)
    print("=" * 100)
    print("FLOOD DAYS vs QUIET DAYS, SAME LOCATIONS — paired")
    print("=" * 100)
    print(rdf.to_string(index=False))

    best = rdf.iloc[0]
    print()
    print(f"strongest: {best.measurement}")
    print(f"  flood beats its own quiet control in {100*best.paired_win_rate:.1f}% of pairs")
    print(f"  p = {best.sign_test_p}")
    print(f"  0.5 would mean no difference at all")

    print()
    print("per region, using the strongest measurement:")
    m = best.measurement
    for reg, g in df.groupby("huc4"):
        fv, qv = g[f"flood_{m}"].values, g[f"quiet_{m}"].values
        ok = np.isfinite(fv) & np.isfinite(qv)
        if ok.sum() < 5:
            continue
        w = int((fv[ok] < qv[ok]).sum() if m in lower_is_flood else (fv[ok] > qv[ok]).sum())
        print(f"  {reg}: {w}/{ok.sum()} = {100*w/ok.sum():.0f}%")

    json.dump({"n_pairs": int(len(df)),
               "best_measurement": best.measurement,
               "paired_win_rate": float(best.paired_win_rate),
               "sign_test_p": float(best.sign_test_p),
               "auc": float(best.auc),
               "all": res,
               "note": "Paired: each quiet date is the same location, season and "
                       "orbit as its flood event. 0.5 = no difference."},
              (ROOT / "data/flood_vs_quiet_result.json").open("w"), indent=2)
    print("\nwrote data/flood_vs_quiet.csv, flood_vs_quiet_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
