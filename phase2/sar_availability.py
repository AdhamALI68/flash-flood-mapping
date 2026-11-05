#!/usr/bin/env python3
"""
Phase 2 - SAR availability. THE KILL GATE.

For every event, decide whether Sentinel-1 can actually support coherence change
detection. That needs three scenes on the SAME relative orbit, not two:

    t-2, t-1   both before the flood  -> baseline coherence
    t-1, t+1   spanning the flood     -> co-event coherence

Coherence in the desert decays on its own between passes, so a co-event drop
means nothing without the baseline to compare it against. Searching for a bare
pre/post pair would overstate how much usable data exists.

Strategy: one ASF query per basin (not per event), then test each event's own
point against the returned scene footprints locally. Basins here run up to
~150 km across and the IW swath is ~250 km, so a basin can straddle a track
boundary - the per-event footprint test is what keeps this honest.

Output: data/sar_availability.csv, data/sar_basin_summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import asf_search as asf
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape

# Sentinel-1 revisit is 6 days with two satellites, 12 with one. Allow two
# cycles of slack so a single missed pass does not disqualify an event.
MAX_PRE_GAP_DAYS = 26
MAX_POST_GAP_DAYS = 26


def basin_scene_inventory(hull_wkt: str, start: str, end: str) -> gpd.GeoDataFrame:
    """All Sentinel-1 IW SLC scenes intersecting a basin, with footprints.

    SLC is required: coherence cannot be computed from GRD, which has already
    discarded the phase.
    """
    res = asf.geo_search(
        intersectsWith=hull_wkt,
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        beamMode=[asf.BEAMMODE.IW],
        start=start,
        end=end,
    )
    rows = []
    for p in res:
        pr = p.properties
        rows.append({
            "scene": pr.get("sceneName"),
            "dt": pd.to_datetime(pr.get("startTime")).tz_localize(None),
            "path": pr.get("pathNumber"),
            "flight": pr.get("flightDirection"),
            "pol": pr.get("polarization"),
            "geometry": shape(p.geojson()["geometry"]),
        })
    if not rows:
        return gpd.GeoDataFrame(
            columns=["scene", "dt", "path", "flight", "pol", "geometry"],
            geometry="geometry", crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def check_event(inv: gpd.GeoDataFrame, pt, t0, t1) -> dict:
    """Does this event have a usable baseline + co-event triple on one track?"""
    covering = inv[inv.contains(pt)]
    if covering.empty:
        return {"usable": False, "reason": "no scene covers the point",
                "n_scenes": 0, "path": None, "pre2": None, "pre1": None, "post": None}

    best = None
    for (path, pol), grp in covering.groupby(["path", "pol"]):
        grp = grp.sort_values("dt")
        pre = grp[grp["dt"] < t0]
        post = grp[grp["dt"] > t1]
        if len(pre) < 2 or len(post) < 1:
            continue
        pre1 = pre.iloc[-1]   # last scene before the flood
        pre2 = pre.iloc[-2]   # the one before that, for the baseline pair
        post1 = post.iloc[0]  # first scene after

        pre_gap = (t0 - pre1["dt"]).total_seconds() / 86400
        post_gap = (post1["dt"] - t1).total_seconds() / 86400
        base_gap = (pre1["dt"] - pre2["dt"]).total_seconds() / 86400

        if pre_gap > MAX_PRE_GAP_DAYS or post_gap > MAX_POST_GAP_DAYS:
            continue
        # The baseline pair should span a comparable interval to the co-event
        # pair, or the two coherence values are not measuring the same thing.
        if base_gap > MAX_PRE_GAP_DAYS:
            continue

        cand = {
            "usable": True, "reason": "", "n_scenes": len(covering),
            "path": int(path), "pol": pol,
            "pre2": pre2["scene"], "pre1": pre1["scene"], "post": post1["scene"],
            "pre_gap_days": round(pre_gap, 2),
            "post_gap_days": round(post_gap, 2),
            "baseline_gap_days": round(base_gap, 2),
            "total_span_days": round(post_gap + pre_gap, 2),
        }
        # Prefer the tightest bracket around the flood.
        if best is None or cand["total_span_days"] < best["total_span_days"]:
            best = cand

    if best:
        return best
    return {"usable": False, "reason": "no track with 2 pre + 1 post in window",
            "n_scenes": len(covering), "path": None, "pre2": None, "pre1": None, "post": None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events.csv")
    ap.add_argument("--min-events", type=int, default=20,
                    help="only check basins with at least this many events")
    ap.add_argument("--limit-basins", type=int, default=0,
                    help="pilot mode: stop after N basins (0 = all)")
    ap.add_argument("--out", default="data/sar_availability.csv")
    ap.add_argument("--summary", default="data/sar_basin_summary.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    ev = pd.read_csv(root / args.events, low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    ev["END_UTC"] = pd.to_datetime(ev["END_UTC"]).fillna(ev["BEGIN_UTC"])
    ev["huc8"] = ev["huc8"].astype(str).str.zfill(8)

    counts = ev["huc8"].value_counts()
    basins = counts[counts >= args.min_events].index.tolist()
    if args.limit_basins:
        basins = basins[: args.limit_basins]
    print(f"checking {len(basins)} basins (>= {args.min_events} events each), "
          f"{ev[ev.huc8.isin(basins)].shape[0]} events\n", flush=True)

    cache_dir = root / "data" / "raw" / "asf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for i, huc in enumerate(basins, 1):
        sub = ev[ev["huc8"] == huc].copy()
        pts = gpd.points_from_xy(sub["BEGIN_LON"], sub["BEGIN_LAT"])
        hull = gpd.GeoSeries(pts, crs="EPSG:4326").union_all().convex_hull.buffer(0.15)

        start = (sub["BEGIN_UTC"].min() - timedelta(days=40)).strftime("%Y-%m-%d")
        end = (sub["END_UTC"].max() + timedelta(days=40)).strftime("%Y-%m-%d")

        name = sub["basin_name"].iloc[0]
        try:
            inv = basin_scene_inventory(hull.wkt, start, end)
        except Exception as e:
            print(f"[{i}/{len(basins)}] {huc} {name}: ASF query FAILED - {e}", file=sys.stderr)
            continue

        ok = 0
        for pt, (_, row) in zip(pts, sub.iterrows()):
            r = check_event(inv, pt, row["BEGIN_UTC"], row["END_UTC"])
            r.update({"EVENT_ID": row["EVENT_ID"], "huc8": huc, "huc4": str(row["huc4"]),
                      "basin_name": name, "BEGIN_UTC": row["BEGIN_UTC"]})
            results.append(r)
            ok += bool(r["usable"])

        print(f"[{i}/{len(basins)}] {huc} {name:<34} "
              f"{len(inv):>4} scenes | {ok:>3}/{len(sub):<3} events usable "
              f"({100*ok/len(sub):.0f}%)", flush=True)

    if not results:
        print("no results", file=sys.stderr)
        return 1

    df = pd.DataFrame(results)
    df.to_csv(root / args.out, index=False)

    summ = (df.groupby(["huc8", "basin_name", "huc4"])
              .agg(events=("usable", "size"), usable=("usable", "sum"))
              .reset_index())
    summ["rate"] = (summ["usable"] / summ["events"]).round(3)
    summ = summ.sort_values("usable", ascending=False)
    summ.to_csv(root / args.summary, index=False)

    print("\n" + "=" * 70)
    print(f"events checked : {len(df)}")
    print(f"usable         : {int(df['usable'].sum())} ({100*df['usable'].mean():.1f}%)")
    print()
    print("KILL GATE: need >= 4 basins with >= 5 usable events each")
    passing = summ[summ["usable"] >= 5]
    print(f"  basins with >= 5 usable events : {len(passing)}")
    print(f"  distinct HUC4 regions among them: {passing['huc4'].nunique()}")
    verdict = ("PASS" if len(passing) >= 4 else
               "REPLAN" if len(passing) >= 2 else "FAIL")
    print(f"  VERDICT: {verdict}")
    print()
    print(summ.head(25).to_string(index=False))

    if not df["usable"].all():
        print("\nwhy events failed:")
        print(df[~df["usable"]]["reason"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
