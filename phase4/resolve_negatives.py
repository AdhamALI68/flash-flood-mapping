#!/usr/bin/env python3
"""
Find Sentinel-1 bursts for the quiet (negative) dates.

Same structure as the flood events, deliberately: two scenes before the quiet
date and one after, on one relative orbit, so the negatives are processed by an
identical pipeline. If negatives were built any other way, a difference between
the groups could come from the processing rather than from flooding.

    pre2 ---- pre1   [quiet date]   post
     \_ baseline _/
        pre1 -----------------------> "co-event" spanning nothing

For a flood event the second pair spans a flood. For a negative it spans an
ordinary interval. Everything else - track, polarisation, looks, burst, masking -
is held identical.

Resumable JSONL.

Output: data/negative_bursts.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import asf_search as asf
import pandas as pd
from shapely.geometry import Point, shape

ROOT = Path(__file__).resolve().parent.parent
MAX_PRE_GAP_DAYS = 26
MAX_POST_GAP_DAYS = 26
MAX_INTERVAL_MISMATCH = 3.0     # same rule the flood set was filtered on


def find_triple(pt: Point, t0: pd.Timestamp) -> dict | None:
    """Two pre + one post around a date, on one track, intervals matched."""
    res = asf.geo_search(
        intersectsWith=pt.wkt,
        platform=[asf.PLATFORM.SENTINEL1],
        processingLevel=[asf.PRODUCT_TYPE.SLC],
        beamMode=[asf.BEAMMODE.IW],
        start=(t0 - pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
        end=(t0 + pd.Timedelta(days=40)).strftime("%Y-%m-%d"),
    )
    scenes = []
    for p in res:
        pr = p.properties
        g = shape(p.geojson()["geometry"])
        if not g.contains(pt):
            continue
        scenes.append({"scene": pr["sceneName"],
                       "dt": pd.to_datetime(pr["startTime"]).tz_localize(None),
                       "path": pr["pathNumber"], "pol": pr["polarization"]})
    if not scenes:
        return None
    df = pd.DataFrame(scenes)

    best = None
    for (path, pol), grp in df.groupby(["path", "pol"]):
        grp = grp.sort_values("dt")
        pre = grp[grp["dt"] < t0]
        post = grp[grp["dt"] > t0]
        if len(pre) < 2 or len(post) < 1:
            continue
        pre1, pre2, post1 = pre.iloc[-1], pre.iloc[-2], post.iloc[0]
        pre_gap = (t0 - pre1["dt"]).total_seconds() / 86400
        post_gap = (post1["dt"] - t0).total_seconds() / 86400
        base_gap = (pre1["dt"] - pre2["dt"]).total_seconds() / 86400
        if pre_gap > MAX_PRE_GAP_DAYS or post_gap > MAX_POST_GAP_DAYS:
            continue
        span = pre_gap + post_gap
        if abs(base_gap - span) > MAX_INTERVAL_MISMATCH:
            continue
        cand = {"path": int(path), "pol": pol,
                "pre2": pre2["scene"], "pre1": pre1["scene"], "post": post1["scene"],
                "pre_gap_days": round(pre_gap, 2), "post_gap_days": round(post_gap, 2),
                "baseline_gap_days": round(base_gap, 2), "span": round(span, 2)}
        if best is None or cand["span"] < best["span"]:
            best = cand
    return best


def resolve_burst(pt: Point, triple: dict) -> dict | None:
    """One burst covering the point on all three dates."""
    times = {tag: pd.to_datetime(triple[tag].split("_")[5], format="%Y%m%dT%H%M%S")
             for tag in ("pre2", "pre1", "post")}
    res = asf.search(
        dataset=asf.DATASET.SLC_BURST,
        intersectsWith=pt.wkt,
        start=(min(times.values()) - pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        end=(max(times.values()) + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        relativeOrbit=triple["path"],
    )
    by_id: dict[str, dict] = defaultdict(dict)
    for p in res:
        pr = p.properties
        b = pr.get("burst") or {}
        fid = b.get("fullBurstID")
        if not fid or pr.get("polarization") != "VV":
            continue
        t = pd.to_datetime(pr["startTime"]).tz_localize(None)
        for tag, want in times.items():
            if abs((t - want).total_seconds()) <= 120:
                by_id[fid][tag] = {"scene": pr.get("fileID") or pr.get("sceneName"),
                                   "subswath": b.get("subswath"),
                                   "geometry": p.geojson()["geometry"]}
    complete = {k: v for k, v in by_id.items() if len(v) == 3}
    if not complete:
        return None
    fid, tri = min(complete.items(),
                   key=lambda kv: shape(kv[1]["pre1"]["geometry"]).centroid.distance(pt))
    return {"full_burst_id": fid, "subswath": tri["pre1"]["subswath"],
            "pairs": {"baseline": [tri["pre2"]["scene"], tri["pre1"]["scene"]],
                      "coevent": [tri["pre1"]["scene"], tri["post"]["scene"]]}}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/negative_bursts.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    neg = pd.read_csv(ROOT / "data/negative_candidates.csv")
    out = ROOT / args.out
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["NEG_ID"])
        print(f"resuming: {len(done)} already resolved")

    todo = neg[~neg["NEG_ID"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"{len(todo)} negatives to resolve\n", flush=True)

    ok = no_scene = no_burst = 0
    with out.open("a") as fh:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            pt = Point(float(r.BEGIN_LON), float(r.BEGIN_LAT))
            t0 = pd.to_datetime(r.quiet_date)
            try:
                tri = find_triple(pt, t0)
            except Exception as e:
                print(f"  {r.NEG_ID} search failed: {type(e).__name__}", flush=True)
                continue
            if not tri:
                no_scene += 1
                continue
            try:
                b = resolve_burst(pt, tri)
            except Exception as e:
                print(f"  {r.NEG_ID} burst failed: {type(e).__name__}", flush=True)
                continue
            if not b:
                no_burst += 1
                continue
            rec = {"NEG_ID": r.NEG_ID, "paired_event": int(r.paired_event),
                   "huc8": str(r.huc8).zfill(8), "huc4": str(r.huc4),
                   "basin_name": r.basin_name,
                   "lat": float(r.BEGIN_LAT), "lon": float(r.BEGIN_LON),
                   "quiet_date": str(r.quiet_date), "path": tri["path"],
                   "pre_gap_days": tri["pre_gap_days"],
                   "post_gap_days": tri["post_gap_days"],
                   "baseline_gap_days": tri["baseline_gap_days"], **b}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ok += 1
            if i % 20 == 0:
                print(f"  [{i}/{len(todo)}] ok {ok}, no scenes {no_scene}, "
                      f"no common burst {no_burst}", flush=True)

    print(f"\nresolved {ok}, no usable scenes {no_scene}, no common burst {no_burst}")
    recs = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    if recs:
        groups = {f"{r['full_burst_id']}|{r['pairs']['baseline'][0]}|"
                  f"{r['pairs']['baseline'][1]}|{r['pairs']['coevent'][1]}" for r in recs}
        print(f"total negatives: {len(recs)} -> {len(groups)} distinct scene triples "
              f"-> {2*len(groups)} HyP3 jobs, ~{2*len(groups)*0.2:.0f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
