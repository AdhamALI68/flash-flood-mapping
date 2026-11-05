#!/usr/bin/env python3
"""
Phase 3, step 1 - resolve Sentinel-1 bursts for all 623 events.

One ASF query per event covering the whole pre2..post window, rather than one
per scene, which cuts 1,869 queries to 623. For each event we need a single
burst present on all three dates, on the relative orbit that Phase 2 already
picked.

Resumable: results append to a JSONL file and finished events are skipped, so a
dropped connection costs one event rather than the whole run.

Output: data/phase3_bursts.jsonl
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


def scene_time(name: str) -> pd.Timestamp:
    """Acquisition time out of a Sentinel-1 scene name (field 5)."""
    return pd.to_datetime(name.split("_")[5], format="%Y%m%dT%H%M%S")


def resolve(row, pt: Point) -> dict | None:
    """Find one burst covering the point on all three of this event's dates."""
    times = {tag: scene_time(row[tag]) for tag in ("pre2", "pre1", "post")}
    start = (min(times.values()) - pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (max(times.values()) + pd.Timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")

    res = asf.search(
        dataset=asf.DATASET.SLC_BURST,
        intersectsWith=pt.wkt,
        start=start,
        end=end,
        relativeOrbit=int(row["path"]),
    )

    # Bucket bursts by full ID, then by which of the three dates they fall on.
    by_id: dict[str, dict[str, dict]] = defaultdict(dict)
    for p in res:
        pr = p.properties
        b = pr.get("burst") or {}
        fid = b.get("fullBurstID")
        if not fid or pr.get("polarization") != "VV":
            continue
        t = pd.to_datetime(pr["startTime"]).tz_localize(None)
        for tag, want in times.items():
            # A burst sits inside its parent scene, so it starts within a
            # minute or so of the scene start.
            if abs((t - want).total_seconds()) <= 120:
                by_id[fid][tag] = {
                    "scene": pr.get("fileID") or pr.get("sceneName"),
                    "subswath": b.get("subswath"),
                    "geometry": p.geojson()["geometry"],
                }

    complete = {k: v for k, v in by_id.items() if len(v) == 3}
    if not complete:
        return None

    # Prefer the burst whose footprint centre is nearest the event, so the
    # flood sits inside the frame rather than on an edge.
    fid, triple = min(
        complete.items(),
        key=lambda kv: shape(kv[1]["pre1"]["geometry"]).centroid.distance(pt),
    )
    return {
        "EVENT_ID": int(row["EVENT_ID"]),
        "huc8": str(row["huc8"]).zfill(8),
        "huc4": str(row["huc4"]),
        "basin_name": row["basin_name"],
        "begin_utc": str(row["BEGIN_UTC"]),
        "lat": float(row["BEGIN_LAT"]), "lon": float(row["BEGIN_LON"]),
        "end_lat": None if pd.isna(row.get("END_LAT")) else float(row["END_LAT"]),
        "end_lon": None if pd.isna(row.get("END_LON")) else float(row["END_LON"]),
        "path": int(row["path"]),
        "full_burst_id": fid,
        "subswath": triple["pre1"]["subswath"],
        "pairs": {
            "baseline": [triple["pre2"]["scene"], triple["pre1"]["scene"]],
            "coevent": [triple["pre1"]["scene"], triple["post"]["scene"]],
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset_phase3.csv")
    ap.add_argument("--out", default="data/phase3_bursts.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    df = pd.read_csv(ROOT / args.dataset, low_memory=False)
    out = ROOT / args.out
    done: set[int] = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["EVENT_ID"])
        print(f"resuming: {len(done)} events already resolved")

    todo = df[~df["EVENT_ID"].isin(done)]
    if args.limit:
        todo = todo.head(args.limit)
    print(f"resolving {len(todo)} of {len(df)} events\n", flush=True)

    ok = failed = 0
    with out.open("a") as fh:
        for i, row in enumerate(todo.itertuples(index=False), 1):
            r = row._asdict()
            pt = Point(r["BEGIN_LON"], r["BEGIN_LAT"])
            try:
                rec = resolve(r, pt)
            except Exception as e:
                print(f"[{i}/{len(todo)}] {r['EVENT_ID']} query failed: {type(e).__name__}", flush=True)
                failed += 1
                continue
            if rec is None:
                print(f"[{i}/{len(todo)}] {r['EVENT_ID']} no common burst", flush=True)
                failed += 1
                continue
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ok += 1
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] resolved {ok}, failed {failed}", flush=True)

    print(f"\nresolved {ok}, failed {failed}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
