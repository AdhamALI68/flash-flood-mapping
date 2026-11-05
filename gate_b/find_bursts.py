#!/usr/bin/env python3
"""
Gate B, step 1 - locate the Sentinel-1 bursts for one test event.

INSAR_ISCE_BURST pairs must be the SAME burst: same full burst ID, same
subswath, same polarisation. So the job is to find one burst that covers the
event location and exists on all three dates:

    pre2  ---- pre1   [FLOOD]   post
     \_ baseline _/
        pre1 ------------------> co-event

Both pairs get processed at 5x1 looks (20 m), the finest coherence HyP3 offers.

Output: gate_b/<event>_bursts.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import asf_search as asf
import pandas as pd
from shapely.geometry import Point, shape


def bursts_on_date(pt_wkt: str, day: pd.Timestamp, pol: str, window_h: int = 12) -> list:
    """Every burst covering a point within +/- window_h of a scene time."""
    start = (day - pd.Timedelta(hours=window_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    end = (day + pd.Timedelta(hours=window_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    res = asf.search(
        dataset=asf.DATASET.SLC_BURST,
        intersectsWith=pt_wkt,
        start=start,
        end=end,
    )
    out = []
    for p in res:
        pr = p.properties
        b = pr.get("burst") or {}
        # Polarisation strings differ between SLC ("VV+VH") and burst ("VV").
        if pol and pr.get("polarization") not in pol:
            continue
        out.append({
            "scene": pr.get("fileID") or pr.get("sceneName"),
            "full_burst_id": b.get("fullBurstID"),
            "subswath": b.get("subswath"),
            "pol": pr.get("polarization"),
            "path": pr.get("pathNumber"),
            "start_time": pr.get("startTime"),
            "geometry": p.geojson()["geometry"],
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event-id", type=int, default=983830)
    ap.add_argument("--availability", default="data/sar_availability.csv")
    ap.add_argument("--events", default="data/events.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    av = pd.read_csv(root / args.availability)
    ev = pd.read_csv(root / args.events, low_memory=False)

    row = av[av["EVENT_ID"] == args.event_id]
    if row.empty:
        raise SystemExit(f"event {args.event_id} not in {args.availability}")
    row = row.iloc[0]
    erow = ev[ev["EVENT_ID"] == args.event_id].iloc[0]

    lat, lon = float(erow["BEGIN_LAT"]), float(erow["BEGIN_LON"])
    pt = Point(lon, lat)
    print(f"event {args.event_id}  {row['basin_name']}  {row['BEGIN_UTC']}")
    print(f"  location : {lat:.4f}, {lon:.4f}")
    print(f"  path     : {row['path']}   pol: {row['pol']}")
    print(f"  scenes   : pre2={row['pre2']}\n             pre1={row['pre1']}\n             post={row['post']}")
    print()

    # Scene names carry their acquisition time in field 5 (…_YYYYMMDDTHHMMSS_…).
    times = {}
    for tag in ("pre2", "pre1", "post"):
        name = row[tag]
        stamp = name.split("_")[5]
        times[tag] = pd.to_datetime(stamp, format="%Y%m%dT%H%M%S")

    found = {}
    for tag, t in times.items():
        b = bursts_on_date(pt.wkt, t, "VV")
        found[tag] = b
        print(f"{tag} ({t}): {len(b)} VV bursts cover the point")

    # A usable burst exists on all three dates.
    by_id = defaultdict(dict)
    for tag, lst in found.items():
        for b in lst:
            fid = b["full_burst_id"]
            if fid:
                by_id[fid][tag] = b

    complete = {k: v for k, v in by_id.items() if len(v) == 3}
    print(f"\nburst IDs present on all three dates: {len(complete)}")
    if not complete:
        print("NO COMMON BURST - cannot build matched pairs for this event")
        return 1

    # Prefer the burst whose footprint centre is closest to the event point,
    # so the flood sits well inside the frame rather than on an edge.
    def dist(item):
        g = shape(item[1]["pre1"]["geometry"])
        return g.centroid.distance(pt)

    fid, triple = sorted(complete.items(), key=dist)[0]
    print(f"selected burst: {fid}  subswath {triple['pre1']['subswath']}")
    for tag in ("pre2", "pre1", "post"):
        print(f"  {tag:<5} {triple[tag]['scene']}")

    out = {
        "event_id": int(args.event_id),
        "basin": str(row["basin_name"]),
        "begin_utc": str(row["BEGIN_UTC"]),
        "lat": lat, "lon": lon,
        "full_burst_id": fid,
        "subswath": triple["pre1"]["subswath"],
        "path": int(row["path"]),
        "scenes": {t: triple[t]["scene"] for t in ("pre2", "pre1", "post")},
        "times": {t: str(times[t]) for t in times},
        "pairs": {
            "baseline": [triple["pre2"]["scene"], triple["pre1"]["scene"]],
            "coevent": [triple["pre1"]["scene"], triple["post"]["scene"]],
        },
    }
    dest = root / "gate_b" / f"{args.event_id}_bursts.json"
    dest.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {dest.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
