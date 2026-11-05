#!/usr/bin/env python3
"""
NEXT_STEPS Track B, step 2 - find gauges HYDROLOGICALLY downstream of each event.

find_gauges.py used nearest-distance, which is the wrong test. A gauge 5 km away
in an adjacent drainage records nothing about this flood; a gauge 30 km
downstream on the same wash records it exactly. Distance mixes those cases and
would feed false negatives into the threshold sweep - events labelled "gauge saw
nothing" that no gauge could ever have seen.

Uses the USGS NLDI, which navigates the real NHD flowline network: snap the
event point to a flowline, walk downstream, return gauges encountered.

Expect this to REDUCE the usable count from the 63%-within-25km figure. That is
the point - what remains is defensible.

Resumable: appends JSONL, skips events already done.

Output: data/downstream_gauges.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
NLDI = "https://api.water.usgs.gov/nldi/linked-data"
RETRIES = 4


def _get(session: requests.Session, url: str, params: dict | None = None):
    delay = 3
    for attempt in range(1, RETRIES + 1):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code in (404, 400):
                return None          # no flowline here - a real answer, not an error
            r.raise_for_status()
            return r.json()
        except (requests.HTTPError, requests.Timeout, requests.ConnectionError):
            if attempt == RETRIES:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def comid_for_point(session, lon: float, lat: float) -> str | None:
    """Snap an event point onto the NHD flowline network."""
    d = _get(session, f"{NLDI}/comid/position", {"coords": f"POINT({lon} {lat})"})
    if not d or not d.get("features"):
        return None
    props = d["features"][0].get("properties", {})
    return str(props.get("identifier") or props.get("comid") or "").split("/")[-1] or None


def downstream_sites(session, comid: str, km: float) -> list[dict]:
    """Gauges downstream along the main flowpath, within km river distance."""
    d = _get(session, f"{NLDI}/comid/{comid}/navigation/DM/nwissite",
             {"distance": int(km)})
    if not d:
        return []
    out = []
    for f in d.get("features", []):
        p = f.get("properties", {})
        ident = str(p.get("identifier", ""))
        out.append({
            "site_no": ident.split("-")[-1] if "-" in ident else ident,
            "name": p.get("name"),
            "river_km": p.get("distance"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="data/dataset_phase3.csv")
    ap.add_argument("--max-km", type=float, default=100,
                    help="river distance to search downstream")
    ap.add_argument("--out", default="data/downstream_gauges.jsonl")
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / args.dataset, low_memory=False)
    out = ROOT / args.out
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["EVENT_ID"])
        print(f"resuming: {len(done)} events already done")

    todo = ev[~ev["EVENT_ID"].isin(done)]
    print(f"{len(todo)} of {len(ev)} events to resolve\n", flush=True)

    session = requests.Session()
    n_ok = n_nocomid = n_nogauge = 0
    with out.open("a") as fh:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            comid = comid_for_point(session, float(r.BEGIN_LON), float(r.BEGIN_LAT))
            if not comid:
                rec = {"EVENT_ID": int(r.EVENT_ID), "huc8": str(r.huc8).zfill(8),
                       "comid": None, "sites": [], "reason": "no flowline at point"}
                n_nocomid += 1
            else:
                sites = downstream_sites(session, comid, args.max_km)
                rec = {"EVENT_ID": int(r.EVENT_ID), "huc8": str(r.huc8).zfill(8),
                       "comid": comid, "sites": sites,
                       "reason": "" if sites else "no downstream gauge"}
                if sites:
                    n_ok += 1
                else:
                    n_nogauge += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] with downstream gauge {n_ok}, "
                      f"no gauge {n_nogauge}, unsnappable {n_nocomid}", flush=True)
            time.sleep(0.2)

    print(f"\nwith downstream gauge : {n_ok}")
    print(f"no downstream gauge   : {n_nogauge}")
    print(f"point not on network  : {n_nocomid}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
