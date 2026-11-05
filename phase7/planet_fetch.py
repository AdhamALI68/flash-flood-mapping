#!/usr/bin/env python3
"""
PlanetScope before/after photos for flood-trace labels.

  search : find daily PlanetScope scenes that cover a flood's 10 km square,
           the last clear one before the flood and the first clear one after.
           Searching is free and uses no download quota.
  order  : order only those scenes, clipped to the square (the quota counts
           the clipped area, about 100 km2 per image), and download them.

The API key is read from the PL_API_KEY environment variable and never
printed or written to disk. Ordering needs a download plan on the account;
the Evaluation Program account without a plan can search but not order.

    ./.venv/bin/python -u phase7/planet_fetch.py search --events 983830
    ./.venv/bin/python -u phase7/planet_fetch.py order  --events 983830 --budget-km2 300

Output: data/planet/<event>/search.json, data/planet/<event>/order.json,
        data/planet/<event>/files/*
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import pandas as pd
import rasterio
import requests
from rasterio.warp import transform_bounds
from shapely.geometry import box, shape, mapping

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/planet"
SEARCH = "https://api.planet.com/data/v1/quick-search"
ORDERS = "https://api.planet.com/compute/ops/orders/v2"
PRE_DAYS, POST_DAYS = 30, 15
MIN_CLEAR, MIN_COVER = 80, 0.98


def session():
    key = os.environ.get("PL_API_KEY", "")
    if not key:
        raise SystemExit("PL_API_KEY is not set")
    s = requests.Session()
    s.auth = (key, "")
    return s


def aoi_of(eid: str):
    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as src:
        w, s_, e, n = transform_bounds(src.crs, "EPSG:4326", *src.bounds)
    return box(w, s_, e, n)


def search(s, aoi, start, end):
    body = {"item_types": ["PSScene"],
            "filter": {"type": "AndFilter", "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": mapping(aoi)},
                {"type": "DateRangeFilter", "field_name": "acquired",
                 "config": {"gte": f"{start}T00:00:00Z", "lte": f"{end}T23:59:59Z"}},
                {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": 0.3}}]}}
    r = s.post(SEARCH, json=body, timeout=120)
    if r.status_code != 200:
        raise SystemExit(f"search failed: HTTP {r.status_code} {r.text[:300]}")
    feats, page = r.json().get("features", []), r.json()
    while page.get("_links", {}).get("_next"):
        page = s.get(page["_links"]["_next"], timeout=120).json()
        feats += page.get("features", [])
    out = []
    for f in feats:
        p = f["properties"]
        cover = shape(f["geometry"]).intersection(aoi).area / aoi.area
        out.append({"id": f["id"], "acquired": p.get("acquired"), "cloud_cover": p.get("cloud_cover"),
                    "clear_percent": p.get("clear_percent"), "cover": round(cover, 3),
                    "instrument": p.get("instrument"), "quality": p.get("quality_category"),
                    "publishing_stage": p.get("publishing_stage")})
    return sorted(out, key=lambda x: x["acquired"] or "")


def pick(scenes, flood: pd.Timestamp):
    good = [x for x in scenes if (x["clear_percent"] or 0) >= MIN_CLEAR and x["cover"] >= MIN_COVER
            and x["quality"] in (None, "standard")]
    before = [x for x in good if pd.Timestamp(x["acquired"]).tz_localize(None) < flood]
    after = [x for x in good if pd.Timestamp(x["acquired"]).tz_localize(None) > flood]
    return (before[-1] if before else None), (after[0] if after else None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["search", "order"])
    ap.add_argument("--events", required=True)
    ap.add_argument("--budget-km2", type=float, default=300.0,
                    help="stop before ordering more than this clipped area in total")
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["EVENT_ID"] = ev.EVENT_ID.astype(str)
    s = session()
    spent = 0.0
    for eid in args.events.split(","):
        r = ev[ev.EVENT_ID == eid].iloc[0]
        flood = pd.Timestamp(r.BEGIN_UTC)
        aoi = aoi_of(eid)
        km2 = aoi.area * (111.0 * 111.0 * abs(__import__("math").cos(__import__("math").radians(aoi.centroid.y))))
        scenes = search(s, aoi, (flood - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d"),
                        (flood + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"))
        before, after = pick(scenes, flood)
        d = OUT / eid
        d.mkdir(parents=True, exist_ok=True)
        (d / "search.json").write_text(json.dumps({"flood": str(flood), "aoi": mapping(aoi), "scenes": scenes,
                                                   "before": before, "after": after}, indent=1))
        def days(x):
            return None if x is None else (pd.Timestamp(x["acquired"]).tz_localize(None) - flood).days
        print(f"{eid}: {len(scenes)} scenes in the window, before {days(before)} days, after {days(after)} days")
        if args.action == "search" or before is None or after is None:
            continue
        cost = 2 * km2
        if spent + cost > args.budget_km2:
            print(f"  budget reached ({spent:.0f} + {cost:.0f} km2 > {args.budget_km2:.0f}), stopping")
            break
        body = {"name": f"flood_{eid}",
                "products": [{"item_ids": [before["id"], after["id"]], "item_type": "PSScene",
                              "product_bundle": "analytic_sr_udm2"}],
                "tools": [{"clip": {"aoi": mapping(aoi)}}]}
        o = s.post(ORDERS, json=body, timeout=120)
        if o.status_code not in (200, 202):
            print(f"  order refused: HTTP {o.status_code} {o.text[:300]}")
            continue
        order = o.json()
        spent += cost
        (d / "order.json").write_text(json.dumps(order, indent=1))
        url = order["_links"]["_self"]
        while True:
            state = s.get(url, timeout=120).json()
            if state["state"] in ("success", "partial", "failed", "cancelled"):
                break
            print(f"  order {state['state']}...", flush=True)
            time.sleep(60)
        (d / "order.json").write_text(json.dumps(state, indent=1))
        if state["state"] not in ("success", "partial"):
            print(f"  order ended {state['state']}")
            continue
        (d / "files").mkdir(exist_ok=True)
        for res in state["_links"].get("results", []):
            name = Path(res["name"]).name
            with s.get(res["location"], stream=True, timeout=600) as g:
                g.raise_for_status()
                with open(d / "files" / name, "wb") as fh:
                    for chunk in g.iter_content(1 << 20):
                        fh.write(chunk)
        print(f"  downloaded {len(state['_links'].get('results', []))} files, about {spent:.0f} km2 used so far")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
