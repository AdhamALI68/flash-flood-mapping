#!/usr/bin/env python3
"""
NEXT_STEPS 4b step 1 - are gauged and ungauged events comparable?

The threshold gets calibrated on 358 gauge-validatable events and applied to all
623. Gauges sit near infrastructure and perennial reaches, so the gauged subset
is a convenience sample. This measures whether the two groups actually resemble
each other on covariates available for both.

Reports full distributions plus a standardised mean difference, because two
groups can share a mean and have completely different spreads. Convention for
|SMD|: <0.2 negligible, 0.2-0.5 small, 0.5-0.8 moderate, >0.8 large.

Covariates needing no bulk download are done here. Slope needs a DEM and waits
for that download; it is listed as pending rather than skipped.

Output: data/transferability.csv, data/event_covariates.csv
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import requests
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parent.parent
EPQS = "https://epqs.nationalmap.gov/v1/json"


def sample_prism(lons, lats) -> np.ndarray:
    """Mean annual precipitation at each event point, not the basin mean.

    Basin means hide within-basin gradients - Whitewater River spans
    68-1,189 mm - and the gauged/ungauged split may fall along exactly such a
    gradient, which a basin mean would conceal.
    """
    tif = next((ROOT / "data/raw/prism_ppt").glob("*.tif"))
    with rasterio.open(tif) as src:
        xs, ys = warp_transform("EPSG:4326", src.crs, list(lons), list(lats))
        vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
        if src.nodata is not None:
            vals[vals == src.nodata] = np.nan
    vals[vals < -1000] = np.nan
    return vals


def fetch_elevation(session, lon: float, lat: float) -> float | None:
    try:
        r = session.get(EPQS, params={"x": lon, "y": lat, "units": "Meters",
                                      "wkid": 4326, "includeDate": "false"},
                        timeout=30)
        if r.status_code != 200:
            return None
        d = r.json()
        v = d.get("value")
        return float(v) if v is not None and str(v) != "-1000000" else None
    except Exception:
        return None


def smd(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference, pooled SD."""
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < 2 or len(b) < 2:
        return np.nan
    sp = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1))
                 / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / sp if sp > 0 else np.nan


def describe(name, a, b, rows):
    a, b = np.asarray(a, "float64"), np.asarray(b, "float64")
    av, bv = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(av) < 2 or len(bv) < 2:
        rows.append({"covariate": name, "note": "insufficient data"})
        return
    d = smd(a, b)
    rows.append({
        "covariate": name,
        "gauged_n": len(av), "ungauged_n": len(bv),
        "gauged_median": round(float(np.median(av)), 1),
        "ungauged_median": round(float(np.median(bv)), 1),
        "gauged_iqr": f"{np.percentile(av,25):.0f}-{np.percentile(av,75):.0f}",
        "ungauged_iqr": f"{np.percentile(bv,25):.0f}-{np.percentile(bv,75):.0f}",
        "smd": round(float(d), 3),
        "magnitude": ("negligible" if abs(d) < 0.2 else "small" if abs(d) < 0.5
                      else "moderate" if abs(d) < 0.8 else "LARGE"),
    })


def main() -> int:
    ev = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    ev["huc8"] = ev["huc8"].astype(str).str.zfill(8)

    rs = [json.loads(l) for l in (ROOT / "data/downstream_gauges.jsonl").read_text().splitlines() if l.strip()]
    gz = pd.DataFrame([{"EVENT_ID": r["EVENT_ID"], "has_gauge": len(r["sites"]) > 0}
                       for r in rs]).drop_duplicates("EVENT_ID")
    ev = ev.merge(gz, on="EVENT_ID", how="left")
    ev["has_gauge"] = ev["has_gauge"].fillna(False)

    vin = pd.read_csv(ROOT / "data/lidar_vintage.csv")
    ev = ev.merge(vin[["EVENT_ID", "has_pre_event_lidar"]], on="EVENT_ID", how="left")

    print(f"events: {len(ev)}  gauged: {int(ev.has_gauge.sum())}  "
          f"ungauged: {int((~ev.has_gauge).sum())}")

    ev["ppt_at_point"] = sample_prism(ev["BEGIN_LON"], ev["BEGIN_LAT"])
    print(f"PRISM sampled at event points ({ev.ppt_at_point.notna().sum()} valid)")

    print("fetching elevations from EPQS...", flush=True)
    session = requests.Session()
    elevs = []
    for i, r in enumerate(ev.itertuples(index=False), 1):
        elevs.append(fetch_elevation(session, float(r.BEGIN_LON), float(r.BEGIN_LAT)))
        if i % 100 == 0:
            print(f"  {i}/{len(ev)}", flush=True)
        time.sleep(0.05)
    ev["elevation_m"] = elevs
    print(f"  elevations obtained: {ev.elevation_m.notna().sum()}/{len(ev)}")

    ev.to_csv(ROOT / "data/event_covariates.csv", index=False)

    g = ev[ev["has_gauge"]]
    u = ev[~ev["has_gauge"]]
    rows = []
    describe("rainfall_mm_at_point", g["ppt_at_point"], u["ppt_at_point"], rows)
    describe("elevation_m", g["elevation_m"], u["elevation_m"], rows)
    describe("basin_area_km2", g["basin_area_km2"], u["basin_area_km2"], rows)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "data/transferability.csv", index=False)

    print()
    print("=" * 78)
    print("GAUGED vs UNGAUGED EVENTS")
    print("=" * 78)
    print(out.to_string(index=False))
    print()
    print("pending, needs the DEM download: slope")
    print("pending, needs point APIs      : SSURGO soil class, NLCD land cover")
    print()

    print("composition of each group:")
    comp = ev.groupby(ev["has_gauge"].map({True: "gauged", False: "ungauged"})).agg(
        events=("EVENT_ID", "size"),
        regions=("huc4", "nunique"),
        basins=("huc8", "nunique"),
        pct_clean_lidar=("has_pre_event_lidar", lambda s: round(100 * s.mean(), 0)),
    )
    print(comp.to_string())
    print()
    print("events per region, gauged vs ungauged:")
    piv = ev.pivot_table(index="huc4", columns=ev["has_gauge"].map({True: "gauged", False: "ungauged"}),
                         values="EVENT_ID", aggfunc="count", fill_value=0)
    print(piv.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
