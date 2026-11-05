#!/usr/bin/env python3
"""
NEXT_STEPS Track B, step 1 - which USGS gauges can validate these events?

Threshold selection needs an independent record of whether each event was a real
flood. That comes from streamflow gauges. This finds every gauge with discharge
data in the 11 STRICT basins and reports, per basin, how many exist and how far
they sit from the events - because the honest answer in the desert may be
"not many, and not close".

Daily MAXIMUM (statCd 00001) matters more than daily mean here. A desert flash
flood can peak for two hours; averaged over 24 it may barely register. Sites are
recorded with which statistics they actually publish.

No thresholds and no judgements - just the inventory, so the limits of the
validation are visible before it is attempted.

Output: data/gauges.csv, data/gauge_coverage.csv
"""

from __future__ import annotations

import io
from pathlib import Path

import geopandas as gpd
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
SITE_URL = "https://waterservices.usgs.gov/nwis/site/"
EQUAL_AREA = "EPSG:5070"
DISCHARGE = "00060"


def sites_for_huc(session: requests.Session, huc8: str) -> pd.DataFrame:
    """Every gauge in a HUC8 that publishes discharge as a daily series."""
    r = session.get(SITE_URL, params={
        "format": "rdb", "huc": huc8, "parameterCd": DISCHARGE,
        "siteType": "ST", "hasDataTypeCd": "dv", "siteStatus": "all",
    }, timeout=180)
    if r.status_code == 404 or not r.text.strip():
        return pd.DataFrame()
    r.raise_for_status()
    lines = [l for l in r.text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    # RDB: header row, then a format row, then data.
    df = pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])), sep="\t",
                     dtype=str, low_memory=False)
    return df


def main() -> int:
    sets = pd.read_csv(ROOT / "data/basin_sets.csv")
    sets["huc8"] = sets["huc8"].astype(str).str.zfill(8)
    strict = sets[sets["set"] == "STRICT"]

    ev = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    ev["huc8"] = ev["huc8"].astype(str).str.zfill(8)

    session = requests.Session()
    frames = []
    for r in strict.itertuples():
        df = sites_for_huc(session, r.huc8)
        if df.empty:
            print(f"  {r.huc8} {r.basin_name:<34} NO discharge gauges")
            continue
        df["huc8"] = r.huc8
        df["basin_name"] = r.basin_name
        frames.append(df)
        print(f"  {r.huc8} {r.basin_name:<34} {len(df):>3} gauges", flush=True)

    if not frames:
        print("no gauges found in any strict basin")
        return 1

    g = pd.concat(frames, ignore_index=True)
    keep = [c for c in ("site_no", "station_nm", "dec_lat_va", "dec_long_va",
                        "huc_cd", "drain_area_va", "huc8", "basin_name")
            if c in g.columns]
    g = g[keep].copy()
    g["dec_lat_va"] = pd.to_numeric(g["dec_lat_va"], errors="coerce")
    g["dec_long_va"] = pd.to_numeric(g["dec_long_va"], errors="coerce")
    g = g.dropna(subset=["dec_lat_va", "dec_long_va"]).drop_duplicates("site_no")
    g.to_csv(ROOT / "data/gauges.csv", index=False)

    # How close is the nearest gauge to each event? This is the real limit on
    # how much of the dataset can be validated at all.
    gp = gpd.GeoDataFrame(g, geometry=gpd.points_from_xy(g.dec_long_va, g.dec_lat_va),
                          crs="EPSG:4326").to_crs(EQUAL_AREA)
    ep = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.BEGIN_LON, ev.BEGIN_LAT),
                          crs="EPSG:4326").to_crs(EQUAL_AREA)

    near = gpd.sjoin_nearest(ep[["EVENT_ID", "huc8", "basin_name", "geometry"]],
                             gp[["site_no", "geometry"]],
                             how="left", distance_col="dist_m")
    near = near[~near.index.duplicated(keep="first")]
    near["dist_km"] = (near["dist_m"] / 1000).round(2)
    near[["EVENT_ID", "huc8", "basin_name", "site_no", "dist_km"]].to_csv(
        ROOT / "data/gauge_coverage.csv", index=False)

    print()
    print("=" * 66)
    print(f"gauges found : {len(g)} across {g['huc8'].nunique()} of {len(strict)} basins")
    print(f"events       : {len(ev)}")
    print()
    print("distance from each event to its nearest discharge gauge:")
    for km in (5, 10, 25, 50):
        n = (near["dist_km"] <= km).sum()
        print(f"  within {km:>3} km : {n:>4} events ({100*n/len(near):.0f}%)")
    print(f"  median distance: {near['dist_km'].median():.1f} km")
    print()
    print("per basin:")
    per = (near.groupby(["huc8", "basin_name"])
               .agg(events=("EVENT_ID", "size"),
                    median_km=("dist_km", "median"),
                    within_25km=("dist_km", lambda s: int((s <= 25).sum())))
               .reset_index().sort_values("median_km"))
    print(per.to_string(index=False))
    print()
    print("wrote data/gauges.csv and data/gauge_coverage.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
