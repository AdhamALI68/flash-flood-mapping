#!/usr/bin/env python3
"""
Phase 1b - Basin assignment.

Joins each Storm Events flash flood point to its HUC8 watershed and converts
event times to UTC, which is what the Phase 2 SAR search needs.

HUC codes are nested by digit prefix, so HUC6/HUC4 come free from the HUC8 join.
That matters for Phase 8: HUC8 basins that share a HUC4 are not independent
regions, and a cross-region split that ignores this leaks.

Output: data/events.csv  +  data/basin_summary.csv
"""

from __future__ import annotations

import argparse
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import pandas as pd

# HUC2 regions covering AZ / NM / UT / NV / southern CA.
REGIONS = ["13", "14", "15", "16", "18"]


def load_huc8(wbd_dir: Path, work: Path) -> gpd.GeoDataFrame:
    """Read the WBDHU8 layer out of each region's zipped GeoPackage."""
    frames = []
    for r in REGIONS:
        zp = wbd_dir / f"WBD_{r}_HU2_GPKG.zip"
        if not zp.exists():
            raise SystemExit(f"missing {zp} - run the WBD download first")
        target = work / f"WBD_{r}_HU2.gpkg"
        if not target.exists():
            with zipfile.ZipFile(zp) as z:
                name = next(n for n in z.namelist() if n.lower().endswith(".gpkg"))
                target.write_bytes(z.read(name))
        g = gpd.read_file(target, layer="WBDHU8")
        g = g[["huc8", "name", "areasqkm", "geometry"]].copy()
        g["huc2"] = r
        frames.append(g)
        print(f"  region {r}: {len(g)} HUC8 basins", flush=True)
    return gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)


def to_utc(local: pd.Series, tz: pd.Series) -> pd.Series:
    """Storm Events stores local time plus a CZ_TIMEZONE like 'MST-7'.

    The trailing number is the offset from UTC, so UTC = local + offset. The
    column is not always clean, so anything unparseable becomes NaT rather than
    silently landing in the wrong hour - a wrong timestamp here would silently
    pair the wrong SAR scenes in Phase 2.
    """
    offs = tz.astype(str).str.extract(r"(-?\d+)", expand=False).astype("Float64")
    return local + pd.to_timedelta(-offs.astype("float"), unit="h")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="data/events_raw.csv")
    ap.add_argument("--wbd", default="data/raw/wbd")
    ap.add_argument("--out", default="data/events.csv")
    ap.add_argument("--summary", default="data/basin_summary.csv")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    work = root / "data" / "raw" / "wbd" / "extracted"
    work.mkdir(parents=True, exist_ok=True)

    ev = pd.read_csv(root / args.events, low_memory=False)
    ev["BEGIN_DT_LOCAL"] = pd.to_datetime(ev["BEGIN_DT_LOCAL"], errors="coerce")
    ev["END_DT_LOCAL"] = pd.to_datetime(ev["END_DT_LOCAL"], errors="coerce")
    print(f"events in: {len(ev)}")

    ev["BEGIN_UTC"] = to_utc(ev["BEGIN_DT_LOCAL"], ev["CZ_TIMEZONE"])
    ev["END_UTC"] = to_utc(ev["END_DT_LOCAL"], ev["CZ_TIMEZONE"])
    lost = ev["BEGIN_UTC"].isna().sum()
    if lost:
        print(f"WARNING: {lost} events have no usable UTC time")

    print("loading HUC8 basins...")
    huc8 = load_huc8(root / args.wbd, work)

    pts = gpd.GeoDataFrame(
        ev,
        geometry=gpd.points_from_xy(ev["BEGIN_LON"], ev["BEGIN_LAT"]),
        crs="EPSG:4326",
    ).to_crs(huc8.crs)

    joined = gpd.sjoin(pts, huc8, how="left", predicate="within")
    # A point on a shared boundary can match two basins; keep the first.
    joined = joined[~joined.index.duplicated(keep="first")]

    unmatched = joined["huc8"].isna().sum()
    print(f"joined: {len(joined) - unmatched} matched, {unmatched} outside the 5 regions")

    joined["huc6"] = joined["huc8"].str[:6]
    joined["huc4"] = joined["huc8"].str[:4]

    cols = [
        "EVENT_ID", "EPISODE_ID", "STATE", "CZ_NAME",
        "BEGIN_UTC", "END_UTC", "BEGIN_DT_LOCAL", "CZ_TIMEZONE",
        "BEGIN_LAT", "BEGIN_LON", "END_LAT", "END_LON",
        "huc8", "huc6", "huc4", "name", "areasqkm", "huc2",
        "FLOOD_CAUSE", "SOURCE", "DEATHS_DIRECT", "DAMAGE_PROPERTY",
    ]
    out = joined[[c for c in cols if c in joined.columns]].rename(
        columns={"name": "basin_name", "areasqkm": "basin_area_km2"}
    )
    out = out[out["huc8"].notna() & out["BEGIN_UTC"].notna()].copy()
    out.to_csv(root / args.out, index=False)
    print(f"\nwrote {args.out}  ({len(out)} events with basin + UTC time)")

    summary = (
        out.groupby(["huc8", "basin_name", "huc4", "basin_area_km2"])
        .agg(events=("EVENT_ID", "count"),
             first=("BEGIN_UTC", "min"),
             last=("BEGIN_UTC", "max"))
        .reset_index()
        .sort_values("events", ascending=False)
    )
    summary.to_csv(root / args.summary, index=False)

    print(f"wrote {args.summary}  ({len(summary)} basins with >=1 event)")
    print()
    print("basins by event count:")
    for n in (5, 10, 20, 50):
        print(f"  HUC8 basins with >= {n:>2} events: {(summary['events'] >= n).sum()}")
    print()
    print("top 20 HUC8 basins:")
    print(summary.head(20).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
