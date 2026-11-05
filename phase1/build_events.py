#!/usr/bin/env python3
"""
Phase 1 - Event list.

Pulls NOAA Storm Events (details) for 2015+, keeps Flash Flood records in the
arid study-area states, and writes one row per event with its begin/end points.

Basin assignment is a separate step (assign_basins.py) because it needs the
Watershed Boundary Dataset, which is a much larger download.

Output: data/events_raw.csv
"""

from __future__ import annotations

import argparse
import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

NCEI = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"

STATES = ["ARIZONA", "NEW MEXICO", "UTAH", "NEVADA", "CALIFORNIA"]

# Sentinel-1A reached routine operations in Oct 2014; 2015 is the first clean year.
FIRST_YEAR = 2015

# Columns worth carrying forward. Storm Events has ~50; most are tornado-specific.
KEEP = [
    "EVENT_ID",
    "EPISODE_ID",
    "STATE",
    "CZ_NAME",
    "CZ_TIMEZONE",
    "YEAR",
    "BEGIN_DATE_TIME",
    "END_DATE_TIME",
    "BEGIN_LAT",
    "BEGIN_LON",
    "END_LAT",
    "END_LON",
    "FLOOD_CAUSE",
    "SOURCE",
    "DEATHS_DIRECT",
    "INJURIES_DIRECT",
    "DAMAGE_PROPERTY",
    "EVENT_NARRATIVE",
]


def list_detail_files(session: requests.Session) -> dict[int, str]:
    """Map year -> filename for every details file at or after FIRST_YEAR.

    NCEI appends a creation date to each filename and republishes periodically,
    so the name cannot be constructed; it has to be read from the index.
    """
    html = session.get(NCEI, timeout=120).text
    found: dict[int, str] = {}
    for name in re.findall(r"StormEvents_details-ftp_v1\.0_d(\d{4})_c(\d+)\.csv\.gz", html):
        year, created = int(name[0]), name[1]
        if year < FIRST_YEAR:
            continue
        fname = f"StormEvents_details-ftp_v1.0_d{year}_c{created}.csv.gz"
        # Keep the most recent revision if NCEI is serving more than one.
        if year not in found or created > found[year][1]:
            found[year] = (fname, created)
    return {y: v[0] for y, v in sorted(found.items())}


def fetch_year(session: requests.Session, fname: str, cache: Path) -> pd.DataFrame:
    local = cache / fname
    if not local.exists():
        r = session.get(NCEI + fname, timeout=600)
        r.raise_for_status()
        local.write_bytes(r.content)
    return pd.read_csv(local, low_memory=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/events_raw.csv")
    ap.add_argument("--cache", default="data/raw/stormevents")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cache = root / args.cache
    cache.mkdir(parents=True, exist_ok=True)
    out = root / args.out
    out.parent.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    files = list_detail_files(session)
    print(f"details files from {FIRST_YEAR}: {len(files)} ({min(files)}-{max(files)})", flush=True)

    frames = []
    for year, fname in files.items():
        df = fetch_year(session, fname, cache)
        ff = df[
            (df["EVENT_TYPE"] == "Flash Flood") & (df["STATE"].isin(STATES))
        ].copy()
        # Every flash flood record carries begin/end points, but be defensive.
        ff = ff[ff["BEGIN_LAT"].notna() & ff["BEGIN_LON"].notna()]
        frames.append(ff[[c for c in KEEP if c in ff.columns]])
        print(f"  {year}: {len(df):>7} rows -> {len(ff):>5} flash floods in study area", flush=True)

    events = pd.concat(frames, ignore_index=True)

    # Storm Events stores local time as "dd-MMM-yy HH:MM:SS" with a separate
    # CZ_TIMEZONE column. Parse to naive local; SAR matching needs UTC, which
    # assign_basins.py resolves once the timezone offset is applied.
    events["BEGIN_DT_LOCAL"] = pd.to_datetime(
        events["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
    )
    events["END_DT_LOCAL"] = pd.to_datetime(
        events["END_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
    )
    bad = events["BEGIN_DT_LOCAL"].isna().sum()
    if bad:
        print(f"WARNING: {bad} rows had an unparseable BEGIN_DATE_TIME", file=sys.stderr)

    events = events.sort_values("BEGIN_DT_LOCAL").reset_index(drop=True)
    events.to_csv(out, index=False)

    print()
    print(f"wrote {out}  ({len(events)} events)")
    print()
    print("by state:")
    print(events["STATE"].value_counts().to_string())
    print()
    print("by year:")
    print(events["YEAR"].value_counts().sort_index().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
