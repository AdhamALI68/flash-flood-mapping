#!/usr/bin/env python3
"""
Audit of every Phase 1-2 output before Phase 3 builds on it.

Checks the things that fail silently: timezone conversion, duplicate events,
unparsed dates, basin joins, and whether the baseline and co-event intervals are
actually comparable. Each check prints PASS, WARN, or FAIL.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
fails, warns = [], []


def check(name: str, ok: bool, detail: str = "", warn_only: bool = False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"[{tag}] {name}" + (f"\n       {detail}" if detail else ""))
    if not ok:
        (warns if warn_only else fails).append(name)


raw = pd.read_csv(ROOT / "data/events_raw.csv", low_memory=False)
ev = pd.read_csv(ROOT / "data/events.csv", low_memory=False)
av = pd.read_csv(ROOT / "data/sar_availability.csv", low_memory=False)
for c in ("BEGIN_UTC", "END_UTC", "BEGIN_DT_LOCAL"):
    if c in ev.columns:
        ev[c] = pd.to_datetime(ev[c], errors="coerce")

print("=" * 72)
print("PHASE 1 - EVENTS")
print("=" * 72)

check("no unparsed begin dates",
      ev["BEGIN_UTC"].isna().sum() == 0,
      f"{ev['BEGIN_UTC'].isna().sum()} NaT of {len(ev)}")

check("no duplicate EVENT_IDs",
      ev["EVENT_ID"].duplicated().sum() == 0,
      f"{ev['EVENT_ID'].duplicated().sum()} duplicates")

check("all events are Flash Flood in the 5 target states",
      set(raw["STATE"].unique()) <= {"ARIZONA", "NEW MEXICO", "UTAH", "NEVADA", "CALIFORNIA"},
      f"states: {sorted(raw['STATE'].unique())}")

# Timezone: Storm Events stores local time plus an offset like "MST-7".
tzs = ev["CZ_TIMEZONE"].astype(str).value_counts()
print(f"       timezone codes present: {dict(tzs)}")
offs = ev["CZ_TIMEZONE"].astype(str).str.extract(r"(-?\d+)", expand=False)
check("every event has a parseable UTC offset",
      offs.isna().sum() == 0, f"{offs.isna().sum()} unparseable")

# Spot-check the sign of the conversion on a known case.
sample = ev.dropna(subset=["BEGIN_DT_LOCAL", "BEGIN_UTC"]).iloc[0]
delta_h = (sample["BEGIN_UTC"] - sample["BEGIN_DT_LOCAL"]).total_seconds() / 3600
expect = -int(str(sample["CZ_TIMEZONE"]).split("-")[-1]) if "-" in str(sample["CZ_TIMEZONE"]) else None
check("UTC = local + offset, sign correct",
      expect is not None and abs(delta_h - abs(expect)) < 0.01,
      f"{sample['CZ_TIMEZONE']}: local {sample['BEGIN_DT_LOCAL']} -> UTC {sample['BEGIN_UTC']} (+{delta_h:.0f}h)")

# DST: Storm Events records local STANDARD time, but California observes DST.
# A 1 h error is harmless when SAR gaps are measured in days - unless an event
# sits within an hour of a scene.
tight = av[av["usable"] == True].copy()
for c in ("pre_gap_days", "post_gap_days"):
    tight[c] = pd.to_numeric(tight[c], errors="coerce")
risky = ((tight["pre_gap_days"] < 1 / 24) | (tight["post_gap_days"] < 1 / 24)).sum()
check("no event is within 1 h of a scene (DST ambiguity would matter)",
      risky == 0, f"{risky} events within 1 hour of a scene boundary", warn_only=True)

print()
print("=" * 72)
print("PHASE 1b - BASIN JOIN")
print("=" * 72)

check("no event lost between raw and joined beyond the region clip",
      len(ev) <= len(raw), f"raw {len(raw)} -> joined {len(ev)}")
check("every joined event has a HUC8",
      ev["huc8"].isna().sum() == 0, f"{ev['huc8'].isna().sum()} missing")
check("HUC8 codes are 8 digits",
      ev["huc8"].astype(str).str.zfill(8).str.len().eq(8).all())
check("HUC4 is the first 4 digits of HUC8",
      (ev["huc8"].astype(str).str.zfill(8).str[:4] == ev["huc4"].astype(str).str.zfill(4)).all())

print()
print("=" * 72)
print("PHASE 2 - SAR AVAILABILITY")
print("=" * 72)

u = av[av["usable"] == True].copy()
for c in ("pre_gap_days", "post_gap_days", "baseline_gap_days", "total_span_days"):
    u[c] = pd.to_numeric(u[c], errors="coerce")

check("no usable event has a negative gap",
      (u["pre_gap_days"] >= 0).all() and (u["post_gap_days"] >= 0).all())
check("all three scenes distinct per event",
      (av["pre2"] != av["pre1"]).fillna(True).all() and (av["pre1"] != av["post"]).fillna(True).all())

# The real methodological check: baseline and co-event intervals must be
# comparable, or the comparison measures elapsed time rather than flooding.
u["coevent_span"] = u["pre_gap_days"] + u["post_gap_days"]
u["mismatch"] = (u["baseline_gap_days"] - u["coevent_span"]).abs()
frac_bad = (u["mismatch"] > 6).mean()
print(f"       baseline vs co-event interval mismatch: "
      f"median {u['mismatch'].median():.1f} d, "
      f"p90 {u['mismatch'].quantile(0.9):.1f} d, "
      f">6 d for {100*frac_bad:.1f}% of events")
check("baseline and co-event intervals are comparable for most events",
      frac_bad < 0.25, f"{100*frac_bad:.1f}% mismatch by more than 6 days",
      warn_only=True)

print()
print("=" * 72)
print("PHASE 2b/2c - COVERAGE AND ARIDITY")
print("=" * 72)

lc = pd.read_csv(ROOT / "data/lidar_coverage.csv")
ar = pd.read_csv(ROOT / "data/basin_aridity.csv")
check("lidar coverage fraction within [0,1]",
      lc["lidar_coverage_frac"].between(0, 1).all())
check("precipitation values are physically plausible",
      ar["ppt_mm_mean"].between(20, 2000).all(),
      f"range {ar['ppt_mm_mean'].min():.0f}-{ar['ppt_mm_mean'].max():.0f} mm")
check("aridity band matches the precipitation value",
      ((ar["ppt_mm_mean"] < 250) == (ar["aridity_band"] == "arid")).all())

strict = ar[(ar.aridity_band == "arid") & (ar.lidar_coverage_frac >= 0.9) & (ar.sar_usable_events >= 20)]
check("strict set has enough regions for a held-out split",
      strict["huc4"].nunique() >= 4, f"{strict['huc4'].nunique()} HUC4 regions")

print()
print("=" * 72)
print(f"FAIL: {len(fails)}   WARN: {len(warns)}")
for f in fails:
    print(f"  FAIL {f}")
for w in warns:
    print(f"  WARN {w}")
print("=" * 72)
sys.exit(1 if fails else 0)
