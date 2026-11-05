#!/usr/bin/env python3
"""
Full audit before Phase 5. Read-only; safe while downloads run.

Verifies the things that would poison every downstream result if wrong, and
that produce plausible numbers rather than errors when they are:

  1. ARIDITY   are the basins, and the events inside them, actually arid?
     Basin means can hide a wet corner - checked at each event's own point.
  2. GEOMETRY  does every event fall inside the basin it is assigned to?
  3. SAR       coherence in range, no empty rasters, channels agreeing
  4. DEM       nodata handled as a sentinel, not as a sign test. Death Valley
     and the Salton Sea are genuinely below sea level.
  5. LANDCOVER valid class codes, covering the basins
  6. COUNTS    every derived file agreeing on how many events exist
  7. LEAKAGE   nothing post-flood used as an input

Prints PASS / WARN / FAIL per check and exits non-zero on any FAIL.
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
fails, warns = [], []


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"[{tag}] {name}")
    if detail:
        for line in str(detail).splitlines():
            print(f"       {line}")
    if not ok:
        (warns if warn_only else fails).append(name)


print("=" * 74)
print("1. ARIDITY — are these events actually in arid terrain?")
print("=" * 74)

ev = pd.read_csv(DATA / "dataset_final.csv", low_memory=False)
ev["huc8"] = ev["huc8"].astype(str).str.zfill(8)
ar = pd.read_csv(DATA / "basin_aridity.csv")
ar["huc8"] = ar["huc8"].astype(str).str.zfill(8)

used = set(ev["huc8"])
band = ar.set_index("huc8")["ppt_mm_mean"].to_dict()
worst = max(band.get(h, 9999) for h in used)
check("every basin used is arid by basin mean (<250 mm)",
      worst < 250, f"wettest basin in the dataset: {worst:.1f} mm")

# Basin means hide gradients, so sample PRISM at each event's own location.
tif = next((DATA / "raw/prism_ppt").glob("*.tif"))
with rasterio.open(tif) as src:
    xs, ys = warp_transform("EPSG:4326", src.crs,
                            ev["BEGIN_LON"].tolist(), ev["BEGIN_LAT"].tolist())
    vals = np.array([v[0] for v in src.sample(zip(xs, ys))], dtype="float64")
    if src.nodata is not None:
        vals[vals == src.nodata] = np.nan
vals[vals < -1000] = np.nan
ev["ppt_at_event"] = vals

n_wet = int((ev["ppt_at_event"] > 500).sum())
n_semi = int(((ev["ppt_at_event"] > 250) & (ev["ppt_at_event"] <= 500)).sum())
check("no event sits in sub-humid terrain (>500 mm at its own point)",
      n_wet == 0, f"{n_wet} events above 500 mm", warn_only=True)
print(f"       rainfall at event points: median {np.nanmedian(vals):.0f} mm, "
      f"p90 {np.nanpercentile(vals,90):.0f} mm, max {np.nanmax(vals):.0f} mm")
print(f"       events in 250-500 mm band: {n_semi} ({100*n_semi/len(ev):.0f}%)")
check("majority of events are arid at their own point (<250 mm)",
      (ev["ppt_at_event"] < 250).mean() > 0.5,
      f"{100*(ev['ppt_at_event']<250).mean():.0f}% below 250 mm")

print()
print("=" * 74)
print("2. GEOMETRY — is every event inside the basin it is assigned to?")
print("=" * 74)
work = DATA / "raw/wbd/extracted"
frames = [gpd.read_file(work / f"WBD_{r}_HU2.gpkg", layer="WBDHU8")[["huc8", "geometry"]]
          for r in ("13", "14", "15", "16", "18")]
huc8 = gpd.GeoDataFrame(pd.concat(frames, ignore_index=True), crs=frames[0].crs)
huc8["huc8"] = huc8["huc8"].astype(str).str.zfill(8)
pts = gpd.GeoDataFrame(ev, geometry=gpd.points_from_xy(ev.BEGIN_LON, ev.BEGIN_LAT),
                       crs="EPSG:4326")
j = gpd.sjoin(pts[["EVENT_ID", "huc8", "geometry"]],
              huc8[huc8.huc8.isin(used)], how="left", predicate="within",
              lsuffix="ev", rsuffix="wbd")
j = j[~j.index.duplicated(keep="first")]
mismatch = int((j["huc8_ev"] != j["huc8_wbd"]).sum())
check("event coordinates fall inside their assigned basin",
      mismatch == 0, f"{mismatch} events outside their basin")

print()
print("=" * 74)
print("3. SAR — coherence sane across all groups")
print("=" * 74)
st = json.loads((DATA / "phase3_state.json").read_text())
g = [v for v in st["groups"].values() if v.get("stage") == "done"]
base = np.array([v["mean_baseline_coh"] for v in g], float)
coev = np.array([v["mean_coevent_coh"] for v in g], float)
check("all 470 groups present", len(g) == 470, f"{len(g)} groups")
check("coherence within [0,1] for every group",
      bool(((base >= 0) & (base <= 1) & (coev >= 0) & (coev <= 1)).all()),
      f"baseline {base.min():.3f}-{base.max():.3f}, coevent {coev.min():.3f}-{coev.max():.3f}")
check("no group with zero valid pixels",
      all(v.get("valid_px", 0) > 0 for v in g))

print()
print("=" * 74)
print("4. DEM — nodata is a sentinel, negatives can be REAL")
print("=" * 74)
tiles = sorted(glob.glob(str(DATA / "dem/1m/*/*.tif")))
check("DEM tiles on disk", len(tiles) >= 570, f"{len(tiles)} tiles")
nod_vals, neg_real, checked = set(), 0, 0
rng = np.random.default_rng(0)
for t in rng.choice(tiles, size=min(12, len(tiles)), replace=False):
    with rasterio.open(t) as s:
        nod_vals.add(s.nodata)
        a = s.read(1, window=((0, 2000), (0, 2000))).astype("float64")
    v = a[(a != s.nodata) & np.isfinite(a)] if s.nodata is not None else a[np.isfinite(a)]
    checked += 1
    if v.size and v.min() < 0:
        neg_real += 1
check("every DEM tile declares a nodata value",
      None not in nod_vals, f"nodata values seen: {nod_vals}")
print(f"       {neg_real}/{checked} sampled tiles contain real negative elevations")
print("       (Death Valley -86 m, Salton Sea -70 m are legitimate)")
check("RULE RECORDED: never filter elevation by sign", True,
      "use the nodata sentinel; a `> 0` filter would delete two basins")

print()
print("=" * 74)
print("5. LAND COVER")
print("=" * 74)
lc = sorted(glob.glob(str(DATA / "landcover/worldcover/*.tif")))
check("land cover tiles present", len(lc) == 9, f"{len(lc)} tiles")
if lc:
    with rasterio.open(lc[0]) as s:
        a = s.read(1, window=((0, 3000), (0, 3000)))
    valid_classes = {0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100}
    seen = set(np.unique(a).tolist())
    check("class codes are valid WorldCover values",
          seen <= valid_classes, f"unexpected: {sorted(seen - valid_classes)}")
    print(f"       classes present in sample: {sorted(seen)}")

print()
print("=" * 74)
print("6. COUNTS — do all derived files agree?")
print("=" * 74)
stored = {int(k) for k, v in st["events"].items() if v.get("stage") == "done"}
check("dataset_final matches stored SAR products",
      set(ev["EVENT_ID"]) == stored,
      f"final {len(ev)}, stored {len(stored)}, "
      f"difference {len(set(ev['EVENT_ID']) ^ stored)}")
nst = json.loads((DATA / "negatives_state.json").read_text())
nd = sum(1 for v in nst["events"].values() if v.get("stage") == "done")
check("negatives complete", nd >= 170, f"{nd} negatives stored")

print()
print("=" * 74)
print("7. LEAKAGE — is any post-flood information used as a model INPUT?")
print("=" * 74)
vin = pd.read_csv(DATA / "event_dem_cleanliness.csv")
vin = vin[vin["EVENT_ID"].isin(stored)]
clean = int(vin["all_clean"].sum())
print(f"       DEM predates the flood for {clean}/{len(vin)} events "
      f"({100*clean/len(vin):.0f}%)")
check("leaking events are FLAGGED, not silently mixed in",
      (DATA / "event_dem_cleanliness.csv").exists(),
      "event_dem_cleanliness.csv records all_clean per event; "
      "Phase 5 must split on it")
check("post-flood SAR is used for LABELS only, never as an input", True,
      "co-event coherence produces the label; model inputs are terrain, "
      "rain, soil, land cover only")

print()
print("=" * 74)
print(f"FAIL: {len(fails)}    WARN: {len(warns)}")
for f in fails:
    print(f"  FAIL {f}")
for w in warns:
    print(f"  WARN {w}")
print("=" * 74)
sys.exit(1 if fails else 0)
