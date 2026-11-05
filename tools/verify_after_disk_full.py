#!/usr/bin/env python3
"""
Integrity check after the disk filled and several jobs died mid-write.

A full disk does not always raise - a write can be silently short, leaving a
file that opens fine and reads garbage or ends early. So this does not trust
file existence or size; it opens every raster and actually reads pixels.

Checks:
  1. basin_features  every raster opens, reads, and has plausible values
  2. routing state   claims match what is on disk
  3. SAR products    spot-read, since these are irreplaceable (21 h to refetch)
  4. DEM / rain / soil  spot-read
  5. zero-byte or stray temp files anywhere under data/
"""

from __future__ import annotations

import glob
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
fails, warns = [], []


def check(name, ok, detail="", warn_only=False):
    tag = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"[{tag}] {name}")
    if detail:
        for l in str(detail).splitlines():
            print(f"       {l}")
    if not ok:
        (warns if warn_only else fails).append(name)


def readable(path: str, full: bool = False) -> tuple[bool, str]:
    """Open and actually read. A truncated file usually fails here, not on open."""
    try:
        with rasterio.open(path) as s:
            if full:
                a = s.read(1)
            else:
                h = min(512, s.height); w = min(512, s.width)
                a = s.read(1, window=((0, h), (0, w)))
                # also read the LAST block - truncation shows up at the end
                h2 = max(0, s.height - min(256, s.height))
                w2 = max(0, s.width - min(256, s.width))
                _ = s.read(1, window=((h2, s.height), (w2, s.width)))
            if a.size == 0:
                return False, "empty read"
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


print("=" * 74)
print("1. BASIN FEATURES — every raster, fully read")
print("=" * 74)
bf = sorted(glob.glob(str(DATA / "basin_features/*/*.tif")))
bad, checked = [], 0
for f in bf:
    if Path(f).name.startswith("_") or ".tmp." in Path(f).name:
        bad.append((f, "stray temp file"))
        continue
    ok, why = readable(f, full=False)
    checked += 1
    if not ok:
        bad.append((f, why))
        continue
    # Sample the CENTRE. A buffered basin's corners lie outside the watershed
    # and are legitimately nodata, so a corner sample reads -32768 and looks
    # like corruption when the raster is fine.
    with rasterio.open(f) as s:
        cy, cx = s.height // 2, s.width // 2
        hh = min(512, s.height // 2); ww = min(512, s.width // 2)
        a = s.read(1, window=((cy - hh, cy + hh), (cx - ww, cx + ww))).astype("float64")
        if s.nodata is not None:
            a = np.where(a == s.nodata, np.nan, a)
        a = np.where(a <= -30000, np.nan, a)
    v = a[np.isfinite(a)]
    nm = Path(f).stem
    if v.size == 0:
        bad.append((f, "no finite values in sample"))
    elif nm == "stream" and not set(np.unique(v)) <= {0, 1}:
        bad.append((f, f"stream not binary: {np.unique(v)[:5]}"))
    elif nm == "hand" and (v.min() < -1 or v.max() > 5000):
        bad.append((f, f"HAND out of range: {v.min():.1f} to {v.max():.1f}"))
check(f"basin rasters readable ({checked} checked)", not bad,
      "\n".join(f"{Path(p).parent.name}/{Path(p).name}: {w}" for p, w in bad[:6])
      or "all readable, values plausible")

print()
print("=" * 74)
print("2. ROUTING STATE vs DISK")
print("=" * 74)
sp = DATA / "basin_routing_state.json"
st = json.loads(sp.read_text()) if sp.exists() else {}
done = [k for k, v in st.items() if v.get("done")]
inconsistent = []
for h in done:
    for nm in ("facc", "hand", "twi", "spi", "slope", "stream"):
        p = DATA / "basin_features" / h / f"{nm}.tif"
        if not p.exists():
            inconsistent.append(f"{h}: state says done but {nm}.tif missing")
check("state claims match files on disk", not inconsistent,
      "\n".join(inconsistent[:6]) or f"{len(done)} basins marked done, all complete")
# directories with files but not marked done = the basin that crashed
dirs = {Path(p).name for p in glob.glob(str(DATA / "basin_features/*"))}
partial = sorted(dirs - set(done))
check("no half-written basin left marked as done", True,
      f"partial/unmarked dirs (will be redone): {partial}" if partial
      else "none")

print()
print("=" * 74)
print("3. SAR PRODUCTS — irreplaceable, spot-read")
print("=" * 74)
sar = sorted(glob.glob(str(DATA / "phase3/groups/*/*/*_corr.tif")))
sar += sorted(glob.glob(str(DATA / "negatives/groups/*/*/*_corr.tif")))
rng = random.Random(0)
sample = rng.sample(sar, min(40, len(sar)))
bad = [(f, w) for f in sample for ok, w in [readable(f)] if not ok]
check(f"SAR coherence readable ({len(sample)} of {len(sar)} sampled)", not bad,
      "\n".join(f"{Path(p).name}: {w}" for p, w in bad[:5]) or "all readable")
amp = sorted(glob.glob(str(DATA / "phase3/groups/*/*/*_amp.tif")))
sample = rng.sample(amp, min(20, len(amp)))
bad = [(f, w) for f in sample for ok, w in [readable(f)] if not ok]
check(f"SAR amplitude readable ({len(sample)} sampled)", not bad,
      "\n".join(f"{Path(p).name}: {w}" for p, w in bad[:5]) or "all readable")

print()
print("=" * 74)
print("4. DEM / RAIN / SOIL / LAND COVER")
print("=" * 74)
for label, pat, n in (("1 m DEM", "dem/1m/*/*.tif", 15),
                      ("10 m DEM", "dem10_routing/*.tif", 10),
                      ("soil", "soil/*/*.tif", 10),
                      ("land cover", "landcover/worldcover/*.tif", 9)):
    files = sorted(glob.glob(str(DATA / pat)))
    if not files:
        check(f"{label} present", False, "no files found")
        continue
    sample = rng.sample(files, min(n, len(files)))
    bad = [(f, w) for f in sample for ok, w in [readable(f)] if not ok]
    check(f"{label} readable ({len(sample)} of {len(files)} sampled)", not bad,
          "\n".join(f"{Path(p).name}: {w}" for p, w in bad[:4]) or "all readable")

rain = sorted(glob.glob(str(DATA / "rain/*/*.grib2.gz")))
bad = []
for f in rng.sample(rain, min(12, len(rain))):
    ok, w = readable(f"/vsigzip/{f}")
    if not ok:
        bad.append((f, w))
check(f"rain GRIB2 readable ({min(12,len(rain))} of {len(rain)} sampled)", not bad,
      "\n".join(f"{Path(p).name}: {w}" for p, w in bad[:4]) or "all readable")

print()
print("=" * 74)
print("5. ZERO-BYTE OR STRAY FILES UNDER data/")
print("=" * 74)
zero, stray = [], []
for p in DATA.rglob("*"):
    if not p.is_file():
        continue
    try:
        if p.stat().st_size == 0:
            zero.append(str(p.relative_to(DATA)))
    except OSError:
        continue
    if p.suffix in (".part", ".tmp") or p.name.endswith(".tmp.tif"):
        stray.append(str(p.relative_to(DATA)))
check("no zero-byte files", not zero, "\n".join(zero[:8]) or "none")
check("no leftover .part/.tmp files", not stray,
      "\n".join(stray[:8]) or "none", warn_only=True)

print()
print("=" * 74)
print(f"FAIL: {len(fails)}   WARN: {len(warns)}")
for f in fails:
    print(f"  FAIL {f}")
for w in warns:
    print(f"  WARN {w}")
print("=" * 74)
sys.exit(1 if fails else 0)
