#!/usr/bin/env python3
"""
NaN / nodata / negative-value audit across DEM and SAR channels.

Carried over from trial 1, where NaNs appeared in some channels and not others,
and negative values needed care.

The trap specific to THIS study area: Death Valley (-86 m) and the Salton Sea
basin (-70 m) are genuinely BELOW SEA LEVEL. Two of the eleven basins have real
negative elevations. Any cleaning rule of the form `elevation > 0` would delete
them entirely - silently, and exactly where the most arid terrain is.

So a sentinel nodata value must be distinguished from a real negative
elevation. They are not the same thing and cannot be filtered the same way.

Also checks whether NaNs line up ACROSS channels. If coherence is missing at a
pixel but amplitude is present, a naive stack produces rows that are partly real
and partly fabricated.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent


def scan_dem(n: int = 8) -> None:
    tiles = sorted(glob.glob(str(ROOT / "data/dem/1m/*/*.tif")))
    random.Random(0).shuffle(tiles)
    print("=" * 74)
    print("1 m DEM")
    print("=" * 74)
    print(f"{'project':<34} {'nodata':>10} {'min':>9} {'max':>9} {'%nodata':>8} {'%neg':>7}")
    for t in tiles[:n]:
        with rasterio.open(t) as s:
            nod = s.nodata
            a = s.read(1, out_shape=(1, 2000, 2000),
                       resampling=rasterio.enums.Resampling.nearest).astype("float64")[0] \
                if False else s.read(1, window=((0, 3000), (0, 3000))).astype("float64")
        raw_min = float(np.nanmin(a))
        if nod is not None:
            valid = a[(a != nod) & np.isfinite(a)]
        else:
            valid = a[np.isfinite(a)]
        pct_nod = 100 * (1 - valid.size / a.size)
        proj = Path(t).parent.name[:33]
        if valid.size == 0:
            # A whole window of nodata. Real: edge tiles cover the corner of a
            # project footprint and can be empty over large areas.
            print(f"{proj:<34} {str(nod):>10} {'--':>9} {'--':>9} "
                  f"{pct_nod:>7.2f}%  ENTIRELY NODATA in this window")
            continue
        pct_neg = 100 * (valid < 0).mean()
        print(f"{proj:<34} {str(nod):>10} {raw_min:>9.1f} {float(valid.max()):>9.1f} "
              f"{pct_nod:>7.2f}% {pct_neg:>6.2f}%")

    # Explicitly check the two below-sea-level basins.
    print()
    print("below-sea-level check (these MUST show real negatives):")
    for pat in ("*Death*", "*Salton*", "*Imperial*", "*FEMAR9Southeast*"):
        hits = glob.glob(str(ROOT / f"data/dem/1m/{pat}/*.tif"))
        if not hits:
            continue
        with rasterio.open(hits[0]) as s:
            a = s.read(1, window=((0, 3000), (0, 3000))).astype("float64")
            nod = s.nodata
        v = a[(a != nod) & np.isfinite(a)] if nod is not None else a[np.isfinite(a)]
        if v.size:
            print(f"  {Path(hits[0]).parent.name:<36} min {v.min():>8.1f} m  "
                  f"{'HAS REAL NEGATIVES' if v.min() < 0 else ''}")


def scan_sar(n: int = 6) -> None:
    st = json.loads((ROOT / "data/phase3_state.json").read_text())
    gids = [v["group_id"] for v in st["groups"].values()
            if v.get("stage") == "done" and v.get("group_id")]
    random.Random(1).shuffle(gids)
    print()
    print("=" * 74)
    print("SAR channels — do NaNs line up across channels?")
    print("=" * 74)
    print(f"{'group':<14} {'corr nan%':>10} {'amp nan%':>9} {'mismatch%':>10}  note")
    worst = 0.0
    for gid in gids[:n]:
        d = ROOT / "data/phase3/groups" / gid / "coevent"
        c = sorted(d.glob("*_corr.tif"))
        a = sorted(d.glob("*_amp.tif"))
        if not c or not a:
            continue
        with rasterio.open(c[0]) as s:
            cv = s.read(1).astype("float64")
            cn = s.nodata
        with rasterio.open(a[0]) as s:
            av = s.read(1).astype("float64")
            an = s.nodata
        cm = ~np.isfinite(cv) | (cv == 0) | ((cn is not None) & (cv == cn))
        am = ~np.isfinite(av) | (av == 0) | ((an is not None) & (av == an))
        mism = 100 * (cm ^ am).mean()
        worst = max(worst, mism)
        note = "channels disagree" if mism > 1 else ""
        print(f"{gid:<14} {100*cm.mean():>9.2f}% {100*am.mean():>8.2f}% {mism:>9.2f}%  {note}")
    print()
    print(f"worst channel mismatch: {worst:.2f}%")
    print("A pixel valid in one channel and missing in the other must be dropped")
    print("from the stack, not filled - otherwise a feature row is part real,")
    print("part fabricated.")


def main() -> int:
    scan_dem()
    scan_sar()
    print()
    print("=" * 74)
    print("RULE FOR FEATURE BUILDING")
    print("=" * 74)
    print("  1. Never filter elevation with `> 0`. Use the raster's own nodata")
    print("     sentinel. Death Valley and Salton Sea are legitimately negative.")
    print("  2. Build a single valid-mask as the AND of every channel, and apply")
    print("     it to all of them, so no row mixes real and missing values.")
    print("  3. Record the dropped-pixel count per event (NEXT_STEPS section 5).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
