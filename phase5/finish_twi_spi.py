#!/usr/bin/env python3
"""
Complete TWI and SPI for basins whose routing was interrupted.

The disk filled exactly at the step that writes these two, which comes last in
route(). Everything expensive - depression filling, D8 pointer, flow
accumulation, HAND - completed and verified fine.

TWI and SPI are arithmetic on facc and slope:
    area  = max(facc, 1) * res^2          contributing area, m2
    tanB  = tan(max(slope, 0.1 deg))      local gradient, floored to avoid /0
    TWI   = ln(area / tanB)               wetness: big catchment, flat ground
    SPI   = ln(area * tanB)               erosive power: big catchment, steep

So they can be rebuilt in minutes instead of re-routing for hours.

Processed in horizontal blocks: these rasters run to ~200 M cells, and holding
several float64 copies is what exhausted memory-and-disk headroom in the first
place.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"
RES = 10.0
BLOCK = 2048


def finish(d: Path) -> dict | None:
    facc_p, slope_p = d / "facc.tif", d / "slope.tif"
    if not (facc_p.exists() and slope_p.exists()):
        return None
    with rasterio.open(facc_p) as fs, rasterio.open(slope_p) as ss:
        if (fs.width, fs.height) != (ss.width, ss.height):
            print(f"    shape mismatch facc {fs.shape} vs slope {ss.shape}")
            return None
        prof = fs.profile
        prof.pop("blockysize", None)
        prof.pop("blockxsize", None)
        prof.update(dtype="float32", compress="deflate", predictor=2,
                    tiled=True, blockxsize=512, blockysize=512, zlevel=6,
                    nodata=np.nan)
        twi_p, spi_p = d / "twi.tif", d / "spi.tif"
        stats = {"max_facc_cells": 0}
        with rasterio.open(twi_p, "w", **prof) as tw, \
             rasterio.open(spi_p, "w", **prof) as sp:
            for y0 in range(0, fs.height, BLOCK):
                h = min(BLOCK, fs.height - y0)
                win = ((y0, y0 + h), (0, fs.width))
                a = fs.read(1, window=win).astype("float32")
                sl = ss.read(1, window=win).astype("float32")
                bad = ~np.isfinite(a) | (a <= -30000) | ~np.isfinite(sl) | (sl <= -30000)
                a = np.where(bad, np.nan, a)
                sl = np.where(bad, np.nan, sl)
                stats["max_facc_cells"] = max(
                    stats["max_facc_cells"],
                    int(np.nanmax(a)) if np.isfinite(a).any() else 0)
                area = np.maximum(np.nan_to_num(a, nan=1.0), 1.0) * RES * RES
                tanb = np.tan(np.radians(np.maximum(np.nan_to_num(sl, nan=0.1), 0.1)))
                with np.errstate(divide="ignore", invalid="ignore"):
                    twi = np.log(area / tanb).astype("float32")
                    spi = np.log(area * tanb).astype("float32")
                twi[bad] = np.nan
                spi[bad] = np.nan
                tw.write(twi, 1, window=win)
                sp.write(spi, 1, window=win)
        stats["cells"] = fs.width * fs.height
        stats["max_drainage_km2"] = round(stats["max_facc_cells"] * RES * RES / 1e6, 1)
    return stats


def main() -> int:
    REQ = ("facc", "hand", "slope", "stream", "twi", "spi")
    state = {}
    for d in sorted(BF.iterdir()):
        if not d.is_dir():
            continue
        have = {n for n in REQ if (d / f"{n}.tif").exists()}
        if {"facc", "slope"} - have:
            print(f"  {d.name}: no facc/slope — needs full routing")
            continue
        if {"twi", "spi"} <= have:
            print(f"  {d.name}: already complete")
        else:
            print(f"  {d.name}: computing TWI/SPI from facc+slope...", flush=True)
            st = finish(d)
            if st is None:
                print(f"    failed")
                continue
            print(f"    max drainage {st['max_drainage_km2']} km2", flush=True)
        # verify all six read at the centre before marking done
        ok = True
        for n in REQ:
            p = d / f"{n}.tif"
            if not p.exists():
                ok = False
                break
            try:
                with rasterio.open(p) as s:
                    cy, cx = s.height // 2, s.width // 2
                    hh = min(256, s.height // 2); ww = min(256, s.width // 2)
                    _ = s.read(1, window=((cy - hh, cy + hh), (cx - ww, cx + ww)))
            except Exception as e:
                print(f"    {n}.tif unreadable: {type(e).__name__}")
                ok = False
                break
        if ok:
            with rasterio.open(d / "facc.tif") as s:
                w, h = s.width, s.height
            state[d.name] = {"done": True, "reconstructed": True,
                             "cells": w * h}

    sp = ROOT / "data/basin_routing_state.json"
    sp.write_text(json.dumps(state, indent=1))
    print(f"\n{len(state)} basins complete and verified -> basin_routing_state.json")
    print(f"remaining to route: {11 - len(state)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
