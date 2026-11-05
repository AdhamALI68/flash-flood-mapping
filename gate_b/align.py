#!/usr/bin/env python3
"""
Read two rasters onto their common pixel grid.

HyP3 products from different date pairs do NOT share an origin - the burst
footprint shifts slightly between acquisitions. For event 983830 the baseline
and co-event coherence rasters were offset by 2500 m in x and 580 m in y.
Cropping both to [:rows, :cols] from their own corners therefore differences
pixels 2.5 km apart, which silently corrupts any before/after comparison.

Both products share a CRS and a 20 m pixel size, and both origins fall on the
same 20 m lattice, so the grids are compatible - they only need to be windowed
to their intersection.
"""

from __future__ import annotations

import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds


def common_window(paths: list[str], aoi: tuple[float, ...] | None = None
                  ) -> tuple[tuple[float, ...], list[Window]]:
    """Intersection bounds plus the read-window into each raster.

    aoi, if given, further restricts to a bounding box in the rasters' CRS -
    used to keep only the neighbourhood of an event instead of the whole burst.
    """
    srcs = [rasterio.open(p) for p in paths]
    try:
        crs = {s.crs for s in srcs}
        if len(crs) != 1:
            raise SystemExit(f"rasters disagree on CRS: {crs}")
        res = {s.res for s in srcs}
        if len(res) != 1:
            raise SystemExit(f"rasters disagree on pixel size: {res}")

        left = max(s.bounds.left for s in srcs)
        bottom = max(s.bounds.bottom for s in srcs)
        right = min(s.bounds.right for s in srcs)
        top = min(s.bounds.top for s in srcs)
        if aoi is not None:
            left, bottom = max(left, aoi[0]), max(bottom, aoi[1])
            right, top = min(right, aoi[2]), min(top, aoi[3])
        if right <= left or top <= bottom:
            raise SystemExit("rasters do not overlap")

        bounds = (left, bottom, right, top)
        wins = []
        for s in srcs:
            w = from_bounds(*bounds, transform=s.transform)
            # Snap to whole pixels; the origins share a 20 m lattice so this is
            # exact rather than a resampling.
            wins.append(Window(round(w.col_off), round(w.row_off),
                               round(w.width), round(w.height)))
        return bounds, wins
    finally:
        for s in srcs:
            s.close()


def read_aligned(paths: list[str], nodata_to_nan: bool = True,
                 aoi: tuple[float, ...] | None = None):
    """Read every raster on the shared grid. Returns (arrays, transform, crs)."""
    bounds, wins = common_window(paths, aoi=aoi)
    arrays, transform, crs = [], None, None
    shapes = set()
    for p, w in zip(paths, wins):
        with rasterio.open(p) as s:
            a = s.read(1, window=w).astype("float32")
            if transform is None:
                transform = s.window_transform(w)
                crs = s.crs
            shapes.add(a.shape)
            if nodata_to_nan:
                a[a <= 0] = np.nan
                if s.nodata is not None:
                    a[a == s.nodata] = np.nan
            arrays.append(a)
    if len(shapes) != 1:
        # Rounding at the edges can differ by one pixel; trim to the minimum.
        h = min(s[0] for s in shapes)
        wd = min(s[1] for s in shapes)
        arrays = [a[:h, :wd] for a in arrays]
    return arrays, transform, crs
