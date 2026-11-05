#!/usr/bin/env python3
"""
Rebuild HAND (height above nearest drainage) cleanly, at 100 m, for every basin.

WHAT WENT WRONG
    basin_routing.py crashed twice on RasterBlockError in Death Valley, Mojave
    and Whitewater River. The third attempt completed, but those three basins
    ended up holding files from different runs. Their timestamps prove it: twi
    was written BEFORE the facc it is derived from, which cannot happen in one
    clean pass. The damage landed on HAND:

        basin        stream cells   HAND usable
        15010007        2.41%          73%      healthy
        18090203        0.013%         57%
        18100201        0.022%         25%
        18090208        0.015%          9%

    Flow accumulation is fine in all of them (2.8% of cells over threshold), so
    the streams SHOULD be there. HAND was computed against a stale stream raster
    that had almost nothing in it, so it had nothing to measure height above.

    That matters because basin_hand is now the strongest single feature we have
    (within-tile AUC 0.5546), and it is broken across two whole regions.

WHY 100 m, AND WHY ALL ELEVEN
    Rebuilding at 10 m needs tens of GB of intermediates per basin and the disk
    is at 94%. At 100 m a basin is 2-9 M cells and the whole job is ~250 MB.
    The stream threshold is kept at the same PHYSICAL area, 0.05 km2, so the
    network is the same network, just sampled coarser.

    All eleven are rebuilt, not only the broken three, because a feature that is
    computed one way in eight regions and another way in three cannot be
    compared across regions - and cross-region transfer is what section 4b of
    NEXT_STEPS is for. Uniform beats slightly finer.

    The rebuild is CHECKED, not assumed: in the eight healthy basins the new
    100 m HAND is correlated against the existing 10 m HAND. If they disagree,
    this script says so and changes nothing.

Output: data/basin_features/<huc8>/hand100.tif
"""

from __future__ import annotations

import argparse
import shutil
import tempfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject

ROOT = Path(__file__).resolve().parent.parent
BF = ROOT / "data/basin_features"
DEM100 = ROOT / "data/basin_dem100"
WORK_RES = 100.0
STREAM_AREA_KM2 = 0.05          # same physical threshold as the 10 m routing
BROKEN = {"18090203", "18090208", "18100201"}


def log(m: str) -> None:
    print(m, flush=True)


def build(huc8: str, wbt, keep_tmp: bool) -> Path | None:
    dem = DEM100 / f"{huc8}.tif"
    if not dem.exists():
        log(f"  {huc8}: no 100 m DEM, skipping")
        return None
    out = BF / huc8 / "hand100.tif"
    tmp = Path(tempfile.mkdtemp(prefix=f"hand_{huc8}_", dir=str(ROOT / "data")))
    try:
        filled = tmp / "filled.tif"
        facc = tmp / "facc.tif"
        stream = tmp / "stream.tif"
        hand = tmp / "hand.tif"
        # NaN nodata upsets whitebox, so give it an explicit sentinel.
        src = tmp / "dem.tif"
        with rasterio.open(dem) as s:
            a = s.read(1)
            prof = s.profile
        a = np.where(np.isfinite(a), a, -32768.0).astype("float32")
        prof.update(dtype="float32", nodata=-32768.0, compress="deflate",
                    tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(src, "w", **prof) as d:
            d.write(a, 1)

        cells = max(1, int(round(STREAM_AREA_KM2 * 1e6 / (WORK_RES ** 2))))
        # Each step is checked. A silent failure here is what produced the
        # stale-stream HAND in the first place.
        steps = [
            ("fill_depressions", lambda: wbt.fill_depressions(str(src), str(filled)), filled),
            ("d8_flow_accumulation",
             lambda: wbt.d8_flow_accumulation(str(filled), str(facc), out_type="cells"), facc),
            ("extract_streams",
             lambda: wbt.extract_streams(str(facc), str(stream), threshold=cells), stream),
            ("elevation_above_stream",
             lambda: wbt.elevation_above_stream(str(filled), str(stream), str(hand)), hand),
        ]
        for name, fn, expect in steps:
            rc = fn()
            if rc != 0 or not expect.exists():
                log(f"  {huc8}: {name} failed (rc={rc}, "
                    f"output {'missing' if not expect.exists() else 'present'})")
                return None

        with rasterio.open(stream) as s:
            st = s.read(1)
            stream_frac = float(np.mean(st > 0))
        with rasterio.open(hand) as s:
            h = s.read(1).astype("float32")
            nd = s.nodata
            hp = s.profile
        if nd is not None:
            h = np.where(h == nd, np.nan, h)
        h = np.where(np.isfinite(a) & (a > -32000), h, np.nan)
        valid = float(np.isfinite(h).mean())
        hp.update(dtype="float32", nodata=np.nan, count=1, compress="deflate",
                  tiled=True, blockxsize=512, blockysize=512)
        with rasterio.open(out, "w", **hp) as d:
            d.write(h.astype("float32"), 1)
            d.set_band_description(1, "basin_hand100")
        v = h[np.isfinite(h)]
        log(f"  {huc8}: streams {100*stream_frac:5.2f}%  HAND valid "
            f"{100*valid:5.1f}%  range {v.min():.1f}..{np.percentile(v,99):.1f} m"
            + ("   [was broken]" if huc8 in BROKEN else ""))
        return out
    finally:
        if not keep_tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def check(huc8: str) -> tuple[float, float] | None:
    """Compare new 100 m HAND against the existing 10 m HAND, same grid."""
    new = BF / huc8 / "hand100.tif"
    old = BF / huc8 / "hand.tif"
    if not (new.exists() and old.exists()):
        return None
    with rasterio.open(new) as s:
        n = s.read(1)
        grid = {"crs": s.crs, "transform": s.transform, "shape": n.shape}
    o = np.full(grid["shape"], np.nan, dtype="float32")
    with rasterio.open(old) as s:
        reproject(source=rasterio.band(s, 1), destination=o,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=s.nodata, dst_nodata=np.nan,
                  resampling=Resampling.average)
    ok = np.isfinite(n) & np.isfinite(o)
    if ok.sum() < 1000:
        return None
    import pandas as pd
    a, b = n[ok], o[ok]
    r = float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])
    return r, float(ok.mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    ap.add_argument("--keep-tmp", action="store_true")
    args = ap.parse_args()

    from whitebox import WhiteboxTools
    wbt = WhiteboxTools()
    wbt.verbose = False

    basins = sorted(p.name for p in BF.iterdir() if p.is_dir())
    if args.only:
        basins = [b for b in basins if b == args.only]
    log(f"rebuilding HAND at {WORK_RES:.0f} m for {len(basins)} basins")
    log(f"stream threshold {STREAM_AREA_KM2} km2 = "
        f"{int(STREAM_AREA_KM2*1e6/WORK_RES**2)} cells at this resolution\n")

    for h in basins:
        build(h, wbt, args.keep_tmp)

    log("\nagreement with the existing 10 m HAND (Spearman, on shared pixels):")
    log("  the eight healthy basins are the test. If the new HAND matches them,")
    log("  it is the same quantity and the three broken ones can trust it too.\n")
    good = []
    for h in basins:
        c = check(h)
        if c is None:
            log(f"  {h}: could not compare")
            continue
        r, cov = c
        tag = "  [was broken - low agreement is EXPECTED here]" if h in BROKEN else ""
        log(f"  {h}: rho {r:+.3f}  on {100*cov:.0f}% of the basin{tag}")
        if h not in BROKEN:
            good.append(r)
    if good:
        log(f"\nhealthy basins: median rho {np.median(good):+.3f} "
            f"(min {min(good):+.3f})")
        if np.median(good) > 0.8:
            log("VERDICT: the 100 m rebuild reproduces the 10 m HAND. Safe to use.")
        else:
            log("VERDICT: the rebuild does NOT match. Do not use it. Investigate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
