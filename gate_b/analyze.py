#!/usr/bin/env python3
"""
Gate B, step 3 - does coherence actually see the flood?

Downloads both finished products and compares them. The test is not "is
co-event coherence low" - coherence is low everywhere in a 6-day desert pair.
The test is whether it dropped MORE over the flood interval than over a matched
interval with no flood, and whether that extra drop lands near where flooding
was reported rather than scattered at random.

    drop = baseline_coherence - coevent_coherence

Positive drop = the surface changed more during the flood window.

Two controls, because a headline number alone would not be convincing:
  1. the drop distribution over the whole burst (most of it did not flood)
  2. the drop near the reported flood point vs far from it

Output: gate_b/<event>_gateb_result.json + a PNG
"""

from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

import numpy as np
import rasterio
from rasterio.warp import transform as warp_transform


def fetch(h, job_id: str, dest: Path) -> Path:
    """Download and unzip one finished HyP3 product."""
    dest.mkdir(parents=True, exist_ok=True)
    existing = [d for d in dest.iterdir() if d.is_dir()]
    if existing:
        return existing[0]
    job = h.get_job_by_id(job_id)
    if job.status_code != "SUCCEEDED":
        raise SystemExit(f"job {job_id} is {job.status_code}, not SUCCEEDED")
    zips = job.download_files(dest)
    for z in zips:
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest)
        z.unlink()
    return next(d for d in dest.iterdir() if d.is_dir())


def coh_path(product_dir: Path) -> str:
    """The correlation raster HyP3 ships with every InSAR product."""
    cands = sorted(product_dir.glob("*corr*.tif")) or sorted(product_dir.glob("*coh*.tif"))
    if not cands:
        raise SystemExit(f"no coherence raster in {product_dir}: "
                         f"{[p.name for p in product_dir.glob('*.tif')]}")
    return str(cands[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True)
    ap.add_argument("--bursts", required=True)
    ap.add_argument("--near-km", type=float, default=5.0,
                    help="radius around the reported flood point counted as 'near'")
    args = ap.parse_args()

    import hyp3_sdk

    root = Path(__file__).resolve().parent
    spec = json.loads(Path(args.jobs).read_text())
    binfo = json.loads(Path(args.bursts).read_text())
    ev = spec["event_id"]

    h = hyp3_sdk.HyP3()
    dirs = {}
    for tag, j in spec["jobs"].items():
        print(f"fetching {tag}...", flush=True)
        dirs[tag] = fetch(h, j["job_id"], root / f"{ev}_{tag}")
        print(f"  {dirs[tag].name}")

    # The two products do NOT share an origin - see align.py. Differencing them
    # on their raw arrays would subtract pixels kilometres apart.
    from align import read_aligned

    paths = [coh_path(dirs["baseline"]), coh_path(dirs["coevent"])]
    (base, coev), transform, crs = read_aligned(paths)
    print(f"\nbaseline raster: {Path(paths[0]).name}")
    print(f"coevent  raster: {Path(paths[1]).name}")
    print(f"aligned onto common grid: {base.shape}")

    valid = np.isfinite(base) & np.isfinite(coev)
    drop = np.where(valid, base - coev, np.nan)

    # Distance of every pixel from the reported flood point, on the shared grid.
    rows, cols = np.indices(base.shape)
    xs, ys = rasterio.transform.xy(transform, rows, cols)
    xs = np.array(xs).reshape(base.shape)
    ys = np.array(ys).reshape(base.shape)
    ex, ey = warp_transform("EPSG:4326", crs, [binfo["lon"]], [binfo["lat"]])
    dist_km = np.sqrt((xs - ex[0]) ** 2 + (ys - ey[0]) ** 2) / 1000.0

    near = valid & (dist_km <= args.near_km)
    far = valid & (dist_km > args.near_km)

    def stats(m):
        return {
            "n": int(m.sum()),
            "baseline_coh": round(float(np.nanmean(base[m])), 4),
            "coevent_coh": round(float(np.nanmean(coev[m])), 4),
            "mean_drop": round(float(np.nanmean(drop[m])), 4),
            "frac_dropped_gt_0.1": round(float(np.nanmean(drop[m] > 0.1)), 4),
        }

    res = {
        "event_id": ev, "basin": binfo["basin"], "begin_utc": binfo["begin_utc"],
        "burst": binfo["full_burst_id"], "looks": spec["looks"],
        "all": stats(valid), "near": stats(near), "far": stats(far),
        "near_km": args.near_km,
    }

    print("\n" + "=" * 66)
    print(f"GATE B - event {ev}, {binfo['basin']}")
    print("=" * 66)
    for label, key in (("whole burst", "all"),
                       (f"within {args.near_km:g} km of report", "near"),
                       (f"beyond {args.near_km:g} km", "far")):
        s = res[key]
        print(f"{label:<28} n={s['n']:>9,}  baseline={s['baseline_coh']:.3f}  "
              f"coevent={s['coevent_coh']:.3f}  drop={s['mean_drop']:+.4f}")

    contrast = res["near"]["mean_drop"] - res["far"]["mean_drop"]
    res["near_minus_far"] = round(float(contrast), 4)
    print(f"\nnear minus far: {contrast:+.4f}")
    print("  > 0 means the surface changed MORE near the reported flood,")
    print("  which is what a real flood signal looks like.")

    out = root / f"{ev}_gateb_result.json"
    out.write_text(json.dumps(res, indent=2))
    print(f"\nwrote {out.name}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        for ax, img, t in ((axes[0], base, "baseline coherence (no flood)"),
                           (axes[1], coev, "co-event coherence (spans flood)")):
            im = ax.imshow(img, vmin=0, vmax=1, cmap="viridis")
            ax.set_title(t); ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046)
        im = axes[2].imshow(drop, vmin=-0.4, vmax=0.4, cmap="RdBu_r")
        axes[2].set_title("drop (baseline - coevent)\nred = changed more during flood")
        axes[2].axis("off")
        plt.colorbar(im, ax=axes[2], fraction=0.046)
        fig.suptitle(f"Gate B - event {ev}, {binfo['basin']}, {binfo['begin_utc']}")
        fig.tight_layout()
        png = root / f"{ev}_gateb.png"
        fig.savefig(png, dpi=130)
        print(f"wrote {png.name}")
    except ImportError:
        print("(matplotlib not installed, skipped the figure)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
