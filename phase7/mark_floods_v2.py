#!/usr/bin/env python3
"""
Flood-trace labels, version 2: the same idea as mark_floods.py (fresh pale
ground between a clear photo before and after the flood) plus rules aimed at
the three things that fooled version 1 on the 10-flood pilot.

  light   : version 1 only shifted each photo by its median, and read the 8-bit
            display image, which saturates on bright desert. Now: surface
            reflectance (blue, green, red), the after photo matched to the
            before photo with a robust straight-line fit, and thresholds set
            per flood from the noise of unchanged ground (4 and 2 sigma).
  clouds  : cloud and shadow edges were painted red. Now: also drop a 200 m
            ring around every cloud and shadow, and the cloud-mask classes
            for dark, unclassified and snow pixels.
  farms   : plowed or watered fields change between photos. Now: drop places
            whose greenness (NDVI) jumped, and places that also changed
            between two clear photos that were both taken BEFORE the flood.

  safety  : skip the flood if more than 40% of the square is marked, if less
            than 0.5% is marked, or if the radar (a different satellite, never
            a model input) shows no extra ground change inside the marks
            during the flood week compared with before it.

Still only photos and radar. No terrain, basin or land-cover layer, because the
model gets those as inputs and a label built from them would leak.

    ./.venv/bin/python -u phase7/mark_floods_v2.py --from data/trace_labels/pilot_10.json

Output: data/trace_labels_v2/<event>.tif + .json, data/trace_labels_v2/pilot_10.json
        handoff/hand_label_pilot/page/e<event>_{before,after,marks}_v2.jpg
"""
from __future__ import annotations
import argparse, json, sys, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image
from rasterio.transform import from_origin
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mark_floods import search, read, drop_small  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
LAB = ROOT / "data/trace_labels_v2"
PAGE = ROOT / "handoff/hand_label_pilot/page"

SCL_BAD = [0, 1, 2, 3, 7, 8, 9, 10, 11]
SCL_CLOUDY = [3, 8, 9, 10]
RING_PX = 20                       # 200 m at 10 m
MIN_CLEAR = 0.90
PRE_DAYS, POST_DAYS, MAX_TRIES = 45, 30, 8
CTRL_MIN_GAP, CTRL_MAX_GAP = 10, 60
STRONG_Z, WEAK_Z = 4.0, 2.0
NDVI_UP, NDVI_DOWN = 0.05, -0.15
MAX_MARKED, MIN_MARKED = 40.0, 0.5
RADAR_MIN = 0.02


def reflect(asset, grid):
    band = (asset.get("raster:bands") or [{}])[0]
    dn = read(asset["href"], grid, [1])[0]
    return dn * band.get("scale", 1e-4) + band.get("offset", 0.0), dn > 0


def photo(f, grid):
    try:
        scl = read(f["assets"]["scl"]["href"], grid, [1], nearest=True)[0].astype(int)
    except Exception:
        return None
    cloudy = np.isin(scl, SCL_CLOUDY)
    ring = ndimage.distance_transform_edt(~cloudy) <= RING_PX if cloudy.any() else cloudy
    clear = ~np.isin(scl, SCL_BAD) & ~ring & (scl > 0)
    if clear.mean() < MIN_CLEAR:
        return None
    try:
        bands = []
        for key in ("blue", "green", "red", "nir"):
            v, ok = reflect(f["assets"][key], grid)
            bands.append(v)
            clear &= ok
    except Exception:
        return None
    if clear.mean() < MIN_CLEAR:
        return None
    return {"feat": f, "date": pd.Timestamp(f["properties"]["datetime"][:10]),
            "refl": np.stack(bands).astype("float32"), "clear": clear}


def first_clear(feats, grid):
    for f in feats[:MAX_TRIES]:
        p = photo(f, grid)
        if p is not None:
            return p
    return None


def bright(p):
    return p["refl"][:3].mean(0)


def ndvi(p):
    nir, red = p["refl"][3], p["refl"][2]
    return (nir - red) / np.maximum(nir + red, 1e-4)


def matched_change(a, b, valid, rng):
    """b minus a, after matching b's brightness to a on ground that did not change."""
    xa, xb = bright(a), bright(b)
    idx = np.flatnonzero(valid.ravel())
    idx = rng.choice(idx, size=min(60000, idx.size), replace=False)
    x, y = xa.ravel()[idx], xb.ravel()[idx]
    keep = np.ones(x.size, bool)
    for _ in range(4):
        s, c = np.polyfit(x[keep], y[keep], 1)
        res = y - (s * x + c)
        mad = np.median(np.abs(res[keep] - np.median(res[keep])))
        keep = np.abs(res - np.median(res[keep])) < 2.5 * 1.4826 * mad
    d = (xb - c) / s - xa
    d = ndimage.gaussian_filter(np.where(valid, d, 0.0), 1.0)
    dv = d[valid]
    sigma = 1.4826 * np.median(np.abs(dv - np.median(dv)))
    return d - np.median(dv), float(sigma), float(s), float(c)


def grow_marks(d, sigma, valid):
    strict = drop_small(ndimage.binary_opening((d > STRONG_Z * sigma) & valid), 50)
    weak = (d > WEAK_Z * sigma) & valid
    lab, _ = ndimage.label(weak)
    grown = (np.isin(lab, np.unique(lab[strict & weak])) & weak) | strict
    grown = ndimage.binary_closing(grown, structure=np.ones((3, 3)), iterations=2)
    holes = ndimage.binary_fill_holes(grown) & ~grown
    grown |= drop_small(holes, 1) & ~drop_small(holes, 150)
    return drop_small(grown & valid, 50)


def blobs(mask, min_px, grow_px):
    m = drop_small(mask, min_px)
    return ndimage.binary_dilation(m, iterations=grow_px) if m.any() else m


def process(eid: str) -> dict:
    rng = np.random.default_rng(int(eid))
    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    r = ev[ev.EVENT_ID.astype(int) == int(eid)].iloc[0]
    out = {"event": eid, "place": f"{str(r.CZ_NAME).title()}, {str(r.STATE).title()}",
           "region": str(r.huc4), "flood_date": str(r.BEGIN_UTC.date()), "damage": str(r.DAMAGE_PROPERTY)}

    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
        b, crs, tr20, shp20 = s.bounds, s.crs, s.transform, (s.height, s.width)
        coh_base, coh_ev = s.read(2), s.read(3)
    with rasterio.open(ROOT / f"data/features/{eid}/valid.tif") as s:
        rvalid = s.read(1) > 0
    grid = {"crs": crs, "shape": (shp20[0] * 2, shp20[1] * 2),
            "transform": from_origin(b.left, b.top, 10, 10)}
    t, lon, lat = r.BEGIN_UTC, r.BEGIN_LON, r.BEGIN_LAT

    before = sorted(search(lon, lat, (t - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d"),
                           (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
                    key=lambda x: x["properties"]["datetime"], reverse=True)
    pre = first_clear(before, grid)
    after = sorted([x for x in search(lon, lat, t.strftime("%Y-%m-%d"),
                                      (t + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"))
                    if pd.Timestamp(x["properties"]["datetime"]).tz_localize(None) > t],
                   key=lambda x: x["properties"]["datetime"])
    post = first_clear(after, grid) if pre is not None else None
    if pre is None or post is None:
        out["skip"] = "no clear photo " + ("before" if pre is None else "after") + " the flood"
        return out
    ctrl_feats = sorted(search(lon, lat, (pre["date"] - pd.Timedelta(days=CTRL_MAX_GAP)).strftime("%Y-%m-%d"),
                               (pre["date"] - pd.Timedelta(days=CTRL_MIN_GAP)).strftime("%Y-%m-%d")),
                        key=lambda x: x["properties"]["datetime"], reverse=True)
    ctrl = first_clear(ctrl_feats, grid)

    valid = pre["clear"] & post["clear"]
    d, sigma, slope, icpt = matched_change(pre, post, valid, rng)
    first = grow_marks(d, sigma, valid)

    dn = ndimage.gaussian_filter(ndvi(post) - ndvi(pre), 1.0)
    plants = blobs(((dn > NDVI_UP) | (dn < NDVI_DOWN)) & valid, 30, 2)
    unstable = np.zeros_like(valid)
    if ctrl is not None:
        vc = ctrl["clear"] & pre["clear"]
        dc, sc, _, _ = matched_change(ctrl, pre, vc, rng)
        dnc = ndimage.gaussian_filter(ndvi(pre) - ndvi(ctrl), 1.0)
        unstable = blobs(((np.abs(dc) > 3 * sc) | (np.abs(dnc) > 0.10)) & vc, 100, 2)

    mark = drop_small(first & ~plants & ~unstable, 50)
    pct = 100 * float(mark[valid].mean())
    out.update({"before_date": str(pre["date"].date()), "after_date": str(post["date"].date()),
                "control_date": str(ctrl["date"].date()) if ctrl is not None else None,
                "days_before": int((t.normalize() - pre["date"]).days),
                "days_after": int((post["date"] - t.normalize()).days),
                "clear_pct": round(100 * float(valid.mean())),
                "noise_sigma": round(sigma, 4), "light_fit": [round(slope, 3), round(icpt, 4)],
                "marked_first_pct": round(100 * float(first[valid].mean()), 1),
                "removed_plants_pct": round(100 * float((first & plants)[valid].mean()), 1),
                "removed_unstable_pct": round(100 * float((first & unstable & ~plants)[valid].mean()), 1),
                "marked_pct": round(pct, 1)})

    m20 = mark.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)) >= 0.5
    v20 = valid.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)) >= 0.5
    rv = rvalid & np.isfinite(coh_ev) & np.isfinite(coh_base) & v20
    if (m20 & rv).sum() >= 200 and (~m20 & rv).sum() >= 200:
        during = float((1 - coh_ev[m20 & rv]).mean() - (1 - coh_ev[~m20 & rv]).mean())
        beforef = float((1 - coh_base[m20 & rv]).mean() - (1 - coh_base[~m20 & rv]).mean())
        out["radar_extra_change"] = round(during - beforef, 3)
    else:
        out["radar_extra_change"] = None

    if pct > MAX_MARKED:
        out["skip"] = f"too much marked ({pct:.0f}%), the photos differ for another reason"
    elif pct < MIN_MARKED:
        out["skip"] = "no visible flood marks"
    elif out["radar_extra_change"] is not None and out["radar_extra_change"] <= RADAR_MIN:
        out["skip"] = "the radar does not see extra change inside the marks in the flood week"

    if "skip" not in out:
        LAB.mkdir(parents=True, exist_ok=True)
        d20 = d.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)).astype("float32")
        prof = dict(driver="GTiff", height=shp20[0], width=shp20[1], count=3, dtype="float32",
                    crs=crs, transform=tr20, compress="deflate")
        with rasterio.open(LAB / f"{eid}.tif", "w", **prof) as o:
            o.write(m20.astype("float32"), 1); o.set_band_description(1, "trace")
            o.write(v20.astype("float32"), 2); o.set_band_description(2, "valid")
            o.write(d20, 3); o.set_band_description(3, "change")

    vis = {}
    for name, p in (("before", pre), ("after", post)):
        vis[name] = np.moveaxis(read(p["feat"]["assets"]["visual"]["href"], grid, [1, 2, 3]), 0, -1)
    lo, hi = np.percentile(np.stack([vis["before"], vis["after"]])[:, valid], [1, 99])
    pic = {k: (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype("uint8") for k, v in vis.items()}
    over = pic["after"].copy()
    removed = first & ~mark
    over[removed] = (0.35 * over[removed] + 0.65 * np.array([60, 110, 200], "float32")).astype("uint8")
    over[mark] = (0.25 * over[mark] + 0.75 * np.array([235, 40, 20], "float32")).astype("uint8")
    for name, a in (("before", pic["before"]), ("after", pic["after"]), ("marks", over)):
        Image.fromarray(a).resize((800, 800), Image.BILINEAR).save(PAGE / f"e{eid}_{name}_v2.jpg", quality=85)
    return out


def safe(eid):
    try:
        return process(eid)
    except Exception as e:
        return {"event": eid, "skip": f"error: {e}", "trace": traceback.format_exc()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src", default="data/trace_labels/pilot_10.json")
    ap.add_argument("--workers", type=int, default=5)
    args = ap.parse_args()
    events = [str(x["event"]) for x in json.load(open(ROOT / args.src))]
    LAB.mkdir(parents=True, exist_ok=True)
    results = []
    with ProcessPoolExecutor(args.workers) as pool:
        for res in pool.map(safe, events):
            results.append(res)
            (LAB / f"{res['event']}.json").write_text(json.dumps(res, indent=1))
            print(json.dumps({k: v for k, v in res.items() if k != "trace"}), flush=True)
    json.dump(results, open(LAB / "pilot_10.json", "w"), indent=1)
    kept = [x for x in results if "skip" not in x]
    print(f"\nkept {len(kept)} of {len(results)} floods")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
