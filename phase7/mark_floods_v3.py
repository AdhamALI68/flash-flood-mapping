#!/usr/bin/env python3
"""
Flood-trace labels, version 3. One script for every stage of the plan:
20 new floods, 40 new floods, then every flood that has training tiles.

The goal is a label the model can reach high recall AND high precision on:
no gaps inside a flood path, no noise dots, and floods skipped rather than
kept when their marks are not clean.

Photos (what version 2 got wrong on the 10-flood pilot, and the fix)
  P1  all photos of a flood from the same satellite track and the same tile
      grid, one processing version per date (mixed versions shifted colours,
      a track seam was marked as flood)
  P2  cloud-mask classes for cloud, shadow, cirrus, dark, unclassified and
      snow are dropped, plus a 200 m ring around every real cloud or shadow
      (a group of at least 1 ha). Stage s20 showed the cloud mask flags
      scattered single-cell "clouds" on bright desert sand; a ring around
      each of those threw away a quarter of near cloud-free photos. A photo
      needs 80% clear ground. "Unclassified" counts as clear: photos made
      with the 2017-2018 cloud-mask software call about a fifth of bright
      desert unclassified (newer photos under 0.3%), which made clear
      2017 photos fail. Haze left in is caught by C1, R3, S3 and S4.
      The ring goes only around high-probability cloud, shadow and cirrus:
      the 2017-2018 software also calls about 6% of bright sand "medium
      probability cloud", and a ring around those dropped clear photos from
      94% to 79% clear. Those pixels are still removed, just without a ring.
      A photo with big clouds over more than 5% of the square is not used,
      because haze reaches past the ring (stage s20, flood 1207877).
  P3  a third clear photo, 10-60 days before the "before" photo, as a
      no-flood control

Change
  C0  each band is scaled to reflectance and then has its own darkest value
      (0.5th percentile of clear ground) subtracted. The catalogue's offset
      is ignored: for the 05.00 copies it says -0.1, but their pixel values
      match the 03.01 copies of the same date (red 2558 vs 2561), so applying
      it made the ground 0.1 too dark and skewed every greenness value.
  C1  surface reflectance (blue, green, red), after photo matched to the
      before photo with a robust straight line on unchanged ground
  C2  noise level measured on the no-flood control pair, so a big flood does
      not raise its own threshold (version 2 lost half of Painted Rock this way)
  C3  seeds above 4 noise levels grown into joined change above 2 levels,
      gaps closed, holes under 1.5 ha filled, specks under 0.5 ha removed

Everything runs on the model's own 20 m grid, read from the photos' built-in
half-resolution copies: the link to the photo store was 50 KB/s to 1 MB/s,
and one flood at 10 m took over 15 minutes.

Removed (never terrain, basin or land cover: the model gets those as inputs)
  R1  ground that got greener (NDVI up by more than 0.05)
  R2  ground that was green in any of the three photos (NDVI above 0.30,
      crops and watered land); version 2 also removed ground that got less
      green, which deleted real flood scour where bushes were torn out
  R3  ground that also changed between the two no-flood photos

Label (option B, gaps filled)
  L1  gaps closed, holes under 1 ha filled, pieces under 0.5 ha removed

Skip the flood when
  S1  no clear photo before, after, or for the control
  S2  marked area above 40% or below 0.5% of the clear ground
  S3  more than 30% of the marked area sits in pieces under 2 ha (scattered
      dots, not flood paths)
  S4  radar cannot check it, or the radar shows no extra ground change inside
      the marks in the flood week (at least +0.02 coherence loss beyond the
      same difference before the flood)

    ./.venv/bin/python -u phase7/mark_floods_v3.py --stage s20 --n 20 --seed 20
    ./.venv/bin/python -u phase7/mark_floods_v3.py --stage s40 --n 40 --seed 40
    ./.venv/bin/python -u phase7/mark_floods_v3.py --stage all --n 0 --images kept

Every flood that gets its photos also saves its arrays (change, masks, colour
previews, light fit) to data/trace_v3/<stage>/arrays/<event>.npz, so rules
that act on the marks can be tuned by phase7/apply_rules.py without
downloading the photos again.

Output: data/trace_v3/<stage>/<event>.json (+ .tif when kept), progress.json,
        summary.json, arrays/<event>.npz;
        handoff/hand_label_pilot/page/<stage>/e<event>.jpg
"""
from __future__ import annotations
import argparse, json, os, re, sys, time, traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif,.TIF")
os.environ.setdefault("GDAL_HTTP_MERGE_CONSECUTIVE_RANGES", "YES")
os.environ.setdefault("GDAL_HTTP_MULTIPLEX", "YES")
os.environ.setdefault("VSI_CACHE", "TRUE")

import numpy as np
import pandas as pd
import rasterio
import requests
from PIL import Image
from rasterio.transform import from_origin
from rasterio.warp import Resampling, reproject
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
STAC = "https://earth-search.aws.element84.com/v1/search"
OUTROOT = ROOT / "data/trace_v3"
PAGE = ROOT / "handoff/hand_label_pilot/page"

SCL_BAD = [0, 1, 2, 3, 8, 9, 10, 11]
SCL_CLOUDY = [3, 9, 10]
MAX_BIG_CLOUD = 0.05
RING_PX = 10
MIN_CLEAR = 0.80
CLOUD_MIN_CELLS = 25
PRE_DAYS, POST_DAYS, MAX_TRIES = 45, 30, 8
CTRL_MIN_GAP, CTRL_MAX_GAP = 10, 60
STRONG_Z, WEAK_Z = 4.0, 2.0
GREENER, GREEN = 0.05, 0.30
MAX_MARKED, MIN_MARKED = 40.0, 0.5
DOT_CELLS, MAX_DOT_SHARE = 50, 0.30
RADAR_MIN = 0.02


# ---------------------------------------------------------------- photos ----
def search(lon, lat, start, end):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 200,
            "query": {"eo:cloud_cover": {"lt": 60}}}
    for k in range(4):
        try:
            r = requests.post(STAC, json=body, timeout=120)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception:
            time.sleep(2 + 3 * k)
    return []


def track(f):
    m = re.search(r"_R(\d{3})_", f["properties"].get("s2:product_uri", "") or "")
    return m.group(1) if m else None


def free(f):
    """Readable without a paying AWS account: a public cloud-optimised copy."""
    return all(f["assets"].get(k, {}).get("href", "").startswith("https://")
               for k in ("scl", "blue", "green", "red", "nir", "visual"))


def dedupe(feats):
    """One item per date and tile: a free copy first, then the newest processing
    version. Stage s20 showed some dates also have a copy in the paid bucket,
    and preferring it by version made readable floods look photo-less."""
    best = {}
    for f in feats:
        if not free(f):
            continue
        p = f["properties"]
        key = (p["datetime"][:10], p.get("grid:code"))
        v = str(p.get("s2:processing_baseline", "0"))
        if key not in best or v > str(best[key]["properties"].get("s2:processing_baseline", "0")):
            best[key] = f
    return list(best.values())


def read(href, grid, band=1, nearest=False, half=True):
    buf = np.zeros(grid["shape"], dtype="float32")
    opts = {"OVERVIEW_LEVEL": 0} if half else {}
    with rasterio.open(href, **opts) as s:
        reproject(source=rasterio.band(s, band), destination=buf,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  resampling=Resampling.nearest if nearest else Resampling.average)
    return buf


def photo(f, grid):
    try:
        scl = read(f["assets"]["scl"]["href"], grid, nearest=True, half=False).astype(int)
    except Exception:
        return None
    cloudy = drop_small(np.isin(scl, SCL_CLOUDY), CLOUD_MIN_CELLS)
    if cloudy.mean() > MAX_BIG_CLOUD:
        return None
    ring = ndimage.distance_transform_edt(~cloudy) <= RING_PX if cloudy.any() else cloudy
    clear = ~np.isin(scl, SCL_BAD) & ~ring & (scl > 0)
    if clear.mean() < MIN_CLEAR:
        return None
    bands = []
    try:
        for key in ("blue", "green", "red", "nir"):
            a = f["assets"][key]
            rb = (a.get("raster:bands") or [{}])[0]
            dn = read(a["href"], grid)
            clear &= dn > 0
            bands.append(dn * rb.get("scale", 1e-4))
    except Exception:
        return None
    if clear.mean() < MIN_CLEAR:
        return None
    bands = [np.maximum(b - np.percentile(b[clear], 0.5), 0.0) for b in bands]
    return {"feat": f, "date": pd.Timestamp(f["properties"]["datetime"][:10]),
            "refl": np.stack(bands).astype("float32"), "clear": clear}


def first_clear(feats, grid, want=None):
    tried = 0
    for f in feats:
        if want and (f["properties"].get("grid:code"), track(f)) != want:
            continue
        if tried >= MAX_TRIES:
            break
        tried += 1
        p = photo(f, grid)
        if p is not None:
            return p
    return None


# ---------------------------------------------------------------- change ----
def drop_small(mask, min_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    size = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(size >= min_px))


def fill_small_holes(mask, max_px):
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return mask | (drop_small(holes, 1) & ~drop_small(holes, max_px + 1))


def bright(p):
    return p["refl"][:3].mean(0)


def ndvi(p):
    nir, red = p["refl"][3], p["refl"][2]
    return (nir - red) / np.maximum(nir + red, 1e-4)


def matched_change(a, b, valid, rng):
    xa, xb = bright(a), bright(b)
    idx = np.flatnonzero(valid.ravel())
    idx = rng.choice(idx, size=min(60000, idx.size), replace=False)
    x, y = xa.ravel()[idx], xb.ravel()[idx]
    keep = np.ones(x.size, bool)
    for _ in range(4):
        s, c = np.polyfit(x[keep], y[keep], 1)
        res = y - (s * x + c)
        med = np.median(res[keep])
        mad = np.median(np.abs(res[keep] - med))
        keep = np.abs(res - med) < 2.5 * 1.4826 * mad
    d = ndimage.gaussian_filter(np.where(valid, (xb - c) / s - xa, 0.0), 0.5)
    d -= np.median(d[valid])
    sigma = 1.4826 * np.median(np.abs(d[valid]))
    return d, float(sigma), float(s), float(c)


def grow_marks(d, sigma, valid):
    strict = drop_small((d > STRONG_Z * sigma) & valid, 12)
    weak = (d > WEAK_Z * sigma) & valid
    lab, _ = ndimage.label(weak)
    grown = (np.isin(lab, np.unique(lab[strict & weak])) & weak) | strict
    grown = ndimage.binary_closing(grown, structure=np.ones((3, 3)))
    grown = fill_small_holes(grown, 38)
    return drop_small(grown & valid, 12)


def blobs(mask, min_px, grow_px):
    m = drop_small(mask, min_px)
    return ndimage.binary_dilation(m, iterations=grow_px) if m.any() else m


# ------------------------------------------------------------------ flood ----
def process(eid: str, stage: str, images: str) -> dict:
    rng = np.random.default_rng(int(eid))
    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    r = ev[ev.EVENT_ID.astype(int) == int(eid)].iloc[0]
    out = {"event": eid, "place": f"{str(r.CZ_NAME).title()}, {str(r.STATE).title()}",
           "region": str(r.huc4), "flood_date": str(r.BEGIN_UTC.date()),
           "damage": str(r.DAMAGE_PROPERTY)}

    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
        b, crs, tr20, shp20 = s.bounds, s.crs, s.transform, (s.height, s.width)
        coh_base, coh_ev = s.read(2), s.read(3)
    grid = {"crs": crs, "shape": shp20, "transform": tr20}
    t, lon, lat = r.BEGIN_UTC, r.BEGIN_LON, r.BEGIN_LAT

    after = sorted(dedupe([x for x in search(lon, lat, t.strftime("%Y-%m-%d"),
                                             (t + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"))
                           if pd.Timestamp(x["properties"]["datetime"]).tz_localize(None) > t]),
                   key=lambda x: x["properties"]["datetime"])
    before = sorted(dedupe(search(lon, lat, (t - pd.Timedelta(days=PRE_DAYS + CTRL_MAX_GAP)).strftime("%Y-%m-%d"),
                                  (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d"))),
                    key=lambda x: x["properties"]["datetime"], reverse=True)
    post = first_clear(after, grid)
    if post is None:
        out["skip"] = "S1 no clear photo after the flood"
        return out
    want = (post["feat"]["properties"].get("grid:code"), track(post["feat"]))
    near_before = [x for x in before if pd.Timestamp(x["properties"]["datetime"][:10]) >= t.normalize() - pd.Timedelta(days=PRE_DAYS)]
    pre = first_clear(near_before, grid, want)
    if pre is None:
        out["skip"] = "S1 no clear photo before the flood on the same track"
        return out
    ctrl_c = [x for x in before
              if CTRL_MIN_GAP <= (pre["date"] - pd.Timestamp(x["properties"]["datetime"][:10])).days <= CTRL_MAX_GAP]
    ctrl = first_clear(ctrl_c, grid, want)
    if ctrl is None:
        out["skip"] = "S1 no clear no-flood control photo on the same track"
        return out

    valid = pre["clear"] & post["clear"]
    vc = ctrl["clear"] & pre["clear"]
    d, sigma_flood, slope, icpt = matched_change(pre, post, valid, rng)
    dc, sigma_ctrl, _, _ = matched_change(ctrl, pre, vc, rng)
    sigma = sigma_ctrl if sigma_ctrl > 0 else sigma_flood
    first = grow_marks(d, sigma, valid)

    n_pre, n_post, n_ctrl = ndvi(pre), ndvi(post), ndvi(ctrl)
    dn = ndimage.gaussian_filter(n_post - n_pre, 0.5)
    greener = blobs((dn > GREENER) & valid, 8, 1)
    green = blobs((np.maximum(np.maximum(n_pre, n_post), np.where(ctrl["clear"], n_ctrl, -1)) > GREEN) & valid, 8, 1)
    dnc = ndimage.gaussian_filter(n_pre - n_ctrl, 0.5)
    unstable = blobs(((np.abs(dc) > 3 * sigma) | (np.abs(dnc) > 0.10)) & vc, 25, 1)

    v20 = valid
    m20 = drop_small(first & ~greener & ~green & ~unstable, 12)
    m20 = ndimage.binary_closing(m20, structure=np.ones((3, 3)))
    m20 = fill_small_holes(m20, 25)
    m20 = drop_small(m20 & v20, 12)

    pct = 100 * float(m20[v20].mean()) if v20.any() else 0.0
    lab, n = ndimage.label(m20)
    sizes = ndimage.sum(m20, lab, range(1, n + 1)) if n else np.array([])
    dot_share = float(sizes[sizes < DOT_CELLS].sum() / max(sizes.sum(), 1)) if n else 0.0
    hole_share = float((ndimage.binary_fill_holes(m20) & ~m20).sum() / max(m20.sum(), 1))

    rv = np.isfinite(coh_ev) & np.isfinite(coh_base) & v20
    radar = None
    if (m20 & rv).sum() >= 200 and (~m20 & rv).sum() >= 200:
        during = float((1 - coh_ev[m20 & rv]).mean() - (1 - coh_ev[~m20 & rv]).mean())
        beforef = float((1 - coh_base[m20 & rv]).mean() - (1 - coh_base[~m20 & rv]).mean())
        radar = round(during - beforef, 3)

    out.update({
        "before_date": str(pre["date"].date()), "after_date": str(post["date"].date()),
        "control_date": str(ctrl["date"].date()), "track": want[1], "tile": want[0],
        "days_before": int((t.normalize() - pre["date"]).days),
        "days_after": int((post["date"] - t.normalize()).days),
        "clear_pct": round(100 * float(valid.mean())),
        "noise_flood_pair": round(sigma_flood, 4), "noise_control_pair": round(sigma_ctrl, 4),
        "light_slope": round(slope, 4), "light_intercept": round(icpt, 5),
        "bright_before": round(float(np.median(bright(pre)[valid])), 4),
        "bright_after": round(float(np.median(bright(post)[valid])), 4),
        "marked_first_pct": round(100 * float(first[valid].mean()), 1),
        "removed_greener_pct": round(100 * float((first & greener)[valid].mean()), 1),
        "removed_green_pct": round(100 * float((first & green & ~greener)[valid].mean()), 1),
        "removed_unstable_pct": round(100 * float((first & unstable & ~green & ~greener)[valid].mean()), 1),
        "marked_pct": round(pct, 2), "pieces": int(n),
        "dot_share": round(dot_share, 3), "hole_share": round(hole_share, 3),
        "radar_extra_change": radar})

    if pct > MAX_MARKED:
        out["skip"] = f"S2 too much marked ({pct:.0f}%)"
    elif pct < MIN_MARKED:
        out["skip"] = "S2 no visible flood marks"
    elif dot_share > MAX_DOT_SHARE:
        out["skip"] = f"S3 scattered dots ({100*dot_share:.0f}% of the marks in small pieces)"
    elif radar is None:
        out["skip"] = "S4 the radar cannot check these marks"
    elif radar <= RADAR_MIN:
        out["skip"] = f"S4 the radar does not agree ({radar:+.3f})"

    od = OUTROOT / stage
    (od / "arrays").mkdir(parents=True, exist_ok=True)

    def preview(p):
        rgb = np.moveaxis(p["refl"][[2, 1, 0]], 0, -1)
        lo, hi = np.percentile(rgb[p["clear"]], [1, 99])
        return (np.clip((rgb - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype("uint8")

    np.savez_compressed(
        od / "arrays" / f"{eid}.npz",
        d=d.astype("float16"), dc=dc.astype("float16"), sigma=sigma,
        valid=valid, vc=vc, first=first, greener=greener, green=green, unstable=unstable,
        marks=m20, ndvi_before=n_pre.astype("float16"), ndvi_after=n_post.astype("float16"),
        rgb_before=preview(pre), rgb_after=preview(post),
        light=np.array([slope, icpt], "float32"))
    if "skip" not in out:
        prof = dict(driver="GTiff", height=shp20[0], width=shp20[1], count=3, dtype="float32",
                    crs=crs, transform=tr20, compress="deflate")
        d20 = d.astype("float32")
        with rasterio.open(od / f"{eid}.tif", "w", **prof) as o:
            o.write(m20.astype("float32"), 1); o.set_band_description(1, "trace")
            o.write(v20.astype("float32"), 2); o.set_band_description(2, "valid")
            o.write(d20, 3); o.set_band_description(3, "change")

    if images == "all" or (images == "kept" and "skip" not in out):
        vis = {}
        for name, p in (("before", pre), ("after", post)):
            v = np.stack([read(p["feat"]["assets"]["visual"]["href"], grid, band=k) for k in (1, 2, 3)], -1)
            vis[name] = v
        lo, hi = np.percentile(np.stack([vis["before"], vis["after"]])[:, valid], [1, 99])
        pic = {k: (np.clip((v - lo) / (hi - lo), 0, 1) * 255).astype("uint8") for k, v in vis.items()}
        over = pic["after"].copy()
        removed = first & ~m20
        final10 = m20
        over[removed] = (0.35 * over[removed] + 0.65 * np.array([60, 110, 200], "float32")).astype("uint8")
        over[final10] = (0.25 * over[final10] + 0.75 * np.array([235, 40, 20], "float32")).astype("uint8")
        S = 420
        row = Image.new("RGB", (3 * S + 20, S), "white")
        for k, a in enumerate((pic["before"], pic["after"], over)):
            row.paste(Image.fromarray(a).resize((S, S), Image.BILINEAR), (k * (S + 10), 0))
        (PAGE / stage).mkdir(parents=True, exist_ok=True)
        row.save(PAGE / stage / f"e{eid}.jpg", quality=84)
    return out


def safe(eid, stage, images):
    try:
        return process(eid, stage, images)
    except Exception as e:
        return {"event": eid, "skip": f"error: {e}", "trace": traceback.format_exc()}


# ------------------------------------------------------------------- main ----
def choose(n, seed, exclude):
    idx = pd.read_csv(ROOT / "data/tiles_v4/index.csv", usecols=["event", "huc4"])
    per = idx.groupby("event").agg(huc4=("huc4", "first"), tiles=("huc4", "size")).reset_index()
    per["event"] = per.event.astype(str)
    per = per[~per.event.isin(exclude)]
    if n <= 0:
        return per.event.tolist()
    # spread over regions in proportion to their floods, at least one each
    rng = np.random.default_rng(seed)
    share = per.huc4.value_counts(normalize=True)
    quota = {h: max(1, int(round(n * s))) for h, s in share.items()}
    while sum(quota.values()) > n:
        quota[max(quota, key=quota.get)] -= 1
    while sum(quota.values()) < n:
        quota[max(share.index, key=lambda h: share[h] * n - quota[h])] += 1
    pick = []
    for h, q in quota.items():
        g = per[per.huc4 == h].event.tolist()
        pick += list(rng.choice(g, size=min(q, len(g)), replace=False))
    return [str(x) for x in pick]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--n", type=int, required=True, help="floods to sample; 0 = every flood with tiles")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--exclude-stages", default="", help="comma-separated stage names whose floods to leave out")
    ap.add_argument("--events", default="", help="use exactly these floods (re-run of a stage)")
    ap.add_argument("--images", choices=["all", "kept", "none"], default="all")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    od = OUTROOT / args.stage
    od.mkdir(parents=True, exist_ok=True)
    if args.events:
        events = args.events.split(",")
    elif args.n <= 0:
        # the full run relabels every flood with the final rules, sample floods included
        events = choose(0, args.seed, set())
    else:
        # rule-finding samples use floods the rules have not been tuned on yet
        exclude = {str(x["event"]) for x in json.load(open(ROOT / "data/trace_labels/pilot_10.json"))}
        for s in [x for x in args.exclude_stages.split(",") if x]:
            exclude |= set(json.load(open(OUTROOT / s / "events.json")))
        events = choose(args.n, args.seed, exclude)
    (od / "events.json").write_text(json.dumps(events))

    t0 = time.time()
    results, kept = [], 0
    prog = od / "progress.json"
    prog.write_text(json.dumps({"stage": args.stage, "total": len(events), "done": 0, "kept": 0,
                                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "finished": None}))
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(safe, e, args.stage, args.images): e for e in events}
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            kept += "skip" not in res
            (od / f"{res['event']}.json").write_text(json.dumps(res, indent=1))
            prog.write_text(json.dumps({"stage": args.stage, "total": len(events), "done": len(results),
                                        "kept": kept, "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
                                        "finished": None}))
            short = res.get("skip", f"KEPT {res.get('marked_pct')}% marked, radar {res.get('radar_extra_change')}")
            print(f"[{len(results)}/{len(events)}] {res['event']}: {short}", flush=True)

    reasons = pd.Series([r.get("skip", "kept").split(" ")[0] for r in results]).value_counts().to_dict()
    summary = {"stage": args.stage, "total": len(events), "kept": kept, "reasons": reasons,
               "minutes": round((time.time() - t0) / 60, 1)}
    (od / "summary.json").write_text(json.dumps(summary, indent=1))
    prog.write_text(json.dumps({**json.loads(prog.read_text()), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}))
    print(f"\nkept {kept} of {len(events)} floods   {reasons}   {summary['minutes']} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
