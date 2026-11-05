#!/usr/bin/env python3
"""
Flood-trace labels from before/after Sentinel-2 photos, one fixed method for
every flood (the Painted Rock pilot, version B "gaps filled").

A desert flash flood drains in hours, but it leaves fresh pale sand and mud
along the washes for weeks. So instead of looking for water, look for ground
that became paler between the last clear photo before the flood and the first
clear photo after it.

Only the two photos are used. No terrain, basin or land-cover layer, because
the model gets those as inputs and a label built from them would leak.

Per flood:
  1. photos : latest photo before and earliest photo after the flood that are
              at least 95% clear over the 10 km square (SCL cloud/shadow mask)
  2. change : brightness after minus before, each minus its own median, so a
              whole-scene shift in light does not count
  3. marks  : strong change (>12) as seeds, grown into joined weaker change
              (>6), small gaps closed, small holes filled, specks removed
  4. label  : 20 m grid of the model, a cell is flood if half of it is marked
  5. checks : radar ground change inside vs outside the marks, and other storm
              reports near the square between the two photo dates

    ./.venv/bin/python -u phase7/mark_floods.py --events 983830 \
        --pick 1809:big,1809:small,1606:big,1606:small,1503:big,1503:small,1501:any,1810:any,1507:small

Output: data/trace_labels/<event>.tif  (trace, valid, change) on the label grid
        data/trace_labels/<event>.json (dates, sizes, checks)
        handoff/hand_label_pilot/page/e<event>_{before,after,marks}.jpg
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

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
LAB = ROOT / "data/trace_labels"
PAGE = ROOT / "handoff/hand_label_pilot/page"
SCL_BAD = [0, 1, 3, 8, 9, 10]
PRE_DAYS, POST_DAYS, MIN_CLEAR, MAX_TRIES = 45, 30, 0.95, 8
STRONG, WEAK = 12.0, 6.0


def dmg(s) -> float:
    s = str(s).strip()
    if s in ("nan", ""):
        return np.nan
    mult = {"K": 1e3, "M": 1e6, "B": 1e9}
    return float(s[:-1]) * mult[s[-1]] if s[-1] in mult else float(s)


def search(lon, lat, start, end):
    body = {"collections": ["sentinel-2-l2a"],
            "intersects": {"type": "Point", "coordinates": [lon, lat]},
            "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z", "limit": 100,
            "query": {"eo:cloud_cover": {"lt": 60}}}
    for _ in range(3):
        try:
            r = requests.post(STAC, json=body, timeout=120)
            r.raise_for_status()
            return r.json().get("features", [])
        except Exception:
            continue
    return []


def read(href, grid, bands, nearest=False):
    buf = np.zeros((len(bands),) + grid["shape"], dtype="float32")
    with rasterio.open(href) as s:
        for k, b in enumerate(bands):
            reproject(source=rasterio.band(s, b), destination=buf[k],
                      dst_transform=grid["transform"], dst_crs=grid["crs"],
                      resampling=Resampling.nearest if nearest else Resampling.bilinear)
    return buf


def clear_photo(feats, grid):
    tried = 0
    for f in feats:
        if tried >= MAX_TRIES:
            break
        tried += 1
        try:
            scl = read(f["assets"]["scl"]["href"], grid, [1], nearest=True)[0]
        except Exception:
            continue
        clear = ~np.isin(scl.astype(int), SCL_BAD) & (scl > 0)
        if clear.mean() < MIN_CLEAR:
            continue
        try:
            rgb = read(f["assets"]["visual"]["href"], grid, [1, 2, 3])
        except Exception:
            continue
        clear &= rgb.sum(0) > 0
        if clear.mean() >= MIN_CLEAR:
            return f, rgb, clear
    return None


def drop_small(mask, min_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    size = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(size >= min_px))


def marks_from(pre, post, valid):
    bp, bq = pre.mean(0), post.mean(0)
    d = (bq - np.median(bq[valid])) - (bp - np.median(bp[valid]))
    d = ndimage.gaussian_filter(np.where(valid, d, 0), 1.0)
    strict = ndimage.binary_opening((d > STRONG) & valid)
    strict = drop_small(strict, 50)
    weak = (d > WEAK) & valid
    lab, _ = ndimage.label(weak)
    grown = (np.isin(lab, np.unique(lab[strict & weak])) & weak) | strict
    grown = ndimage.binary_closing(grown, structure=np.ones((3, 3)), iterations=2)
    holes = ndimage.binary_fill_holes(grown) & ~grown
    grown |= drop_small(holes, 1) & ~drop_small(holes, 150)
    return drop_small(grown & valid, 50), d


def process(r, near_reports) -> dict | None:
    eid = str(int(r.EVENT_ID))
    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
        b, crs, tr20, shp20 = s.bounds, s.crs, s.transform, (s.height, s.width)
        coh_base, coh_ev = s.read(2), s.read(3)
    with rasterio.open(ROOT / f"data/features/{eid}/valid.tif") as s:
        rvalid = s.read(1) > 0
    grid = {"crs": crs, "shape": (shp20[0] * 2, shp20[1] * 2),
            "transform": from_origin(b.left, b.top, 10, 10)}
    t = r.BEGIN_UTC
    lon, lat = r.BEGIN_LON, r.BEGIN_LAT

    before = sorted(search(lon, lat, (t - pd.Timedelta(days=PRE_DAYS)).strftime("%Y-%m-%d"),
                           (t - pd.Timedelta(days=1)).strftime("%Y-%m-%d")),
                    key=lambda x: x["properties"]["datetime"], reverse=True)
    got_pre = clear_photo(before, grid)
    if got_pre is None:
        print(f"  {eid}: no clear photo before", flush=True)
        return None
    after = [x for x in search(lon, lat, t.strftime("%Y-%m-%d"),
                               (t + pd.Timedelta(days=POST_DAYS)).strftime("%Y-%m-%d"))
             if pd.Timestamp(x["properties"]["datetime"]).tz_localize(None) > t]
    after = sorted(after, key=lambda x: x["properties"]["datetime"])
    got_post = clear_photo(after, grid)
    if got_post is None:
        print(f"  {eid}: no clear photo after", flush=True)
        return None
    (fp, pre, cp), (fq, post, cq) = got_pre, got_post
    valid = cp & cq
    mark, d = marks_from(pre, post, valid)

    m20 = mark.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)) >= 0.5
    v20 = valid.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)) >= 0.5
    d20 = d.reshape(shp20[0], 2, shp20[1], 2).mean((1, 3)).astype("float32")
    LAB.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", height=shp20[0], width=shp20[1], count=3, dtype="float32",
                crs=crs, transform=tr20, compress="deflate")
    with rasterio.open(LAB / f"{eid}.tif", "w", **prof) as o:
        o.write(m20.astype("float32"), 1); o.set_band_description(1, "trace")
        o.write(v20.astype("float32"), 2); o.set_band_description(2, "valid")
        o.write(d20, 3); o.set_band_description(3, "change")

    rv = rvalid & np.isfinite(coh_ev) & np.isfinite(coh_base) & v20
    radar = {}
    if (m20 & rv).sum() >= 200 and (~m20 & rv).sum() >= 200:
        radar = {"change_in_marks": round(float(1 - coh_ev[m20 & rv].mean()), 3),
                 "change_elsewhere": round(float(1 - coh_ev[~m20 & rv].mean()), 3),
                 "change_in_marks_before_flood": round(float(1 - coh_base[m20 & rv].mean()), 3),
                 "change_elsewhere_before_flood": round(float(1 - coh_base[~m20 & rv].mean()), 3)}

    dp = pd.Timestamp(fp["properties"]["datetime"][:10])
    dq = pd.Timestamp(fq["properties"]["datetime"][:10])
    others = near_reports(r, dp, dq)
    info = {"event": eid, "region": str(r.huc4), "place": f"{str(r.CZ_NAME).title()}, {str(r.STATE).title()}",
            "basin": r.basin_name, "flood_date": str(t.date()), "damage": str(r.DAMAGE_PROPERTY),
            "before_date": str(dp.date()), "after_date": str(dq.date()),
            "days_before": int((t.normalize() - dp).days), "days_after": int((dq - t.normalize()).days),
            "clear_pct": round(100 * float(valid.mean())),
            "marked_pct": round(100 * float(mark[valid].mean()), 1),
            "marked_pct_20m": round(100 * float(m20[v20].mean()), 1),
            "pieces": int(ndimage.label(mark)[1]),
            "radar": radar, "other_reports_between_photos": others}
    (LAB / f"{eid}.json").write_text(json.dumps(info, indent=1))

    lo, hi = np.percentile(np.stack([pre, post])[:, :, valid], [1, 99])
    def pic(a):
        return (np.clip((np.moveaxis(a, 0, -1) - lo) / (hi - lo), 0, 1) * 255).astype("uint8")
    pb, pa = pic(pre), pic(post)
    over = pa.copy()
    red = np.array([235, 40, 20], "float32")
    over[mark] = (0.25 * over[mark] + 0.75 * red).astype("uint8")
    PAGE.mkdir(parents=True, exist_ok=True)
    for name, a in (("before", pb), ("after", pa), ("marks", over)):
        Image.fromarray(a).resize((800, 800), Image.BILINEAR).save(PAGE / f"e{eid}_{name}.jpg", quality=85)
    print(f"  {eid}: r{r.huc4}  before {dp.date()} (-{info['days_before']}d)  after {dq.date()} "
          f"(+{info['days_after']}d)  marked {info['marked_pct']}%  radar {radar}", flush=True)
    return info


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", default="", help="comma-separated event ids to always include")
    ap.add_argument("--pick", default="", help="region:size groups, size in big/small/any")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    ev["dmg"] = ev.DAMAGE_PROPERTY.map(dmg)
    have = {p.name for p in (ROOT / "data/features").iterdir() if (p / "label.tif").exists()}
    ev = ev[ev.EVENT_ID.astype(int).astype(str).isin(have)]

    def near_reports(r, d0, d1):
        k = ev[(np.hypot((ev.BEGIN_LAT - r.BEGIN_LAT) * 111, (ev.BEGIN_LON - r.BEGIN_LON) * 93) < 30)
               & (ev.BEGIN_UTC >= d0) & (ev.BEGIN_UTC <= d1 + pd.Timedelta(days=1))
               & (ev.EVENT_ID != r.EVENT_ID)]
        return [f"{str(x.BEGIN_UTC.date())} {x.DAMAGE_PROPERTY}" for x in k.itertuples()]

    done, results = set(), []
    for e in [x for x in args.events.split(",") if x]:
        row = ev[ev.EVENT_ID.astype(int) == int(e)].iloc[0]
        info = process(row, near_reports)
        done.add(e)
        if info:
            results.append(info | {"group": "pilot"})

    rng = np.random.default_rng(args.seed)
    for g in [x for x in args.pick.split(",") if x]:
        region, size = g.split(":")
        c = ev[(ev.huc4.astype(str) == region) & (ev.BEGIN_UTC.dt.year >= 2017)]
        if size == "big":
            c = c[c.dmg >= 1e5]
        elif size == "small":
            c = c[c.dmg.fillna(0) < 1e4]
        c = c.iloc[rng.permutation(len(c))]
        print(f"group {g}: {len(c)} candidates", flush=True)
        for row in c.itertuples(index=False):
            e = str(int(row.EVENT_ID))
            if e in done:
                continue
            done.add(e)
            info = process(pd.Series(row._asdict()), near_reports)
            if info:
                results.append(info | {"group": g})
                break

    pd.DataFrame(results).to_json(LAB / "pilot_10.json", orient="records", indent=1)
    print(f"\nmarked {len(results)} floods, wrote {LAB / 'pilot_10.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
