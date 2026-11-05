#!/usr/bin/env python3
"""
Apply the flood-trace rules to floods that phase7/mark_floods_v3.py already
downloaded and saved, without downloading anything again. Every rule setting
lives in one JSON file, so a rule can be changed and re-checked in seconds.

    ./.venv/bin/python -u phase7/apply_rules.py --rules data/trace_v3/rules_v1.json --stage all

Rules on the marks (recomputed from the saved change and greenness arrays)
  noise level = the larger of the no-flood control pair's noise and
  flood_noise_share x the flood pair's own noise. The control alone can be
  too small: pairs that look identical got 7-22% marked. The flood pair's
  own noise is inflated by a big flood, hence the share.
  seeds above strong_z noise levels grown into change above weak_z levels,
  minus ground that got greener, was green, or also changed between the two
  no-flood photos, minus ground within cloud_gap_cells of a cloud, shadow or
  other unclear area (marks kept appearing along cloud edges); gaps closed,
  small holes filled, small pieces removed.

Rules that skip a whole flood (whole-flood skips cannot teach the model a
pixel rule from its own inputs, unlike removing town or farm pixels)
  photos      : no clear photos (decided at download time)
  clear       : less than min_clear_pct % of the square clear in both photos;
                haze around clouds leaks into the rest (the 2017-2018 cloud mask
                calls many big clouds "medium probability", which the download
                rule for big clouds does not see)
  size        : marked share of the clear ground outside [min_marked, max_marked] %
  dots        : share of the marks in pieces under dot_cells cells above max_dot_share
  blocky      : share of the marks in path-like pieces below min_path_share.
                Flood traces are long or branching; fields, lots and town blocks
                are compact blocks. A piece is path-like when its compactness
                (4 pi area / perimeter^2: square 0.785, ragged network near 0)
                is at most max_compact. Elongation alone is not used: a long
                field strip is elongated too (stage s20: towns and farms)
  busy        : share of the marks on busy ground above max_busy_share. Busy
                ground (towns, house lots, rock) has high local contrast in the
                before photo: brightness spread in a busy_window x busy_window
                cell window above busy_factor x the square's median spread.
                Flood traces lie on smooth valley floors and washes (review of
                380 floods: most false keeps were street grids and lots)
  light       : after photo much darker or brighter than before (wet ground
                after rain, haze). Measured by the slope of the robust light
                fit (after against before on unchanged ground); skip below
                min_light_slope or above max_light_slope. Wet-ground floods
                1295895 and 782683 have slopes 0.76 and 0.73. An earlier
                version used the whole-square brightness ratio, which a big
                bright flood raises itself: it skipped Painted Rock (ratio
                1.198, slope 1.07). Rules files without min_light_slope keep
                the old ratio rule.
  radar       : radar cannot check, or extra coherence loss inside the marks
                below radar_min (stage s20: fake floods passed at +0.03,
                Painted Rock scores +0.21)

Output: data/trace_v3/labels/<rules name>/<event>.tif (kept floods only),
        decisions.csv, summary.json, sheets/*.jpg for checking by eye
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent


def drop_small(mask, min_px):
    lab, n = ndimage.label(mask)
    if n == 0:
        return mask
    size = ndimage.sum(mask, lab, range(1, n + 1))
    return np.isin(lab, 1 + np.flatnonzero(size >= min_px))


def fill_small_holes(mask, max_px):
    holes = ndimage.binary_fill_holes(mask) & ~mask
    return mask | (drop_small(holes, 1) & ~drop_small(holes, max_px + 1))


def blobs(mask, min_px, grow_px):
    m = drop_small(mask, min_px)
    return ndimage.binary_dilation(m, iterations=grow_px) if (m.any() and grow_px) else m


def piece_shapes(mask):
    """Per piece: size, compactness 4*pi*area/perimeter^2, elongation.
    A square block scores 0.785, a ragged branching flood network close to 0.
    (A first try used fill of the piece's spread box; gap filling makes a big
    flood network look full, and it called Painted Rock blocky.)"""
    lab, n = ndimage.label(mask)
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), 1):
        piece = np.pad(lab[sl] == k, 1)
        a = int(piece.sum())
        per = int(np.abs(np.diff(piece.astype(np.int8), axis=0)).sum()
                  + np.abs(np.diff(piece.astype(np.int8), axis=1)).sum())
        yy, xx = np.nonzero(piece)
        if a >= 3:
            l1, l2 = sorted(np.maximum(np.linalg.eigvalsh(np.cov(np.vstack([yy, xx]))), 1e-6), reverse=True)
            elong = float(np.sqrt(l1 / l2))
        else:
            elong = 1.0
        out.append((a, float(4 * np.pi * a / max(per, 1) ** 2), elong))
    return out


def decide(eid: str, info: dict, z, R: dict) -> tuple[dict, np.ndarray | None]:
    d, valid = z["d"].astype("float32"), z["valid"]
    dv = d[valid]
    sigma_flood = 1.4826 * float(np.median(np.abs(dv - np.median(dv)))) if dv.size else 0.0
    sigma = max(float(z["sigma"]), R["flood_noise_share"] * sigma_flood)
    strict = drop_small((d > R["strong_z"] * sigma) & valid, R["seed_cells"])
    weak = (d > R["weak_z"] * sigma) & valid
    lab, _ = ndimage.label(weak)
    first = (np.isin(lab, np.unique(lab[strict & weak])) & weak) | strict
    first = ndimage.binary_closing(first, structure=np.ones((3, 3)))
    first = drop_small(fill_small_holes(first, R["hole_cells"]) & valid, R["piece_cells"])

    nb, na = z["ndvi_before"].astype("float32"), z["ndvi_after"].astype("float32")
    dn = ndimage.gaussian_filter(na - nb, 0.5)
    greener = blobs((dn > R["greener"]) & valid, 8, 1)
    green = blobs((np.maximum(nb, na) > R["green"]) & valid, 8, 1) | z["green"]
    dc, vc = z["dc"].astype("float32"), z["vc"]
    unstable = blobs((np.abs(dc) > R["unstable_z"] * sigma) & vc, 25, 1) | z["unstable"]

    near_cloud = ndimage.distance_transform_edt(valid) <= R["cloud_gap_cells"]
    m = drop_small(first & ~greener & ~green & ~unstable & ~near_cloud, R["piece_cells"])
    m = ndimage.binary_closing(m, structure=np.ones((3, 3)))
    m = drop_small(fill_small_holes(m, R["label_hole_cells"]) & valid & ~near_cloud, R["piece_cells"])

    pct = 100 * float(m[valid].mean()) if valid.any() else 0.0
    shapes = piece_shapes(m)
    area = sum(s[0] for s in shapes) or 1
    dots = sum(s[0] for s in shapes if s[0] < R["dot_cells"]) / area
    paths = sum(s[0] for s in shapes if s[0] >= R["dot_cells"] and s[1] <= R["max_compact"]) / area
    light = abs(info["bright_after"] / max(info["bright_before"], 1e-6) - 1)
    slope = float(info.get("light_slope", 1.0))
    if "min_light_slope" in R:
        light_bad = not (R["min_light_slope"] <= slope <= R["max_light_slope"])
    else:
        light_bad = light > R["max_light"]
    g = z["rgb_before"].astype("float32").mean(-1)
    w = R["busy_window"]
    mean = ndimage.uniform_filter(g, w)
    spread = np.sqrt(np.maximum(ndimage.uniform_filter(g * g, w) - mean * mean, 0))
    busy_ground = spread > R["busy_factor"] * np.median(spread[valid]) if valid.any() else np.zeros_like(valid)
    busy = float(busy_ground[m].mean()) if m.any() else 0.0

    with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
        coh_base, coh_ev = s.read(2), s.read(3)
    rv = np.isfinite(coh_ev) & np.isfinite(coh_base) & valid
    radar = None
    if (m & rv).sum() >= R["radar_cells"] and (~m & rv).sum() >= R["radar_cells"]:
        radar = float(((1 - coh_ev[m & rv]).mean() - (1 - coh_ev[~m & rv]).mean())
                      - ((1 - coh_base[m & rv]).mean() - (1 - coh_base[~m & rv]).mean()))

    row = {"event": eid, "region": info["region"], "flood_date": info["flood_date"], "damage": info["damage"],
           "days_after": info["days_after"], "marked_pct": round(pct, 2), "pieces": len(shapes),
           "dot_share": round(dots, 3), "path_share": round(paths, 3), "busy_share": round(busy, 3),
           "light_change": round(light, 3), "light_slope": round(slope, 3),
           "radar": None if radar is None else round(radar, 3)}
    row["clear_pct"] = info.get("clear_pct")
    row["noise"] = round(sigma, 4)
    if info.get("clear_pct", 100) < R["min_clear_pct"]:
        row["skip"] = "clear: clouds or haze in a photo"
    elif pct > R["max_marked"]:
        row["skip"] = "size: too much marked"
    elif pct < R["min_marked"]:
        row["skip"] = "size: no visible marks"
    elif dots > R["max_dot_share"]:
        row["skip"] = "dots: scattered"
    elif busy > R["max_busy_share"]:
        row["skip"] = "busy: marks on towns or rock"
    elif light_bad:
        row["skip"] = "light: photos differ in overall brightness"
    elif paths < R["min_path_share"]:
        row["skip"] = "blocky: marks are blocks, not flood paths"
    elif radar is None and R.get("require_radar", True):
        row["skip"] = "radar: cannot check"
    elif radar is not None and radar < R["radar_min"]:
        row["skip"] = "radar: does not agree"
    return row, m


def sheet(rows, arrays, labels, path, cols=2, S=300):
    rows = list(rows)
    if not rows:
        return
    W = cols * (3 * S + 20)
    H = (S + 22) * ((len(rows) + cols - 1) // cols)
    img = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(img)
    for k, r in enumerate(rows):
        x0, y0 = (k % cols) * (3 * S + 20), (k // cols) * (S + 22)
        z = arrays[r["event"]]
        m = z["marks"]
        over = z["rgb_after"].copy()
        colour = [235, 40, 20] if r["event"] in labels else [250, 210, 0]
        over[m] = (0.25 * over[m] + 0.75 * np.array(colour, "float32")).astype("uint8")
        for j, a in enumerate((z["rgb_before"], z["rgb_after"], over)):
            img.paste(Image.fromarray(a).resize((S, S)), (x0 + j * S, y0 + 22))
        dr.text((x0 + 4, y0 + 5), f"{r['event']} {r['flood_date']} {r['damage']} marked {r['marked_pct']}% "
                f"paths {r['path_share']} busy {r.get('busy_share')} dots {r['dot_share']} light {r['light_change']} radar {r['radar']} "
                f"{r.get('skip', 'KEPT')}", fill="black")
    img.save(path, quality=82)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rules", required=True)
    ap.add_argument("--stage", default="all")
    ap.add_argument("--sheets", type=int, default=12, help="floods per sheet")
    args = ap.parse_args()

    R = json.loads(Path(args.rules).read_text())
    src = ROOT / "data/trace_v3" / args.stage
    out = ROOT / "data/trace_v3/labels" / R["name"]
    (out / "sheets").mkdir(parents=True, exist_ok=True)
    for f in out.glob("*.tif"):
        f.unlink()

    rows, keep_masks, arrays = [], {}, {}
    for jf in sorted(src.glob("[0-9]*.json")):
        info = json.loads(jf.read_text())
        eid = info["event"]
        npz = src / "arrays" / f"{eid}.npz"
        if not npz.exists():
            rows.append({"event": eid, "region": info.get("region"), "flood_date": info.get("flood_date"),
                         "damage": info.get("damage"), "skip": "photos: " + info.get("skip", "none saved")})
            continue
        z = dict(np.load(npz))
        row, m = decide(eid, info, z, R)
        rows.append(row)
        arrays[eid] = {"rgb_before": z["rgb_before"], "rgb_after": z["rgb_after"], "marks": m}
        if "skip" not in row:
            keep_masks[eid] = m
            with rasterio.open(ROOT / f"data/features/{eid}/label.tif") as s:
                prof = dict(driver="GTiff", height=s.height, width=s.width, count=2, dtype="float32",
                            crs=s.crs, transform=s.transform, compress="deflate")
            with rasterio.open(out / f"{eid}.tif", "w", **prof) as o:
                o.write(m.astype("float32"), 1); o.set_band_description(1, "trace")
                o.write(z["valid"].astype("float32"), 2); o.set_band_description(2, "valid")

    df = pd.DataFrame(rows)
    if "skip" not in df:
        df["skip"] = None
    df.to_csv(out / "decisions.csv", index=False)
    reasons = df["skip"].fillna("kept").str.split(":").str[0].value_counts().to_dict()
    idx = pd.read_csv(ROOT / "data/tiles_v4/index.csv", usecols=["event"])
    tiles = int(idx.event.astype(str).isin(keep_masks).sum())
    summary = {"rules": R["name"], "floods": len(df), "kept": len(keep_masks), "reasons": reasons,
               "tiles_of_kept_floods_before_masking": tiles}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))

    for f in (out / "sheets").glob("*.jpg"):
        f.unlink()
    kept = df[df.skip.isna()].to_dict("records") if "skip" in df else df.to_dict("records")
    for k in range(0, len(kept), args.sheets):
        sheet(kept[k:k + args.sheets], arrays, keep_masks, out / "sheets" / f"kept_{k // args.sheets + 1:02d}.jpg")
    with_arrays = df[df.event.isin(arrays) & df.skip.notna()] if "skip" in df else df.iloc[0:0]
    for reason, g in with_arrays.groupby(with_arrays.skip.str.split(":").str[0]):
        sheet(g.head(args.sheets).to_dict("records"), arrays, {}, out / "sheets" / f"skipped_{reason}.jpg")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
