#!/usr/bin/env python3
"""
Build the small sample pack in samples/ from the full tiles in data/.

Reads only. Run from the repo root (needs data/tiles_v4 and data/tiles_neg_v4):

    ./.venv/bin/python -u samples/make_samples.py

All feature values in the output are RAW: before the log1p and z-score steps
that phase6/train.py applies when it loads tiles.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "samples"
THR = 0.15
TEST, VAL = {"1606", "1809"}, {"1507"}
SHARD = 512
N_PIXELS = 20_000
TAG = "terrain+basin+rain+rain_routed+soil+landcover"
LOG = {"basin_facc", "basin_spi", "catchment_area_km2", "rain_1h_upstream_vol_m3",
       "rain_24h_upstream_vol_m3", "runoff_1h_upstream_vol_m3"}

# name: (family, unit, meaning)
FEATURES = {
 "elev_mean_10m": ("terrain", "m", "Mean elevation in the 20 m cell. From 1 m lidar degraded to 10 m. Also used by the elevation term in the model output."),
 "elev_min_10m": ("terrain", "m", "Lowest elevation in the 20 m cell, same source."),
 "microrelief_10m": ("terrain", "m", "Elevation range (max minus min) inside the 20 m cell."),
 "roughness_10m": ("terrain", "m", "Standard deviation of elevation inside the 20 m cell."),
 "slope_10m": ("terrain", "degrees", "Slope from the degraded DEM."),
 "basin_facc": ("basin routing", "cell count (10 m cells)", "Flow accumulation: how many cells drain into this one. Routed over the whole basin, not the tile."),
 "basin_hand": ("basin routing", "m", "Height above the nearest drainage."),
 "basin_slope": ("basin routing", "degrees", "Slope from the basin-wide 10 m DEM."),
 "basin_spi": ("basin routing", "unitless", "Stream power index."),
 "basin_twi": ("basin routing", "unitless", "Topographic wetness index."),
 "rain_1h_peak": ("rain at the point", "mm", "MRMS radar rain over the 1 hour ending at the hour the flood report starts. The name says peak, but it is that single hour, not a maximum over a window."),
 "rain_24h": ("rain at the point", "mm", "MRMS radar rain over the 24 hours ending at that same hour."),
 "rain_ante48h": ("rain at the point", "mm", "MRMS 24 hour total ending 48 hours before the report hour."),
 "rain_ante72h": ("rain at the point", "mm", "MRMS 24 hour total ending 72 hours before the report hour."),
 "soil_bd": ("soil", "g/cm3", "POLARIS bulk density."),
 "soil_clay": ("soil", "percent", "POLARIS clay content."),
 "soil_ksat": ("soil", "log10(cm/hr)", "POLARIS saturated hydraulic conductivity: how fast water can soak in. Stored as a log, so negative values are normal."),
 "soil_sand": ("soil", "percent", "POLARIS sand content."),
 "lc_class": ("land cover", "class code", "ESA WorldCover class code. A category, not a quantity: 10 tree, 20 shrub, 30 grass, 40 crop, 50 built, 60 bare, 70 snow, 80 water, 90 wetland, 95 mangrove, 100 moss."),
 "lc_cropland": ("land cover", "0 or 1", "1 where lc_class is 40."),
 "lc_water": ("land cover", "0 or 1", "1 where lc_class is 80."),
 "catchment_area_km2": ("routed rain", "km2", "Area that drains into this pixel. Routed at 100 m."),
 "rain_1h_over_ksat_upstream_mean": ("routed rain", "ratio", "Mean over the catchment of 1 hour rain divided by ksat."),
 "rain_1h_upstream_depth_mm": ("routed rain", "mm", "Mean 1 hour rain depth over the catchment."),
 "rain_1h_upstream_vol_m3": ("routed rain", "m3", "Total 1 hour rain volume falling on the catchment."),
 "rain_24h_upstream_depth_mm": ("routed rain", "mm", "Mean 24 hour rain depth over the catchment."),
 "rain_24h_upstream_vol_m3": ("routed rain", "m3", "Total 24 hour rain volume falling on the catchment."),
 "rain_cover_frac": ("routed rain", "0 to 1", "Share of the catchment the radar actually saw. Exactly 1.0 on every valid pixel, so it carries no information."),
 "runoff_1h_upstream_depth_mm": ("routed rain", "mm", "Routed infiltration-excess runoff, mean depth. Runoff per cell is max(0, rain_1h - ksat)."),
 "runoff_1h_upstream_vol_m3": ("routed rain", "m3", "Routed infiltration-excess runoff, total volume."),
}


def split_of(h) -> str:
    h = str(h)
    return "test" if h in TEST else "val" if h in VAL else "train"


def load_index(d: str, quiet: bool) -> pd.DataFrame:
    x = pd.read_csv(ROOT / "data" / d / "index.csv")
    x["set"], x["pos"] = d, np.arange(len(x))
    x["is_quiet_control"] = quiet
    x["split"] = x.huc4.map(split_of)
    x["tile_id"] = x.event.astype(str) + "_" + x.row.astype(str) + "_" + x.col.astype(str)
    return x


def pick_tiles(idx: pd.DataFrame) -> pd.DataFrame:
    """12 tiles: 6 with flood at low to high share, 3 dry on a flood day, 3 quiet."""
    good = idx[idx.valid_frac >= 0.999]
    rng = np.random.default_rng(0)
    out = []
    fl = good[(~good.is_quiet_control) & good.has_flood]
    bins = [(.01, .03, "train"), (.03, .08, "train"), (.08, .2, "train"),
            (.2, .5, "train"), (.05, .2, "val"), (.05, .2, "test")]
    for lo, hi, sp in bins:
        c = fl[(fl["flood_frac_0.15"].between(lo, hi)) & (fl.split == sp)]
        out.append(c.iloc[[rng.integers(len(c))]])
    dry = good[(~good.is_quiet_control) & (~good.has_flood)]
    out.append(dry.iloc[rng.choice(len(dry), 3, replace=False)])
    q = good[good.is_quiet_control]
    out.append(q.iloc[rng.choice(len(q), 3, replace=False)])
    return pd.concat(out).reset_index(drop=True)


def main() -> None:
    ck = torch.load(ROOT / f"data/model_{TAG}.pt", map_location="cpu", weights_only=False)
    names = list(ck["channels"])
    assert len(names) == 30 and set(names) == set(FEATURES), "channel list changed"
    chan = json.loads((ROOT / "data/tiles_v4/channels.json").read_text())
    assert names == chan["rung_channels"]["10"] + chan["shared_channels"], "channel order changed"

    idx = pd.concat([load_index("tiles_v4", False), load_index("tiles_neg_v4", True)],
                    ignore_index=True)
    picks = pick_tiles(idx)
    rng = np.random.default_rng(0)

    n_valid_est = float((idx.valid_frac * 112 * 112).sum())
    p = N_PIXELS / n_valid_est

    C = 30
    vmin = np.full(C, np.inf); vmax = np.full(C, -np.inf)
    s1 = np.zeros(C); s2 = np.zeros(C); nvalid = 0; nonfinite = np.zeros(C, int)
    res = [[] for _ in range(C)]
    label_res = []
    rows, tile_x, tile_y, tile_m, tile_meta = [], {}, {}, {}, {}
    cnt = {s: {"tiles": 0, "valid_pixels": 0, "flood_pixels": 0} for s in ("train", "val", "test")}
    shards = [("tiles_v4", p_.name) for p_ in sorted((ROOT / "data/tiles_v4/shared").glob("shard-*.npz"))] \
           + [("tiles_neg_v4", p_.name) for p_ in sorted((ROOT / "data/tiles_neg_v4/shared").glob("shard-*.npz"))]

    for k, (d, name) in enumerate(shards):
        zs = np.load(ROOT / "data" / d / "shared" / name)
        xt = np.load(ROOT / "data" / d / "rung10m" / name)["xt"]
        X = np.concatenate([xt, zs["xs"]], axis=1).astype("float32")
        Y, M = zs["y"].astype("float32"), zs["m"]
        sh = int(name.split("-")[1].split(".")[0])
        rec = idx[idx.set == d].iloc[sh * SHARD: sh * SHARD + len(X)]
        assert len(rec) == len(X)

        for sp in cnt:
            sel = (rec.split == sp).values
            cnt[sp]["tiles"] += int(sel.sum())
            cnt[sp]["valid_pixels"] += int(M[sel].sum())
            cnt[sp]["flood_pixels"] += int(((Y[sel] > THR) & M[sel]).sum())

        ti, r_, c_ = np.nonzero(M)
        for c in range(C):
            v = X[ti, c, r_, c_]
            ok = np.isfinite(v)
            nonfinite[c] += int((~ok).sum())
            v = v[ok].astype("float64")
            vmin[c] = min(vmin[c], v.min()); vmax[c] = max(vmax[c], v.max())
            s1[c] += v.sum(); s2[c] += (v * v).sum()
            res[c].append(rng.choice(v, min(40_000, v.size), replace=False))
        nvalid += ti.size
        lv = Y[ti, r_, c_]
        label_res.append(rng.choice(lv, min(40_000, lv.size), replace=False))

        take = rng.random(ti.size) < p
        tt, rr, cc = ti[take], r_[take], c_[take]
        fr = pd.DataFrame(X[tt, :, rr, cc], columns=names)
        meta = rec.iloc[tt]
        fr.insert(0, "tile_id", meta.tile_id.values)
        fr.insert(1, "event", meta.event.astype(str).values)
        fr.insert(2, "is_quiet_control", meta.is_quiet_control.values)
        fr.insert(3, "huc4", meta.huc4.values)
        fr.insert(4, "split", meta.split.values)
        fr.insert(5, "px_row", rr)
        fr.insert(6, "px_col", cc)
        fr["coh_drop"] = Y[tt, rr, cc]
        fr["flood_label"] = (Y[tt, rr, cc] > THR).astype(int)
        rows.append(fr)

        for _, pk in picks[picks.set == d].iterrows():
            if sh * SHARD <= pk.pos < sh * SHARD + len(X):
                i = pk.pos - sh * SHARD
                tile_x[pk.tile_id], tile_y[pk.tile_id], tile_m[pk.tile_id] = X[i].copy(), Y[i].copy(), M[i].copy()
        print(f"{k + 1} of {len(shards)} shards done, {len(shards) - k - 1} left", flush=True)

    # feature dictionary
    pool = [np.concatenate(r) for r in res]
    mean = s1 / (nvalid - nonfinite); std = np.sqrt(np.maximum(s2 / (nvalid - nonfinite) - mean ** 2, 0))
    fd = []
    for i, n in enumerate(names):
        q = np.percentile(pool[i], [1, 50, 99])
        fam, unit, desc = FEATURES[n]
        fd.append({"index": i, "name": n, "family": fam, "unit": unit,
                   "log1p_at_load": n in LOG, "min": vmin[i], "p01": q[0], "median": q[1], "p99": q[2],
                   "max": vmax[i], "mean_raw": mean[i], "std_raw": std[i], "non_finite_in_mask": nonfinite[i],
                   "model_mean_after_log": float(ck["mean"][i]), "model_std_after_log": float(ck["std"][i]),
                   "meaning": desc})
    pd.DataFrame(fd).to_csv(OUT / "feature_dictionary.csv", index=False, float_format="%.6g")

    # pixels
    px = pd.concat(rows, ignore_index=True).sample(frac=1, random_state=0)
    px.to_csv(OUT / "pixels_sample.csv", index=False, float_format="%.6g")

    # tiles
    picks = picks[picks.tile_id.isin(tile_x)].reset_index(drop=True)
    order = list(picks.tile_id)
    np.savez_compressed(OUT / "tiles_sample.npz",
                        X=np.stack([tile_x[t] for t in order]),
                        y=np.stack([tile_y[t] for t in order]),
                        mask=np.stack([tile_m[t] for t in order]),
                        tile_id=np.array(order), channels=np.array(names))
    keep = ["tile_id", "event", "huc4", "huc8", "split", "is_quiet_control", "row", "col",
            "valid_frac", "flood_frac_0.15", "has_flood", "dem_clean"]
    picks[keep].to_csv(OUT / "tiles_sample_index.csv", index=False)
    idx[keep].to_csv(OUT / "tile_index_full.csv", index=False)

    # one label as a plain grid
    ex = picks[picks.split.eq("train") & picks["flood_frac_0.15"].between(.08, .5)].iloc[0].tile_id
    y, m = tile_y[ex], tile_m[ex]
    pd.DataFrame(np.where(m, y, np.nan)).to_csv(OUT / "label_grid_coh_drop.csv", index=False, header=False, float_format="%.4f", na_rep="")
    b = pd.DataFrame(np.where(m, (y > THR).astype(float), np.nan))
    b.to_csv(OUT / "label_grid_flood_binary.csv", index=False, header=False, float_format="%.0f", na_rep="")

    # photo labels
    d = "tiles_trace_checked_v1"
    zp = np.load(ROOT / "data" / d / "shared/shard-00000.npz")
    ip = pd.read_csv(ROOT / "data" / d / "index.csv")
    ip["tile_id"] = ip.event.astype(str) + "_" + ip.row.astype(str) + "_" + ip.col.astype(str)
    frac = ((zp["y"] > THR) & zp["m"]).sum((1, 2)) / zp["m"].sum((1, 2)).clip(1)
    top = np.argsort(-frac)[:3]
    np.savez_compressed(OUT / "photo_labels_sample.npz", y=zp["y"][top].astype("float32"), mask=zp["m"][top],
                        tile_id=ip.tile_id.iloc[top].to_numpy().astype(str))

    # summary
    q = np.percentile(np.concatenate(label_res), [1, 5, 25, 50, 75, 95, 99])
    summ = {"tile_px": 112, "pixel_m": 20, "tile_km": 2.24, "features": 30,
            "flood_threshold_on_coh_drop": THR,
            "flood_day_tiles": int((~idx.is_quiet_control).sum()),
            "quiet_control_tiles": int(idx.is_quiet_control.sum()),
            "flood_events_with_tiles": int(idx[~idx.is_quiet_control].event.nunique()),
            "quiet_dates_with_tiles": int(idx[idx.is_quiet_control].event.nunique()),
            "valid_pixels_total": int(nvalid),
            "splits": {s: {**v, "flood_share": round(v["flood_pixels"] / v["valid_pixels"], 4),
                           "huc4": sorted(idx[idx.split == s].huc4.astype(str).unique())} for s, v in cnt.items()},
            "coh_drop_percentiles_p1_p5_p25_p50_p75_p95_p99": [round(float(v), 4) for v in q],
            "label_grid_tile": ex, "pixels_sample_rows": int(len(px)),
            "pixels_sample_flood_share": round(float(px.flood_label.mean()), 4)}
    (OUT / "dataset_summary.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
