#!/usr/bin/env python3
"""
Phase 4 - download the looks pilot and measure whether the label got cleaner.

    ./.venv/bin/python -u phase4/looks_collect.py

THE MEASUREMENT
    Same scene pairs, same ground, three processings:

        5x1  looks (N=5)   20 m   <- what the whole project used
       10x2  looks (N=20)  40 m
       20x4  looks (N=80)  80 m

    Coherence from N looks carries noise ~ (1-g^2)/sqrt(2N), so the drop, being
    a difference of two estimates, carries sqrt(2) more. If the theory holds,
    that noise should fall by about 2x from 5x1 to 10x2 and 4x to 20x4.

    Two things are reported per setting:
      1  the measured noise on the drop, and what our 0.15 threshold is worth
         in sigma - at 5x1 it is 1.6 sigma and a 3 sigma cut fires on 0.00%
      2  agreement with the INDEPENDENT Sentinel-2 optical flood label, for the
         events that have one. That is the real test: a cleaner label should
         agree better with a different instrument looking at the same flood.

Output: data/looks_pilot_result.json
"""
from __future__ import annotations
import json, sys, zipfile, shutil
from pathlib import Path
import numpy as np, rasterio
from rasterio.warp import Resampling, reproject

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT / "data/looks_pilot"
STATE = ROOT / "data/looks_pilot_state.json"
LOOKS = ["10x2", "20x4"]


def warp_to(path, grid):
    out = np.full(grid["shape"], np.nan, dtype="float32")
    with rasterio.open(path) as s:
        reproject(source=rasterio.band(s, 1), destination=out,
                  dst_transform=grid["transform"], dst_crs=grid["crs"],
                  src_nodata=s.nodata, dst_nodata=np.nan,
                  resampling=Resampling.bilinear)
    return out


def coh_sigma(g, n):
    g = np.clip(g, 0, 0.999)
    return (1 - g ** 2) / np.sqrt(2 * n)


def main() -> int:
    import hyp3_sdk
    h = hyp3_sdk.HyP3()
    st = json.loads(STATE.read_text())
    WORK.mkdir(parents=True, exist_ok=True)

    # ---- download ------------------------------------------------------
    got = 0
    for eid, d in st["events"].items():
        for lk in LOOKS:
            j = d.get(lk, {})
            if not j.get("jobs") or j.get("downloaded"):
                continue
            dest = WORK / lk / eid
            dest.mkdir(parents=True, exist_ok=True)
            ok = True
            for tag, jid in j["jobs"].items():
                if list(dest.glob(f"{tag}_*_corr.tif")):
                    continue
                try:
                    job = h.get_job_by_id(jid)
                    if job.status_code != "SUCCEEDED":
                        ok = False; break
                    z = job.download_files(dest)[0]
                    with zipfile.ZipFile(z) as zf:
                        for nm in zf.namelist():
                            if nm.endswith("_corr.tif"):
                                zf.extract(nm, dest)
                                src = dest / nm
                                src.rename(dest / f"{tag}_{src.name}")
                    z.unlink()
                    for p in dest.iterdir():
                        if p.is_dir():
                            shutil.rmtree(p, ignore_errors=True)
                except Exception as e:
                    print(f"  {eid} {lk} {tag}: {type(e).__name__} {e}", flush=True)
                    ok = False; break
            if ok:
                j["downloaded"] = True
                got += 1
                if got % 10 == 0:
                    print(f"  downloaded {got} job sets", flush=True)
        STATE.write_text(json.dumps(st, indent=1))
    print(f"downloaded {got} new job sets\n", flush=True)

    # ---- measure -------------------------------------------------------
    opt_dir = ROOT / "data/optical_labels"
    res = {}
    for lk, nlooks in (("5x1", 5), ("10x2", 20), ("20x4", 80)):
        sig, frac3, agree = [], [], []
        for eid in list(st["events"]):
            lab = ROOT / "data/features" / eid / "label.tif"
            if not lab.exists():
                continue
            with rasterio.open(lab) as s:
                nm = [s.descriptions[i] for i in range(s.count)]
                grid = {"crs": s.crs, "transform": s.transform,
                        "shape": (s.height, s.width)}
                if lk == "5x1":
                    base = s.read(nm.index("coh_baseline") + 1)
                    coev = s.read(nm.index("coh_coevent") + 1)
                    drop = s.read(nm.index("coh_drop") + 1)
            if lk != "5x1":
                d = WORK / lk / eid
                b = sorted(d.glob("baseline_*_corr.tif"))
                c = sorted(d.glob("coevent_*_corr.tif"))
                if not b or not c:
                    continue
                base, coev = warp_to(b[0], grid), warp_to(c[0], grid)
                drop = base - coev
            m = np.isfinite(drop) & np.isfinite(base) & np.isfinite(coev)
            if m.sum() < 5000:
                continue
            s_ = np.sqrt(coh_sigma(base[m], nlooks) ** 2
                         + coh_sigma(coev[m], nlooks) ** 2)
            sig.append(float(np.median(s_)))
            with np.errstate(invalid="ignore", divide="ignore"):
                z = drop[m] / s_
            frac3.append(float((z > 3).mean()))
            op = opt_dir / f"{eid}.tif"
            if op.exists():
                with rasterio.open(op) as s:
                    onm = [s.descriptions[i] for i in range(s.count)]
                    of = s.read(onm.index("flood_optical") + 1) > 0.5
                    ov = s.read(onm.index("valid") + 1) > 0.5
                mm = m & ov
                if mm.sum() > 5000 and of[mm].sum() > 20:
                    from sklearn.metrics import roc_auc_score
                    try:
                        agree.append(float(roc_auc_score(of[mm].astype(int),
                                                         drop[mm])))
                    except Exception:
                        pass
        if not sig:
            continue
        res[lk] = {"looks": nlooks, "events": len(sig),
                   "sigma_drop": float(np.median(sig)),
                   "threshold_in_sigma": float(0.15 / np.median(sig)),
                   "frac_above_3sigma": float(np.mean(frac3)),
                   "auc_vs_optical": float(np.mean(agree)) if agree else None,
                   "n_optical": len(agree)}

    print("=" * 84)
    print("DOES MORE MULTILOOKING CLEAN THE LABEL?")
    print("=" * 84)
    print(f"  {'looks':<8}{'N':>5}{'events':>8}{'noise on drop':>15}"
          f"{'0.15 is':>10}{'>3 sigma':>11}{'AUC vs optical':>16}")
    for lk in ("5x1", "10x2", "20x4"):
        r = res.get(lk)
        if not r:
            print(f"  {lk:<8}  (no data)")
            continue
        a = f"{r['auc_vs_optical']:.4f}" if r["auc_vs_optical"] else "-"
        print(f"  {lk:<8}{r['looks']:>5}{r['events']:>8}"
              f"{r['sigma_drop']:>15.4f}{r['threshold_in_sigma']:>9.1f}o"
              f"{100*r['frac_above_3sigma']:>10.2f}%{a:>16}")
    print("\n  'AUC vs optical' = how well the coherence drop itself predicts")
    print("  the independent Sentinel-2 flood map. Higher = cleaner label.")
    json.dump(res, (ROOT / "data/looks_pilot_result.json").open("w"), indent=1)
    print("\nwrote data/looks_pilot_result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
