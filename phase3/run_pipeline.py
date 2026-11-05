#!/usr/bin/env python3
"""
Phase 3, step 2 - submit, download, crop, store. In resumable waves.

For each distinct SCENE TRIPLE, two InSAR jobs at 5x1 looks (20 m): a baseline
pair entirely before the flood, and a co-event pair spanning it. What lands on
disk is the CONTINUOUS coherence and amplitude, cropped to each event's
neighbourhood and aligned onto one grid.

Work is grouped by scene triple, not by event. 617 events resolve to only 470
distinct triples, because a flood reported across several counties produces
several Storm Events records over one burst. Submitting per event would run 344
identical jobs and download 34 GB twice. Grouping cuts 1,234 jobs to 890 and
123 GB to 89 GB.

That collapse is also a fact about the dataset: 470 is the number of independent
flood observations, and the paper should say so rather than claim 617.

No threshold is applied here, and no masking. Phase 4 chooses the flood
threshold by validating against USGS gauges, and re-thresholding continuous
rasters is seconds of arithmetic. Baking a guessed threshold in now would mean
re-running every job to change it.

State lives in data/phase3_state.json - every stage is restartable, so a dropped
connection costs one wave, not the run.

Output: data/phase3/<event>/{baseline,coevent}_{corr,amp}.tif
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyproj
import rasterio

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gate_b"))
from align import read_aligned  # noqa: E402

LOOKS = "5x1"

# Keep the FULL burst extent, not a crop around the reported flood.
#
# The download is the same either way - HyP3 ships one zip and we always fetch
# all of it - so cropping saves disk, not time. Storing full extent costs 27.6 GB
# instead of 4.5 GB, against 341 GB free, and means no later decision about
# context windows can force a repeat of the ~21 hour download.
#
# Set a value in metres to go back to cropping.
AOI_BUFFER_M: int | None = None

STATE = ROOT / "data/phase3_state.json"
OUTDIR = ROOT / "data/phase3"
WORK = ROOT / "data/phase3_work"


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {"groups": {}, "events": {}}


def save_state(st: dict):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def group_key(rec: dict) -> str:
    """Events sharing this key need exactly the same two HyP3 products."""
    b = rec["pairs"]["baseline"]
    c = rec["pairs"]["coevent"]
    return f"{rec['full_burst_id']}|{b[0]}|{b[1]}|{c[1]}"


def group_id(key: str) -> str:
    """Stable short id for a group's storage directory.

    Not built from hash() - Python salts string hashing per process, so that
    would produce a different directory on every run and silently re-download
    work already on disk.
    """
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def aoi_bounds(rec: dict, crs) -> tuple[float, float, float, float] | None:
    """Bounding box around the event, buffered, in the product CRS.

    Returns None when AOI_BUFFER_M is unset, which keeps the whole burst.

    Storm Events gives a begin and an end point which can be tens of km apart
    for a flood that travelled down a wash, so both are included.
    """
    if AOI_BUFFER_M is None:
        return None
    lons = [rec["lon"]] + ([rec["end_lon"]] if rec.get("end_lon") else [])
    lats = [rec["lat"]] + ([rec["end_lat"]] if rec.get("end_lat") else [])
    tf = pyproj.Transformer.from_crs("EPSG:4326", crs, always_xy=True)
    xs, ys = tf.transform(lons, lats)
    return (min(xs) - AOI_BUFFER_M, min(ys) - AOI_BUFFER_M,
            max(xs) + AOI_BUFFER_M, max(ys) + AOI_BUFFER_M)


def product_files(d: Path) -> dict[str, str]:
    out = {}
    for kind in ("corr", "amp"):
        hits = sorted(d.glob(f"*_{kind}.tif"))
        if hits:
            out[kind] = str(hits[0])
    return out


def save_group(rec: dict, dirs: dict[str, Path], dest: Path) -> dict:
    """Keep every raster the product ships, once per scene triple.

    Stored raw and unaligned on purpose. Alignment is a read-time concern -
    read_aligned() puts any set of these on a shared grid whenever they are
    differenced - so writing aligned copies would duplicate ~14 GB for nothing.

    Written per group rather than per event: 147 of the 617 events share a
    triple and would otherwise get byte-identical copies.

    Everything is kept, including the phase layers, which this project has no
    use for: phase carries information only where coherence is high, and a flood
    is coherence collapsing, so phase is noise exactly where the signal is.
    They are kept anyway because deleting later is instant and free, while
    re-downloading is another ~21 hours. Decide once the DEM's storage needs are
    known, not now.
    """
    dest.mkdir(parents=True, exist_ok=True)
    kept = 0
    for tag, d in dirs.items():
        sub = dest / tag
        sub.mkdir(parents=True, exist_ok=True)
        # Every file, no filter. The .kmz and .png files are only renderings of
        # the tifs and cost ~11 MB per product, but keeping everything means no
        # later question about what was dropped can ever require a re-download.
        for f in sorted(d.iterdir()):
            if f.is_file():
                shutil.copy2(f, sub / f.name)
                kept += 1

    fb, fc = product_files(dest / "baseline"), product_files(dest / "coevent")
    missing = [k for k in ("corr", "amp") if k not in fb or k not in fc]
    if missing:
        raise RuntimeError(f"missing rasters {missing}")

    with rasterio.open(fc["corr"]) as s:
        crs = s.crs
    aoi = aoi_bounds(rec, crs)
    (bc, cc), transform, crs = read_aligned([fb["corr"], fc["corr"]], aoi=aoi)

    valid = np.isfinite(bc) & np.isfinite(cc)
    if not valid.any():
        raise RuntimeError("no overlapping valid pixels between the two products")
    return {
        "files_kept": kept,
        "shape": list(bc.shape),
        "valid_px": int(valid.sum()),
        "mean_baseline_coh": round(float(np.nanmean(bc)), 4),
        "mean_coevent_coh": round(float(np.nanmean(cc)), 4),
        "mean_drop": round(float(np.nanmean(bc[valid] - cc[valid])), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", default="data/phase3_bursts.jsonl")
    ap.add_argument("--wave", type=int, default=30, help="scene triples per wave")
    ap.add_argument("--max-waves", type=int, default=0, help="0 = run to completion")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--state", default=None, help="override the state file")
    ap.add_argument("--outdir", default=None, help="override the output directory")
    args = ap.parse_args()

    import hyp3_sdk

    global STATE, OUTDIR, WORK
    if args.state:
        STATE = ROOT / args.state
    if args.outdir:
        OUTDIR = ROOT / args.outdir
        WORK = ROOT / (args.outdir + "_work")

    recs = [json.loads(l) for l in (ROOT / args.bursts).read_text().splitlines() if l.strip()]
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        groups[group_key(r)].append(r)
    print(f"{len(recs)} events -> {len(groups)} distinct scene triples "
          f"-> {2*len(groups)} HyP3 jobs")

    st = load_state()
    st.setdefault("groups", {})
    st.setdefault("events", {})
    h = hyp3_sdk.HyP3()
    OUTDIR.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)

    waves = 0
    while True:
        pending = [k for k in groups if st["groups"].get(k, {}).get("stage") != "done"]
        if not pending:
            print("\nall scene triples processed")
            break
        if args.max_waves and waves >= args.max_waves:
            print(f"\nstopping after {waves} waves ({len(pending)} triples remain)")
            break

        batch = pending[: args.wave]
        waves += 1
        credits = h.my_info()["remaining_credits"]
        n_ev = sum(len(groups[k]) for k in batch)
        print(f"\n=== wave {waves}: {len(batch)} triples / {n_ev} events "
              f"({len(pending)} triples left) | credits {credits} ===", flush=True)

        # ---- submit -----------------------------------------------------
        for k in batch:
            cur = st["groups"].setdefault(k, {"stage": "new"})
            if cur.get("jobs"):
                continue
            rec = groups[k][0]
            jobs = {}
            try:
                for tag, (ref, sec) in rec["pairs"].items():
                    b = h.submit_insar_isce_multi_burst_job(
                        reference=[ref], secondary=[sec],
                        name=f"p3-{rec['EVENT_ID']}-{tag}",
                        apply_water_mask=False, looks=LOOKS)
                    jobs[tag] = b.jobs[0].job_id
            except Exception as ex:
                cur.update(stage="submit_failed", error=f"{type(ex).__name__}: {ex}")
                print(f"  submit failed {k[:40]}: {ex}", flush=True)
                continue
            cur.update(stage="submitted", jobs=jobs)
        save_state(st)

        # ---- wait -------------------------------------------------------
        while True:
            states = []
            for k in batch:
                for jid in (st["groups"].get(k, {}).get("jobs") or {}).values():
                    states.append(h.get_job_by_id(jid).status_code)
            if not states:
                break
            n_done = sum(s in ("SUCCEEDED", "FAILED") for s in states)
            print(f"    {n_done}/{len(states)} jobs finished", flush=True)
            if n_done == len(states):
                break
            time.sleep(args.poll)

        # ---- download once, crop for every event sharing it -------------
        for k in batch:
            cur = st["groups"].get(k, {})
            if cur.get("stage") in ("done", "submit_failed"):
                continue
            work = WORK / group_id(k)
            dirs = {}
            try:
                for tag, jid in cur["jobs"].items():
                    job = h.get_job_by_id(jid)
                    if job.status_code != "SUCCEEDED":
                        raise RuntimeError(f"{tag} job {job.status_code}")
                    d = work / tag
                    d.mkdir(parents=True, exist_ok=True)
                    for z in job.download_files(d):
                        with zipfile.ZipFile(z) as zf:
                            zf.extractall(d)
                        z.unlink()
                    dirs[tag] = next(p for p in d.iterdir() if p.is_dir())

                gid = group_id(k)
                stats = save_group(groups[k][0], dirs, OUTDIR / "groups" / gid)
                for rec in groups[k]:
                    st["events"][str(rec["EVENT_ID"])] = {
                        "stage": "done", "group_id": gid,
                        "lat": rec["lat"], "lon": rec["lon"],
                        "end_lat": rec.get("end_lat"), "end_lon": rec.get("end_lon"),
                    }
                cur.update(stage="done", group_id=gid, events=len(groups[k]), **stats)
                print(f"  {gid}  {len(groups[k])} events  {stats['shape']}  "
                      f"drop={stats['mean_drop']:+.4f}", flush=True)
            except Exception as ex:
                cur.update(stage="failed", error=f"{type(ex).__name__}: {ex}")
                print(f"  group FAILED {k[:34]}: {ex}", flush=True)
            finally:
                shutil.rmtree(work, ignore_errors=True)
                save_state(st)

    ev_done = sum(1 for v in st["events"].values() if v.get("stage") == "done")
    ev_fail = sum(1 for v in st["events"].values() if v.get("stage") == "failed")
    g_fail = sum(1 for v in st["groups"].values() if v.get("stage", "").endswith("failed"))
    print(f"\nevents done {ev_done}, failed {ev_fail} of {len(recs)}"
          f"   |   groups failed {g_fail} of {len(groups)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
