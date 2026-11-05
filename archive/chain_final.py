#!/usr/bin/env python3
"""
The queue: fusion label -> retrain -> audit -> maps -> cross validation.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u chain_final.py >> data/chain_final.log 2>&1 </dev/null &

Runs strictly one at a time. Everything here wants the GPU, and two training
jobs sharing a 3090 slow each other down and make the timings meaningless.

STAGES
  1  tile on the FUSION label (coherence loss AND amplitude darkening)
     Measured against the 174 paired quiet dates, fusion halves the label's own
     false-alarm rate, 3.55% -> 1.98%, at the cost of tiles where amplitude is
     missing. Amplitude is only ~84% complete, so that cost is real and is
     reported rather than hidden.
  2  train on it, selecting the best epoch on PIXEL AUC, which is the research
     question, rather than on within-tile AUC which we no longer report
  3  audit - 7 checks that must all pass before any number is quotable
  4  example maps - truth beside prediction, chosen best/typical/worst/quiet
  5  restart the six-region cross validation (resumes from the 8 folds already
     saved, so nothing is recomputed)

EVERY STAGE IS SKIPPED IF ITS OUTPUT EXISTS, so this is safe to restart.
Nothing here deletes anything: the coh_drop tiles stay exactly where they are,
so if fusion turns out worse we simply keep the old result.
"""

from __future__ import annotations

import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PY = str(ROOT / ".venv/bin/python")
STATE = DATA / "chain_final_state.json"

SHARED = "basin_v2,rain,soil,landcover,rain_routed"
GROUPS = "terrain,basin,rain,rain_routed,soil,landcover"
TAG = "fusion_pix"


def log(m: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def load() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def run(name: str, argv: list[str], logname: str, post=None) -> bool:
    st = load()
    if st.get(name, {}).get("done"):
        log(f"SKIP  {name}")
        return True
    if post is not None:
        ok, why = post()
        if ok:
            log(f"SKIP  {name} - output already present ({why})")
            st[name] = {"done": True, "note": "pre-existing"}
            save(st)
            return True
    log(f"START {name} -> data/{logname}")
    t0 = time.time()
    with (DATA / logname).open("a") as fh:
        fh.write(f"\n=== {datetime.now()} {name} ===\n")
        fh.flush()
        rc = subprocess.call(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
    mins = (time.time() - t0) / 60
    if rc != 0:
        log(f"FAIL  {name} rc={rc} after {mins:.1f} min")
        st[name] = {"done": False, "error": f"rc={rc}"}
        save(st)
        return False
    if post is not None:
        ok, why = post()
        if not ok:
            log(f"FAIL  {name}: exited 0 but no usable output ({why})")
            st[name] = {"done": False, "error": why}
            save(st)
            return False
        log(f"DONE  {name} in {mins:.1f} min - {why}")
    else:
        log(f"DONE  {name} in {mins:.1f} min")
    st[name] = {"done": True, "at": str(datetime.now()), "minutes": round(mins, 1)}
    save(st)
    return True


def tiles_ok(sub: str):
    """The label guard. Tiling the wrong band cost this project a full pass."""
    def post():
        import numpy as np
        import pandas as pd
        d = DATA / sub
        if not (d / "index.csv").exists():
            return False, "no index.csv"
        ix = pd.read_csv(d / "index.csv")
        sh = sorted((d / "shared").glob("shard-*.npz"))
        if not sh or len(ix) == 0:
            return False, "no shards"
        z = np.load(sh[0])
        y = z["y"][z["m"]]
        lo, hi = float(y.min()), float(y.max())
        if lo < -1.001 or hi > 1.001:
            return False, f"LABEL OUT OF RANGE {lo:.2f}..{hi:.2f}"
        return True, (f"{len(ix)} tiles, {ix.event.nunique()} events, "
                      f"{ix.huc4.nunique()} regions, label {lo:.3f}..{hi:.3f}")
    return post


def main() -> int:
    log("final chain started")

    # 1. tile on the fusion label
    if not run("tiles_fusion",
               [PY, "-u", "phase5/build_tiles.py", "--out", "data/tiles_fusion",
                "--shared", SHARED, "--label", "fusion"],
               "build_tiles_fusion.log", post=tiles_ok("tiles_fusion")):
        log("ABORT: fusion tiling failed")
        return 1
    if not run("tiles_fusion_neg",
               [PY, "-u", "phase5/build_tiles.py", "--features",
                "data/features_neg", "--out", "data/tiles_neg_fusion",
                "--shared", SHARED, "--label", "fusion"],
               "build_tiles_fusion.log", post=tiles_ok("tiles_neg_fusion")):
        log("ABORT: fusion control tiling failed")
        return 1

    # 2. train on it
    run("train_fusion",
        [PY, "-u", "phase6/train.py",
         "--tiles", "data/tiles_fusion", "--neg", "data/tiles_neg_fusion",
         "--rung", "10", "--groups", GROUPS,
         "--test-regions", "1606,1809", "--val-regions", "1507",
         "--epochs", "100", "--patience", "15",
         "--select-on", "pixel", "--rank-weight", "0", "--prior", "downhill",
         "--tag", TAG],
        "train_fusion.log")

    # 3. audit, then the pixel-level scoring, for whichever model exists
    for tag in (TAG, "terrain+basin+rain+rain_routed+soil+landcover"):
        if (DATA / f"model_{tag}.pt").exists():
            run(f"audit_{tag}", [PY, "-u", "phase6/audit_pixel.py", tag],
                f"audit_{tag}.log")
            run(f"pixeleval_{tag}",
                [PY, "-u", "phase6/eval_pixel_full.py", tag],
                f"pixeleval_{tag}.log")

    # 4. example maps - truth beside prediction
    for tag, tiles, neg in ((TAG, "data/tiles_fusion", "data/tiles_neg_fusion"),
                            ("terrain+basin+rain+rain_routed+soil+landcover",
                             "data/tiles_v4", "data/tiles_neg_v4")):
        if (DATA / f"model_{tag}.pt").exists():
            run(f"maps_{tag}",
                [PY, "-u", "phase6/make_maps.py", "--tag", tag,
                 "--tiles", tiles, "--neg", neg],
                f"maps_{tag}.log")

    # 5. finish the cross validation (resumes; 8 folds are already on disk)
    run("cross_region",
        [PY, "-u", "phase8/cross_region.py"], "cross_region_driver.log")

    log("final chain finished")
    for k, v in load().items():
        if isinstance(v, dict):
            log(f"  {'OK  ' if v.get('done') else 'FAIL'} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
