#!/usr/bin/env python3
"""
Re-tile with the CORRECT label, then re-measure. Unattended.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u chain_v2.py >> data/chain_v2.log 2>&1 </dev/null &

WHY THIS RUN EXISTS
    build_tiles.py took band 0 of label.tif and called it coh_drop. Band 0 is
    amp_change_db. Every tile, baseline and evaluation so far was built against
    radar brightness change in decibels instead of the flood label. That is the
    single reason every baseline sat at chance while the label itself validated
    at 86.8%.

    build_tiles.py now selects the band by name and refuses to run without it.
    Everything downstream of tiling has to be rebuilt.

    Output goes to data/tiles_v2, NOT over data/tiles. The old tiles are wrong,
    but they are the evidence for what went wrong and they get deleted on
    Adham's word, not mine.

The routed rainfall joins in the same pass, but only if every event has it.
A partial stack would silently drop events from the tile set, which is how we
lost 120 events last time without noticing.
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
STATE = DATA / "chain_v2_state.json"
TILES = "data/tiles_v2"


def log(m: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def running(pat: str) -> bool:
    try:
        return bool(subprocess.run(["pgrep", "-f", pat], capture_output=True,
                                   text=True, timeout=10).stdout.strip())
    except Exception:
        return False


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


def n_events() -> int:
    idx = DATA / "features/_index.csv"
    return sum(1 for _ in idx.read_text().splitlines()) - 1 if idx.exists() else 0


def tiles_ok() -> tuple[bool, str]:
    """The label check that should have existed the first time.

    coh_drop is a difference of two coherences, so it CANNOT leave -1..+1.
    If it does, we are tiling the wrong band again and the run is worthless.
    """
    import numpy as np
    import pandas as pd
    d = DATA / "tiles_v2"
    idx = d / "index.csv"
    if not idx.exists():
        return False, "no index.csv"
    n = len(pd.read_csv(idx))
    sh = sorted((d / "shared").glob("shard-*.npz"))
    if not sh or n == 0:
        return False, "no shards"
    z = np.load(sh[0])
    y = z["y"][z["m"]]
    lo, hi = float(y.min()), float(y.max())
    if lo < -1.001 or hi > 1.001:
        return False, (f"LABEL OUT OF RANGE {lo:.2f}..{hi:.2f} - "
                       f"still tiling the wrong band")
    return True, f"{n} tiles, {len(sh)} shards, label range {lo:.3f}..{hi:.3f}"


def main() -> int:
    log("chain v2 started")

    # 1. let the rainfall routing finish. Never interrupt it.
    if running("route_rainfall.py"):
        log("catchment rainfall routing is still running - waiting")
        last = -1
        while running("route_rainfall.py"):
            n = len(list((DATA / "rain_routed").glob("*.tif")))
            if n != last:
                log(f"  routed {n} events")
                last = n
            time.sleep(60)
        log("routing finished")

    # 2. sample the routed rainfall onto each event grid
    run("add_routed", [PY, "-u", "phase5/add_routed_to_stacks.py"],
        "add_routed.log")

    # 3. include it only if EVERY event has it
    have = len(list((DATA / "features").glob("*/rain_routed.tif")))
    total = n_events()
    shared = "basin,rain,soil,landcover"
    if total and have == total:
        shared += ",rain_routed"
        log(f"routed rainfall on all {have}/{total} events - including it")
    else:
        log(f"routed rainfall on {have}/{total} events - LEAVING IT OUT so no "
            f"event gets silently dropped. It can go in on a later pass.")
    st = load()
    st["shared_stacks"] = shared
    save(st)

    # 4. re-tile with the corrected label
    if not run("tiles_v2",
               [PY, "-u", "phase5/build_tiles.py", "--out", TILES,
                "--shared", shared],
               "build_tiles_v2.log", post=tiles_ok):
        log("ABORT: tiling failed or the label is still out of range")
        return 1

    # 5. the moment of truth
    run("baselines_v2",
        [PY, "-u", "phase6/baselines.py", "--tiles", TILES, "--rung", "10"],
        "baselines_v2.log")
    run("scale_tile_v2",
        [PY, "-u", "phase6/scale_and_tile_eval.py", "--tiles", TILES, "--rung", "10"],
        "scale_tile_v2.log")

    log("chain v2 finished")
    for k, v in load().items():
        if isinstance(v, dict):
            log(f"  {'OK  ' if v.get('done') else 'FAIL'} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
