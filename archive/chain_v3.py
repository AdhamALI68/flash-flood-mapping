#!/usr/bin/env python3
"""
The three fixes, then re-tile, then measure. Unattended.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u chain_v3.py >> data/chain_v3.log 2>&1 </dev/null &

WHAT CHANGES FROM tiles_v2
  1. REPAIRED HAND. basin_hand was built against a stale stream raster in Death
     Valley, Mojave and Whitewater River - 9.4% of Mojave had any value at all.
     It is the strongest single feature we have (0.5546), and because the valid
     mask is an AND, its holes deleted whole regions: 1810 gave zero tiles from
     32 events, 1809 gave tiles from 54 of 113.
  2. CATCHMENT-ROUTED RAINFALL joins the tiles. Point rainfall is not what
     floods a wash; rain routed downhill over the upstream catchment is.
  3. THE 174 QUIET CONTROLS get tiled. Every tile so far came from a flood
     event, so there was nothing to measure a false alarm against.

Nothing here touches data/tiles_v2. If this run is worse, that one still stands.
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
STATE = DATA / "chain_v3_state.json"
TILES = "data/tiles_v3"
TILES_NEG = "data/tiles_neg"
SHARED = "basin_v2,rain,soil,landcover,rain_routed"


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


def check_tiles(sub: str, expect_flood: bool):
    """Guard the label, every time. This is the check that was missing.

    coh_drop is a difference of two coherences and CANNOT leave -1..+1. Tiling
    the wrong band is what cost us the whole first pass, so it gets asserted
    rather than assumed.
    """
    def post() -> tuple[bool, str]:
        import numpy as np
        import pandas as pd
        d = DATA / Path(sub).name
        idx = d / "index.csv"
        if not idx.exists():
            return False, "no index.csv"
        ix = pd.read_csv(idx)
        sh = sorted((d / "shared").glob("shard-*.npz"))
        if not sh or len(ix) == 0:
            return False, "no shards"
        z = np.load(sh[0])
        y = z["y"][z["m"]]
        lo, hi = float(y.min()), float(y.max())
        if lo < -1.001 or hi > 1.001:
            return False, f"LABEL OUT OF RANGE {lo:.2f}..{hi:.2f}"
        frac = float(ix["flood_frac_0.15"].mean())
        note = (f"{len(ix)} tiles, {ix.event.nunique()} events, "
                f"{ix.huc4.nunique()} regions, mean flood frac {frac:.3f}, "
                f"label {lo:.3f}..{hi:.3f}")
        return True, note
    return post


def main() -> int:
    log("chain v3 started")

    # 1. repaired HAND into the event basin stacks
    if not run("basin_v2_events",
               [PY, "-u", "phase5/rebuild_basin_stack.py"], "rebuild_basin.log"):
        return 1
    if not run("basin_v2_neg",
               [PY, "-u", "phase5/rebuild_basin_stack.py",
                "--features", "data/features_neg"], "rebuild_basin.log"):
        return 1

    # 2. routed rainfall for the controls (events were done in chain v2)
    run("routed_neg",
        [PY, "-u", "phase5/add_routed_to_stacks.py",
         "--features", "data/features_neg", "--routed", "data/rain_routed_neg"],
        "add_routed.log")

    # 3. re-tile the flood events
    if not run("tiles_v3",
               [PY, "-u", "phase5/build_tiles.py", "--out", TILES,
                "--shared", SHARED],
               "build_tiles_v3.log", post=check_tiles(TILES, True)):
        log("ABORT: event tiling failed")
        return 1

    # 4. tile the quiet controls - the negatives we have never had
    run("tiles_neg",
        [PY, "-u", "phase5/build_tiles.py", "--features", "data/features_neg",
         "--out", TILES_NEG, "--shared", SHARED],
        "build_tiles_neg.log", post=check_tiles(TILES_NEG, False))

    # 5. measure
    run("baselines_v3",
        [PY, "-u", "phase6/baselines.py", "--tiles", TILES, "--rung", "10"],
        "baselines_v3.log")
    run("scale_tile_v3",
        [PY, "-u", "phase6/scale_and_tile_eval.py", "--tiles", TILES, "--rung", "10"],
        "scale_tile_v3.log")

    log("chain v3 finished")
    for k, v in load().items():
        if isinstance(v, dict):
            log(f"  {'OK  ' if v.get('done') else 'FAIL'} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
