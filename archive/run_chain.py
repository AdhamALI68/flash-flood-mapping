#!/usr/bin/env python3
"""
The compute chain. Start once, runs unattended, resumable at every stage.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u run_chain.py >> data/chain.log 2>&1 </dev/null &

Stages run in dependency order. Each one:
  - CHECKS ITS INPUTS FIRST and refuses to run without them, rather than
    producing a half-built output that looks finished
  - CHECKS ITS OUTPUTS AFTER and only then writes a completion marker
  - is skipped on restart if its marker exists

Why the input checks matter more than they look: basin routing on an incomplete
DEM mosaic does not fail, it silently truncates drainage area at the hole - the
exact fault worth +0.073 that basin-wide routing exists to fix. A stage that
cannot verify its inputs is a stage that can quietly poison everything after it.

Markers live in data/chain_state.json. Delete an entry to force a re-run.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
PY = str(ROOT / ".venv/bin/python")
STATE = DATA / "chain_state.json"
RUNGS = [1, 2, 5, 10, 30]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{stamp()}] {msg}", flush=True)


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


# --------------------------------------------------------------- checks
def dem10_ready() -> tuple[bool, str]:
    """All 36 tiles, or every remaining one genuinely unpublished."""
    sp = DATA / "dem10_state.json"
    if not sp.exists():
        return False, "dem10_state.json missing"
    st = json.loads(sp.read_text())
    n_ok = sum(1 for v in st.values() if v == "ok")
    n_np = sum(1 for v in st.values() if "not published" in str(v))
    n_fail = sum(1 for v in st.values() if str(v).startswith("failed"))
    tifs = len(list((DATA / "dem10_routing").glob("*.tif")))
    if n_fail:
        return False, f"{n_fail} tiles still failed ({n_ok} ok, {tifs} tifs)"
    if n_ok + n_np < 36:
        return False, f"only {n_ok + n_np}/36 tiles resolved"
    return True, f"{n_ok} tiles ok, {n_np} unpublished, {tifs} on disk"


def routing_ready() -> tuple[bool, str]:
    ok, why = dem10_ready()
    return (ok, why) if not ok else (True, why)


def routing_done() -> tuple[bool, str]:
    sp = DATA / "basin_routing_state.json"
    if not sp.exists():
        return False, "no routing state"
    st = json.loads(sp.read_text())
    done = [k for k, v in st.items() if v.get("done")]
    # Every routed basin must actually have its rasters on disk.
    missing = []
    for h in done:
        d = DATA / "basin_features" / h
        for nm in ("hand", "twi", "spi", "facc", "slope"):
            if not (d / f"{nm}.tif").exists():
                missing.append(f"{h}/{nm}")
    if missing:
        return False, f"state claims done but files missing: {missing[:3]}"
    return len(done) >= 11, f"{len(done)}/11 basins routed"


def features_ready() -> tuple[bool, str]:
    ok, why = routing_done()
    if not ok:
        return False, f"routing incomplete: {why}"
    for p, name in ((DATA / "dem_download_state.json", "1 m DEM state"),
                    (DATA / "rain", "rain"), (DATA / "soil", "soil"),
                    (DATA / "landcover/worldcover", "land cover"),
                    (DATA / "dataset_final.csv", "dataset_final.csv")):
        if not p.exists():
            return False, f"{name} missing"
    return True, "inputs present"


def features_done(sub: str = "features") -> tuple[bool, str]:
    idx = DATA / sub / "_index.csv"
    if not idx.exists():
        return False, "no _index.csv"
    n = sum(1 for _ in idx.read_text().splitlines()) - 1
    return n > 0, f"{n} events with feature stacks"


def tiles_done() -> tuple[bool, str]:
    """One index, shared shards, and a shard set per rung."""
    d = DATA / "tiles"
    idx = d / "index.csv"
    if not idx.exists():
        return False, "no index.csv"
    n = sum(1 for _ in idx.read_text().splitlines()) - 1
    sh = len(list((d / "shared").glob("shard-*.npz")))
    missing = [r for r in RUNGS
               if not list((d / f"rung{r}m").glob("shard-*.npz"))]
    if missing:
        return False, f"rungs with no shards: {missing}"
    if sh == 0:
        return False, "no shared shards"
    return n > 0, f"{n} tiles, {sh} shared shards, all {len(RUNGS)} rungs"


# --------------------------------------------------------------- runner
def run(name: str, argv: list[str], logname: str,
        pre=None, post=None, retries: int = 2) -> bool:
    st = load_state()
    if st.get(name, {}).get("done"):
        log(f"SKIP  {name} (already done)")
        return True

    if pre is not None:
        ok, why = pre()
        if not ok:
            log(f"BLOCKED {name}: {why}")
            return False
        log(f"       {name} inputs: {why}")

    logf = DATA / logname
    rc = -1
    for attempt in range(1, retries + 1):
        log(f"START {name} (attempt {attempt}/{retries}) -> data/{logname}")
        t0 = time.time()
        with logf.open("a") as fh:
            fh.write(f"\n=== {stamp()} {name} attempt {attempt} ===\n")
            fh.flush()
            rc = subprocess.call(argv, cwd=ROOT, stdout=fh,
                                 stderr=subprocess.STDOUT)
        mins = (time.time() - t0) / 60
        if rc != 0:
            log(f"FAIL  {name} rc={rc} after {mins:.1f} min")
            if attempt < retries:
                time.sleep(60)
            continue
        if post is not None:
            ok, why = post()
            if not ok:
                # Exit 0 but no usable output is the dangerous case - a stage
                # that "succeeded" while producing nothing would let the next
                # stage run on emptiness.
                log(f"FAIL  {name}: exited 0 but output check failed: {why}")
                if attempt < retries:
                    time.sleep(60)
                continue
            log(f"DONE  {name} in {mins:.1f} min — {why}")
        else:
            log(f"DONE  {name} in {mins:.1f} min")
        st = load_state()
        st[name] = {"done": True, "at": stamp(), "minutes": round(mins, 1)}
        save_state(st)
        return True

    st = load_state()
    st[name] = {"done": False, "at": stamp(), "error": f"rc={rc}"}
    save_state(st)
    return False


def wait_for(check, label: str, timeout_min: int = 180) -> bool:
    """Poll until a prerequisite is satisfied, rather than failing immediately."""
    deadline = time.time() + timeout_min * 60
    while time.time() < deadline:
        ok, why = check()
        if ok:
            log(f"READY {label}: {why}")
            return True
        log(f"WAIT  {label}: {why}")
        time.sleep(120)
    log(f"TIMEOUT waiting for {label}")
    return False


def main() -> int:
    log("chain started")
    (DATA).mkdir(exist_ok=True)

    # 0. the DEM tiles must be complete before routing can be trusted
    if not wait_for(dem10_ready, "10 m DEM tiles", timeout_min=240):
        log("ABORT: cannot route without a complete DEM mosaic")
        return 1

    # 1. flow routing, per basin
    if not run("basin_routing", [PY, "-u", "phase5/basin_routing.py"],
               "basin_routing.log", pre=routing_ready, post=routing_done):
        log("ABORT: routing failed; features would be built on truncated drainage")
        return 1

    # 2. feature stacks — events, then the quiet controls
    if not run("features_events",
               [PY, "-u", "phase5/build_features.py"],
               "build_features.log", pre=features_ready,
               post=lambda: features_done("features")):
        log("ABORT: feature build failed")
        return 1

    run("features_negatives",
        [PY, "-u", "phase5/build_features.py", "--negatives",
         "--out", "data/features_neg"],
        "build_features_neg.log",
        post=lambda: features_done("features_neg"))

    # 3. tiles - ONE pass writes shared/ plus every rung, so the 16 channels
    #    that do not vary by rung are stored once instead of five times, and
    #    each event is read once instead of five times.
    run("tiles_all_rungs",
        [PY, "-u", "phase5/build_tiles.py"],
        "build_tiles.log", post=tiles_done)

    # 4. baselines on the middle rung — the number a model must beat
    run("baselines_10m",
        [PY, "-u", "phase6/baselines.py", "--tiles", "data/tiles", "--rung", "10"],
        "baselines.log")

    st = load_state()
    log("chain finished")
    for k, v in st.items():
        log(f"  {'OK  ' if v.get('done') else 'FAIL'} {k}"
            + (f" ({v['minutes']} min)" if v.get("minutes") else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
