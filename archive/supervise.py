#!/usr/bin/env python3
"""
Waits for the running feature build, then finishes the pipeline. Unattended.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u supervise.py >> data/supervise.log 2>&1 </dev/null &

The feature build is currently running ORPHANED - the chain that launched it was
killed. It cannot be restarted without losing hours, because build_features has
no resume logic. So this supervises from outside: it watches for that process to
exit, verifies its output, then runs the remaining stages itself.

It never touches the running job. Watching a PID and checking files is the only
safe way to chain onto something already in flight.

Stages after features:
  1. feature stacks for the 174 quiet controls
  2. tiling - one pass writing shared/ + all 5 rungs
  3. baselines - the number a model has to beat

Every stage is skipped if its output already exists, so this is safe to restart.
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
STATE = DATA / "supervise_state.json"


def log(m: str) -> None:
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {m}", flush=True)


def feature_build_running() -> bool:
    try:
        out = subprocess.run(["pgrep", "-f", "build_features.py"],
                             capture_output=True, text=True, timeout=10).stdout
        return bool(out.strip())
    except Exception:
        return False


def n_feature_dirs(sub: str) -> int:
    d = DATA / sub
    return len([p for p in d.glob("*") if p.is_dir()]) if d.exists() else 0


def features_ok(sub: str) -> tuple[bool, str]:
    """_index.csv is written only at the END, so it is the completion signal."""
    idx = DATA / sub / "_index.csv"
    dirs = n_feature_dirs(sub)
    if not idx.exists():
        return False, f"{dirs} event dirs but no _index.csv (incomplete)"
    n = sum(1 for _ in idx.read_text().splitlines()) - 1
    return n > 0, f"{n} events indexed, {dirs} dirs"


def tiles_ok() -> tuple[bool, str]:
    idx = DATA / "tiles/index.csv"
    if not idx.exists():
        return False, "no index.csv"
    n = sum(1 for _ in idx.read_text().splitlines()) - 1
    sh = len(list((DATA / "tiles/shared").glob("shard-*.npz")))
    missing = [r for r in (1, 2, 5, 10, 30)
               if not list((DATA / f"tiles/rung{r}m").glob("shard-*.npz"))]
    if missing:
        return False, f"rungs missing shards: {missing}"
    return n > 0 and sh > 0, f"{n} tiles, {sh} shared shards"


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
            log(f"SKIP  {name} — output already present ({why})")
            st[name] = {"done": True, "at": str(datetime.now()), "note": "pre-existing"}
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
            st[name] = {"done": False, "error": f"no output: {why}"}
            save(st)
            return False
        log(f"DONE  {name} in {mins:.1f} min — {why}")
    else:
        log(f"DONE  {name} in {mins:.1f} min")
    st[name] = {"done": True, "at": str(datetime.now()), "minutes": round(mins, 1)}
    save(st)
    return True


def main() -> int:
    log("supervisor started")

    # 1. wait for the orphaned feature build
    if feature_build_running():
        log("feature build is running — waiting, will not disturb it")
        last = -1
        while feature_build_running():
            n = n_feature_dirs("features")
            if n != last:
                log(f"  features: {n}/617 events built")
                last = n
            time.sleep(120)
        log("feature build process has exited")
    ok, why = features_ok("features")
    if not ok:
        # Died without writing _index.csv. The per-event dirs survive, so rerun:
        # it rebuilds from scratch but nothing is corrupt.
        log(f"features incomplete ({why}) — rerunning build_features")
        if not run("features_events", [PY, "-u", "phase5/build_features.py"],
                   "build_features.log", post=lambda: features_ok("features")):
            log("ABORT: feature build failed")
            return 1
    else:
        log(f"features complete — {why}")
        st = load()
        st["features_events"] = {"done": True, "note": "completed by the orphaned run"}
        save(st)

    # 2. controls
    run("features_negatives",
        [PY, "-u", "phase5/build_features.py", "--negatives",
         "--out", "data/features_neg"],
        "build_features_neg.log", post=lambda: features_ok("features_neg"))

    # 3. tiles
    if not run("tiles", [PY, "-u", "phase5/build_tiles.py"],
               "build_tiles.log", post=tiles_ok):
        log("ABORT: tiling failed")
        return 1

    # 4. baselines
    run("baselines_10m",
        [PY, "-u", "phase6/baselines.py", "--tiles", "data/tiles", "--rung", "10"],
        "baselines.log")

    log("pipeline finished")
    for k, v in load().items():
        log(f"  {'OK  ' if v.get('done') else 'FAIL'} {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
