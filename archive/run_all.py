#!/usr/bin/env python3
"""
Non-stop runner. Chains every download back to back so nothing sits idle.

Each stage is independently resumable, so this can be killed and restarted at
any point and it picks up where it left off. A stage that fails is retried a few
times, then skipped so it cannot block the stages behind it - a failed rain
download should not stop the DEM from proceeding.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u run_all.py >> data/run_all.log 2>&1 </dev/null &

Watch progress in STATUS.md. Stop it with the PID printed at startup, or
`kill $(cat data/run_all.pid)`.

Order is deliberate: SAR and NLCD come first because they are the only two
things the threshold sweep needs. The DEM and rain are much larger but feed
Phase 5, so they run after rather than delaying the threshold work.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = str(ROOT / ".venv/bin/python")
LOG = ROOT / "data"
RETRIES = 3

# (name, argv, log file, needed_for)
STAGES = [
    ("SAR products (Phase 3)",
     [PY, "-u", "phase3/run_pipeline.py", "--wave", "30"],
     "phase3_run.log", "threshold sweep"),
    ("NLCD land cover",
     [PY, "-u", "phase5/download_nlcd.py"],
     "nlcd.log", "threshold sweep (farmland mask)"),
    ("DEM 1 m @ 5 km buffer",
     [PY, "-u", "phase5/download_dem.py"],
     "dem_download.log", "Phase 5 features"),
    ("Rain + soil",
     [PY, "-u", "phase5/download_rain_soil.py"],
     "rain_soil.log", "Phase 5 features"),
]


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_stage(name: str, argv: list[str], logname: str) -> bool:
    script = ROOT / argv[2]
    if not script.exists():
        print(f"[{stamp()}] SKIP {name} - {argv[2]} not written yet", flush=True)
        return False
    logf = LOG / logname
    for attempt in range(1, RETRIES + 1):
        print(f"[{stamp()}] START {name} (attempt {attempt}/{RETRIES}) "
              f"-> data/{logname}", flush=True)
        with logf.open("a") as fh:
            fh.write(f"\n=== {stamp()} attempt {attempt} ===\n")
            fh.flush()
            rc = subprocess.call(argv, cwd=ROOT, stdout=fh, stderr=subprocess.STDOUT)
        if rc == 0:
            print(f"[{stamp()}] DONE  {name}", flush=True)
            return True
        print(f"[{stamp()}] FAILED {name} rc={rc}", flush=True)
        if attempt < RETRIES:
            # Most failures here are a dropped connection; the stage resumes
            # from its own checkpoint, so retrying costs only the backoff.
            wait = 120 * attempt
            print(f"[{stamp()}] retrying in {wait}s", flush=True)
            time.sleep(wait)
    print(f"[{stamp()}] GIVING UP on {name}, continuing to the next stage", flush=True)
    return False


def main() -> int:
    (ROOT / "data").mkdir(exist_ok=True)
    (ROOT / "data/run_all.pid").write_text(str(os.getpid()))
    print(f"[{stamp()}] run_all started, PID {os.getpid()}", flush=True)
    print(f"[{stamp()}] stages: {', '.join(s[0] for s in STAGES)}", flush=True)

    results = {}
    for name, argv, logname, needed in STAGES:
        results[name] = run_stage(name, argv, logname)

    print(f"\n[{stamp()}] all stages attempted")
    for name, ok in results.items():
        print(f"  {'OK  ' if ok else 'FAIL'} {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
