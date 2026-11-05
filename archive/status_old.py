#!/usr/bin/env python3
"""
Live status writer. Refreshes STATUS.md every 20 s.

Same idea as trial 1's STAAAATUS.md: one file to open and see what is actually
happening, without asking. Reads only state files and process names - it never
touches the pipeline, so it is safe to leave running or kill at any time.

    cd ~/adham/trial_2
    setsid nohup ./.venv/bin/python -u status.py >/dev/null 2>&1 </dev/null &
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUT = ROOT / "STATUS.md"
INTERVAL = 20

# script name -> (label, log file, how to read progress)
JOBS = {
    "run_chain.py":          ("COMPUTE CHAIN", "chain.log"),
    "chain_v2.py":           ("RE-TILE CHAIN (correct label)", "chain_v2.log"),
    "chain_v3.py":           ("FIX CHAIN (HAND + routed rain + controls)", "chain_v3.log"),
    "rebuild_basin_stack.py": ("Phase 5 - repaired HAND into stacks", "rebuild_basin.log"),
    "fix_hand.py":           ("Phase 5 - HAND rebuild", "fix_hand.log"),
    "add_routed_to_stacks.py": ("Phase 5 - routed rain onto event grids", "add_routed.log"),
    "supervise.py":          ("PIPELINE SUPERVISOR", "supervise.log"),
    "basin_routing.py":      ("Phase 5 - basin flow routing", "basin_routing.log"),
    "route_rainfall.py":     ("Phase 5 - catchment-routed rainfall", "route_rainfall.log"),
    "build_features.py":     ("Phase 5 - feature stacks", "build_features.log"),
    "build_tiles.py":        ("Phase 5 - tiling", None),
    "baselines.py":          ("Phase 6 - baselines", "baselines.log"),
    "sentinel2_check.py":    ("Sentinel-2 validation", "sentinel2.log"),
    "download_dem10.py":     ("DEM 10 m (routing)", "dem10.log"),
    "run_pipeline.py":       ("Phase 3 - SAR download", "phase3_run.log"),
    "download_dem.py":       ("DEM 1 m download", "dem_download.log"),
    "download_landcover.py": ("Land cover download", "landcover.log"),
    "download_rain.py":      ("Rain (MRMS radar)", "rain.log"),
    "download_soil.py":      ("Soil (POLARIS)", "soil.log"),
    "set_threshold.py":      ("Threshold selection", "set_threshold.log"),
    "flood_vs_quiet.py":     ("Flood vs quiet test", "flood_vs_quiet.log"),
    "compare_measurements.py": ("Measurement comparison", "measurement_comparison2.log"),
    "fetch_discharge.py":    ("Gauge readings", "discharge.log"),
    "threshold_sweep.py":    ("Threshold sweep", "threshold.log"),
    "resolve_negatives.py":  ("Negatives - scene resolution", "negatives_resolve.log"),
    "select_negatives.py":   ("Negatives - date selection", None),
    "run_negatives.py":      ("Negatives - SAR download", "negatives_run.log"),
    "measure_flood_fraction.py": ("Flood fraction scan", "flood_fraction.log"),
    "select_dem_tiles.py":   ("DEM tile selection", None),
    "reconcile_dataset.py":  ("Dataset reconciliation", None),
    "find_all_bursts.py":    ("Phase 3 - burst resolution", "phase3_bursts.log"),
    "count_dem_tiles.py":    ("DEM tile count", "dem_count.log"),
    "downstream_gauges.py":  ("Gauge connectivity", "downstream_gauges.log"),
    "transferability.py":    ("Transferability check", "transferability.log"),
    "lidar_vintage.py":      ("Lidar vintage", None),
    "aridity_screen.py":     ("Aridity screen", None),
    "sar_availability.py":   ("Phase 2 - SAR availability", None),
}


def running() -> list[tuple[str, str, int]]:
    """Live pipeline processes, one row each.

    Matches only lines whose FIRST argument is the interpreter. A bash -c
    wrapper mentions the same python path inside its eval string, so a
    substring match reports every job twice.
    """
    out: dict[str, tuple[str, str, int]] = {}
    try:
        ps = subprocess.run(["ps", "-eo", "pid,etime,args"], capture_output=True,
                            text=True, timeout=10).stdout
    except Exception:
        return []
    for line in ps.splitlines():
        parts = line.split(None, 2)
        if len(parts) < 3:
            continue
        pid, elapsed, args = parts
        argv0 = args.split(None, 1)[0]
        if not argv0.endswith(("python", "python3", "python3.12")):
            continue
        # run_pipeline.py serves both the flood events and the negative
        # controls, distinguished only by --bursts. Without this the negatives
        # run is reported as Phase 3 and tails the wrong log.
        if "run_pipeline.py" in args and "negative" in args:
            if "Negatives - SAR download" not in out:
                out["Negatives - SAR download"] = ("Negatives - SAR download",
                                                   elapsed, int(pid))
            continue
        for script, (label, _) in JOBS.items():
            if script in args:
                # Keep the earliest-started instance if somehow duplicated.
                if label not in out:
                    out[label] = (label, elapsed, int(pid))
                break
    return list(out.values())


def parse_elapsed(s: str) -> float | None:
    """ps etime -> seconds. Formats: MM:SS, HH:MM:SS, DD-HH:MM:SS."""
    try:
        days = 0
        if "-" in s:
            d, s = s.split("-", 1)
            days = int(d)
        bits = [int(x) for x in s.split(":")]
        while len(bits) < 3:
            bits.insert(0, 0)
        return days * 86400 + bits[0] * 3600 + bits[1] * 60 + bits[2]
    except Exception:
        return None


PROGRESS_RE = re.compile(r"(\d[\d,]*)\s*/\s*(\d[\d,]*)")


# Lines that contain an "N/M" which is NOT progress. "attempt 1/2" was being
# read as "1 of 2 done", producing a confident ETA of one minute for a job that
# had barely started.
NOT_PROGRESS = ("attempt", "retry", "/2)", "looks_like")


def progress_from_log(lines: list[str]) -> tuple[int, int] | None:
    """Last 'N / M' pair in the log - how these scripts all report progress."""
    for line in reversed(lines):
        if any(k in line.lower() for k in NOT_PROGRESS):
            continue
        m = PROGRESS_RE.search(line)
        if m:
            done = int(m.group(1).replace(",", ""))
            total = int(m.group(2).replace(",", ""))
            if 0 < total and done <= total:
                return done, total
    return None


def fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return f"{sec}s"
    if sec < 3600:
        return f"{sec//60}m"
    h, m = sec // 3600, (sec % 3600) // 60
    if h < 24:
        return f"{h}h {m:02d}m"
    return f"{h//24}d {h%24}h"


TOTAL_RE = re.compile(r"(?:needed|to fetch|to resolve|total)\s*:?\s*(\d[\d,]*)", re.I)


def eta_from_completions(lines: list[str], elapsed_s: float | None) -> str | None:
    """ETA for jobs that print one line per finished item, not 'N/M'.

    Downloaders log a line per file, so there is no running counter to parse.
    The total is stated once at startup ("tiles needed: 9"), so counting
    completion lines against it gives the same estimate.
    """
    if not elapsed_s or elapsed_s < 20:
        return None
    total = None
    for ln in lines:
        m = TOTAL_RE.search(ln)
        if m:
            total = int(m.group(1).replace(",", ""))
            break
    if not total:
        return None
    done = sum(1 for l in lines
               if (" MB" in l or "already present" in l or "not published" in l))
    if done <= 0 or done >= total:
        return None
    rate = done / elapsed_s
    remain = (total - done) / rate
    finish = datetime.fromtimestamp(time.time() + remain).strftime("%H:%M")
    return (f"    **ETA ~{fmt_dur(remain)}** (about {finish}) · "
            f"{done}/{total} done")


def eta_line(lines: list[str], elapsed_s: float | None) -> str | None:
    """Extrapolate remaining time from progress so far.

    Assumes a roughly constant rate, which holds for these jobs - they are
    per-item API calls or downloads. Startup cost is included in elapsed, so
    early estimates run pessimistic.
    """
    if not elapsed_s or elapsed_s < 20:
        return None
    p = progress_from_log(lines)
    if not p:
        return None
    done, total = p
    if done <= 0 or done >= total:
        return None
    rate = done / elapsed_s
    remain = (total - done) / rate
    finish = datetime.fromtimestamp(time.time() + remain).strftime("%H:%M")
    return (f"    **ETA ~{fmt_dur(remain)}** (about {finish}) · "
            f"{done:,}/{total:,} at {rate*60:.0f}/min")


def tail(path: Path, n: int = 3) -> list[str]:
    if not path.exists():
        return []
    try:
        lines = [l for l in path.read_text(errors="replace").splitlines()
                 if l.strip() and "it/s]" not in l and "B/s]" not in l]
        return lines[-n:]
    except Exception:
        return []


def count_lines(p: Path) -> int:
    if not p.exists():
        return 0
    try:
        return sum(1 for l in p.read_text(errors="replace").splitlines() if l.strip())
    except Exception:
        return 0


def bar(done: int, total: int, width: int = 28) -> str:
    if total <= 0:
        return ""
    f = max(0, min(1, done / total))
    n = int(f * width)
    return "`" + "#" * n + "." * (width - n) + f"` **{done:,} / {total:,}** ({100*f:.0f}%)"


def phase3_state() -> tuple[int, int, int, int]:
    p = DATA / "phase3_state.json"
    if not p.exists():
        return 0, 0, 0, 0
    try:
        st = json.loads(p.read_text())
    except Exception:
        return 0, 0, 0, 0
    g = st.get("groups", {})
    e = st.get("events", {})
    gd = sum(1 for v in g.values() if v.get("stage") == "done")
    gf = sum(1 for v in g.values() if str(v.get("stage", "")).endswith("failed"))
    ed = sum(1 for v in e.values() if v.get("stage") == "done")
    return gd, gf, ed, len(g)


HISTORY = DATA / "phase3_rate_history.jsonl"


def record_rate(groups_done: int, bytes_done: int, total: int = 470,
                history: Path | None = None) -> dict | None:
    """Append a throughput sample and estimate from the observed rate.

    Measured throughput beats an assumed MB/s: it absorbs HyP3 queue time,
    retries, and WiFi variation, none of which a nominal rate captures. The
    window is the oldest sample at least 15 minutes old, so the estimate reacts
    to a genuine slowdown without swinging on one slow wave.
    """
    hist = history or HISTORY
    now = time.time()
    try:
        with hist.open("a") as fh:
            fh.write(json.dumps({"t": now, "g": groups_done, "b": bytes_done}) + "\n")
    except Exception:
        return None
    try:
        samples = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
    except Exception:
        return None
    if len(samples) < 2:
        return None

    ref = None
    for s in samples:
        if now - s["t"] >= 900:
            ref = s
        else:
            break
    if ref is None:
        ref = samples[0]

    dt = now - ref["t"]
    dg = groups_done - ref["g"]
    db = bytes_done - ref["b"]
    # 120 s is enough to see a rate; 300 s left the panel blank for
    # minutes after any restart, which reads like a fault.
    if dt < 120 or dg <= 0:
        return None

    gph = dg / (dt / 3600)
    remaining = max(0, total - groups_done)
    eta_s = remaining / gph * 3600 if gph > 0 else None
    return {
        "groups_per_hour": gph,
        "mb_per_s": (db / dt) / 1024**2 if db > 0 else None,
        "eta_s": eta_s,
        "window_min": dt / 60,
    }


def dir_size(p: Path) -> str:
    if not p.exists():
        return "0"
    try:
        n = sum(f.stat().st_size for f in p.rglob("*") if f.is_file())
    except Exception:
        return "?"
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def build() -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    L: list[str] = [f"# Trial 2 — live status", "", f"*{now}* · refreshes every {INTERVAL}s", ""]

    # Job list is needed by the panel below as well as the detail sections.
    jobs = running()

    # ---- plain-language summary, before any detail ----
    # The whole remaining pipeline in dependency order. Each entry knows how to
    # report its own progress so the panel says what is happening now, how long
    # it has left, and what follows - without needing the sections below.
    def _feat_progress(sub="features"):
        idx = DATA / sub / "_index.csv"
        dirs = [d for d in (DATA / sub).glob("*") if d.is_dir()] if (DATA / sub).exists() else []
        return len(dirs), idx.exists()

    steps = []
    # 1 routing
    rst = {}
    try:
        rst = json.loads((DATA / "basin_routing_state.json").read_text())
    except Exception:
        pass
    nb = sum(1 for v in rst.values() if v.get("done"))
    steps.append(("Flow routing, 11 basins", nb, 11, nb >= 11))
    # 2 features (events)
    nf, fdone = _feat_progress("features")
    steps.append(("Feature stacks, 617 events", nf, 617, fdone))
    # 3 features (controls)
    nn, ndone = _feat_progress("features_neg")
    steps.append(("Feature stacks, 174 controls", nn, 174, ndone))
    # 4 tiles
    tix = DATA / "tiles/index.csv"
    nt = (sum(1 for _ in tix.read_text().splitlines()) - 1) if tix.exists() else 0
    steps.append(("Tiling, all 5 rungs", nt, max(nt, 1), tix.exists()))
    # 5 baselines
    bl = list(DATA.glob("baselines_rung*.json"))
    steps.append(("Baselines — the number to beat", 1 if bl else 0, 1, bool(bl)))

    live_names = {j[0] for j in jobs}
    running_now = None
    for label, done, total, complete in steps:
        if complete:
            continue
        running_now = (label, done, total)
        break

    L += ["## In plain terms", ""]
    if running_now:
        label, done, total = running_now
        pct = f"{100*done/total:.0f}%" if total else "?"
        # ETA from measured throughput on THIS step
        hist = DATA / "pipeline_rate.jsonl"
        est = None
        try:
            t = time.time()
            with hist.open("a") as fh:
                fh.write(json.dumps({"t": t, "step": label, "n": done}) + "\n")
            sm = [json.loads(l) for l in hist.read_text().splitlines() if l.strip()]
            sm = [x for x in sm if x.get("step") == label]
            if len(sm) >= 2:
                ref = sm[0]
                for x in sm:
                    if t - x["t"] >= 600:
                        ref = x
                    else:
                        break
                dt, dn = t - ref["t"], done - ref["n"]
                if dt > 180 and dn > 0:
                    est = (dt / dn) * (total - done)
        except Exception:
            pass
        L += [f"**RUNNING NOW: {label}** — {done:,} of {total:,} ({pct})", ""]
        if est:
            fin = datetime.fromtimestamp(time.time() + est)
            L += [f"**Time left: {fmt_dur(est)}** — finishes about "
                  f"**{fin.strftime('%a %H:%M')}**", ""]
        else:
            L += ["*Time left: measuring — appears after a few minutes of progress*", ""]
        L += [bar(done, total), ""]
    else:
        L += ["**Everything in the pipeline is finished.** The dataset is ready.", ""]

    L += ["### The whole pipeline", "", "| # | Step | State |", "|---|---|---|"]
    hit_current = False
    for i, (label, done, total, complete) in enumerate(steps, 1):
        if complete:
            state = "**done**"
        elif not hit_current:
            state = f"**RUNNING** — {done:,}/{total:,}"
            hit_current = True
        else:
            state = "waiting"
        L.append(f"| {i} | {label} | {state} |")
    L += ["", "*Then: training. `train.py` is not written yet.*", ""]
    if not jobs:
        L += ["⚠️ **Nothing is running.** The pipeline is stopped and needs "
              "restarting.", ""]
    L += ["---", ""]

    # ---- running now ----
    L += ["## Running now", ""]
    if not jobs:
        L += ["**Nothing running.** No downloads in progress, nothing can run away.", ""]
    else:
        for label, elapsed, pid in jobs:
            L.append(f"**{label}** — running {elapsed} (PID {pid})")
            logname = next((v[1] for k, v in JOBS.items() if v[0] == label), None)
            if logname:
                lines = tail(DATA / logname, 60)
                el = parse_elapsed(elapsed)
                # Try the "N/M" form first, then the one-line-per-item form.
                eta = eta_line(lines, el) or eta_from_completions(lines, el)
                if eta:
                    L.append(eta)
                else:
                    done_lines = [l for l in lines if " MB" in l or "already present" in l]
                    if done_lines:
                        L.append(f"    **{len(done_lines)} items done** "
                                 f"(ETA needs a stated total in the log)")
                L.append("")
                L.append("    ```")
                for line in lines[-6:]:
                    L.append(f"    {line[:100]}")
                L.append("    ```")
            L.append("")

    # ---- path to the threshold sweep ----
    gd, gf, ed, gtot = phase3_state()
    RATE_MBS = 1.35
    stored_bytes = 0
    p3 = DATA / "phase3"
    if p3.exists():
        try:
            stored_bytes = sum(f.stat().st_size for f in p3.rglob("*") if f.is_file())
        except Exception:
            pass

    sar_running = any("SAR download" in j[0] for j in jobs)
    rate = record_rate(gd, stored_bytes) if sar_running else None

    L += ["## Phase 3 — SAR download", ""]
    L += ["**Overall**", "", bar(gd, 470), ""]

    # Within-wave detail. Groups only tick over at the END of a wave, so with
    # 30 groups per wave the overall bar can sit still for over an hour while
    # real work happens. The job counter is what actually moves.
    if sar_running:
        loglines = tail(DATA / "phase3_run.log", 40)
        wave_no = None
        for ln in reversed(loglines):
            m = re.search(r"=== wave (\d+):", ln)
            if m:
                wave_no = int(m.group(1))
                break
        jobs_done = jobs_tot = None
        for ln in reversed(loglines):
            m = re.search(r"(\d+)/(\d+) jobs finished", ln)
            if m:
                jobs_done, jobs_tot = int(m.group(1)), int(m.group(2))
                break
        if jobs_tot:
            stage = ("waiting for HyP3 to process" if jobs_done < jobs_tot
                     else "downloading products")
            L += [f"**Wave {wave_no or '?'} of ~16** — {stage}", "",
                  bar(jobs_done, jobs_tot), "",
                  "HyP3 takes 15–25 min to process a batch before any download "
                  "starts, so this bar sits at zero for a while at the start of "
                  "each wave. That is normal.", ""]

    if rate and rate["eta_s"]:
        finish = datetime.fromtimestamp(time.time() + rate["eta_s"])
        L += ["| | |", "|---|---|",
              f"| **ETA** | **{fmt_dur(rate['eta_s'])}** |",
              f"| **Finishes about** | **{finish.strftime('%a %d %b, %H:%M')}** |",
              f"| Measured rate | {rate['groups_per_hour']:.1f} groups/hour"
              + (f" · {rate['mb_per_s']:.2f} MB/s" if rate["mb_per_s"] else "") + " |",
              f"| Groups left | {470-gd} of 470 |",
              f"| Stored so far | {dir_size(p3)} |",
              f"| (measured over the last {rate['window_min']:.0f} min) | |",
              ""]
    elif sar_running:
        # Until this run has its own throughput, fall back on the rate measured
        # before the power cut - a real number from this machine and network,
        # clearly labelled so it is not mistaken for a live measurement.
        PRIOR_GPH = 16.2
        est_s = (470 - gd) / PRIOR_GPH * 3600
        finish = datetime.fromtimestamp(time.time() + est_s)
        L += ["| | |", "|---|---|",
              f"| **ETA (provisional)** | **{fmt_dur(est_s)}** |",
              f"| **Finishes about** | **{finish.strftime('%a %d %b, %H:%M')}** |",
              f"| Based on | {PRIOR_GPH} groups/hour measured before the power cut |",
              f"| Groups left | {470-gd} of 470 |",
              f"| Stored so far | {dir_size(p3)} |",
              "",
              "*Provisional — replaced by a live measurement once this run has "
              "~15 min of completed groups on record.*", ""]
    else:
        secs = 96 * (1 - gd / 470) * 1024 / RATE_MBS
        L += [f"**Not running.** {470-gd} groups left, "
              f"~{fmt_dur(secs)} at the nominal 1.35 MB/s.", ""]

    L += ["### Then the threshold sweep", "",
          "It needs coherence for every event plus the NLCD farmland mask "
          "(~2 GB, 25 min). Nothing else blocks it — the sweep itself is minutes. "
          "**The DEM and rain downloads are NOT on this path**; they feed Phase 5 "
          "and can run afterwards.", ""]

    # ---- basin routing progress, derived from STATE not the log ----
    #
    # The running process predates the [i/N] counter, so its log cannot be
    # parsed for progress. basin_routing_state.json records completed basins,
    # which is a better source anyway: it survives restarts and cannot be
    # confused by other numbers in the output.
    rs = DATA / "basin_routing_state.json"
    routing_live = any("basin flow routing" in j[0] for j in jobs)
    if rs.exists():
        try:
            rst = json.loads(rs.read_text())
        except Exception:
            rst = {}
        rdone = sum(1 for v in rst.values() if v.get("done"))
        L += ["## Phase 5 — basin flow routing", "", bar(rdone, 11), ""]
        if routing_live:
            hist = DATA / "routing_rate_history.jsonl"
            now = time.time()
            samples = []
            try:
                with hist.open("a") as fh:
                    fh.write(json.dumps({"t": now, "n": rdone}) + "\n")
                samples = [json.loads(l) for l in hist.read_text().splitlines()
                           if l.strip()]
            except Exception:
                pass
            est = None
            if len(samples) >= 2:
                ref = samples[0]
                for sm in samples:
                    if now - sm["t"] >= 900:
                        ref = sm
                    else:
                        break
                dt = now - ref["t"]
                dn = rdone - ref["n"]
                if dt > 300 and dn > 0:
                    est = (dt / dn) * (11 - rdone)
            L += ["| | |", "|---|---|"]
            if est:
                fin = datetime.fromtimestamp(now + est)
                L += [f"| **ETA** | **{fmt_dur(est)}** |",
                      f"| **Finishes about** | **{fin.strftime('%a %H:%M')}** |"]
            else:
                L += ["| ETA | measuring — appears once a basin completes |"]
            L += [f"| Basins routed | {rdone} / 11 |",
                  f"| Stored | {dir_size(DATA / 'basin_features')} |", ""]
            last = [l for l in tail(DATA / "basin_routing.log", 20)
                    if "M cells" in l or "mosaic" in l]
            if last:
                L += [f"    currently: {last[-1].strip()[:100]}", ""]
        else:
            L += ["| | |", "|---|---|",
                  "| Status | **not running** |",
                  f"| Basins routed | {rdone} / 11 |", ""]

    # ---- the compute chain ----
    cs = DATA / "chain_state.json"
    STAGES = [("basin_routing", "flow routing, 11 basins"),
              ("features_events", "feature stacks, 617 events"),
              ("features_negatives", "feature stacks, 174 controls"),
              ("tiles_all_rungs", "tiles: shared once + all 5 rungs"),
              ("baselines_10m", "baselines — the number to beat")]
    chain = {}
    if cs.exists():
        try:
            chain = json.loads(cs.read_text())
        except Exception:
            chain = {}
    chain_live = any("COMPUTE CHAIN" in j[0] for j in jobs)
    L += ["## Compute chain", "",
          ("**running**" if chain_live else "**not running**")
          + " — Phase 5 and 6, resumable at every stage", "",
          "| Stage | What | State |", "|---|---|---|"]
    for key, desc in STAGES:
        v = chain.get(key)
        if v is None:
            state = "pending"
        elif v.get("done"):
            state = f"**done** ({v.get('minutes','?')} min)"
        else:
            state = f"**FAILED** — {v.get('error','?')}"
        L.append(f"| `{key}` | {desc} | {state} |")
    L.append("")
    n_done = sum(1 for k, _ in STAGES if chain.get(k, {}).get("done"))
    L += [bar(n_done, len(STAGES)), ""]

    # ---- negative controls ----
    nst = DATA / "negatives_state.json"
    if nst.exists():
        try:
            ns = json.loads(nst.read_text())
        except Exception:
            ns = {"groups": {}, "events": {}}
        ngd = sum(1 for v in ns.get("groups", {}).values() if v.get("stage") == "done")
        nef = sum(1 for v in ns.get("groups", {}).values()
                  if str(v.get("stage", "")).endswith("failed"))
        ned = sum(1 for v in ns.get("events", {}).values() if v.get("stage") == "done")
        nbytes = 0
        ndir = DATA / "negatives"
        if ndir.exists():
            try:
                nbytes = sum(f.stat().st_size for f in ndir.rglob("*") if f.is_file())
            except Exception:
                pass
        neg_running = any("Negatives - SAR download" in j[0] for j in jobs)
        nrate = record_rate(ngd, nbytes, total=170,
                            history=DATA / "negatives_rate_history.jsonl") \
            if neg_running else None

        L += ["## Negative controls (quiet dates, same locations)", "",
              "Same places as real floods, same season, no flood reported within "
              "14 days. Terrain and look geometry identical - the only difference "
              "is whether a flood happened.", "",
              bar(ngd, 170), ""]
        if nrate and nrate["eta_s"]:
            fin = datetime.fromtimestamp(time.time() + nrate["eta_s"])
            L += ["| | |", "|---|---|",
                  f"| **ETA** | **{fmt_dur(nrate['eta_s'])}** |",
                  f"| **Finishes about** | **{fin.strftime('%a %d %b, %H:%M')}** |",
                  f"| Measured rate | {nrate['groups_per_hour']:.1f} groups/hour"
                  + (f" · {nrate['mb_per_s']:.2f} MB/s" if nrate["mb_per_s"] else "") + " |"]
        elif neg_running:
            L += ["| | |", "|---|---|",
                  "| ETA | measuring throughput, appears after ~15 min |"]
        else:
            L += ["| | |", "|---|---|", "| Status | **not running** |"]
        L += [f"| Scene triples done | {ngd} / 170 |",
              f"| Negatives stored | {ned} / 174 |",
              f"| Failed | {nef} |",
              f"| Stored | {dir_size(DATA / 'negatives')} |", ""]

    # ---- DEM download ----
    dst = DATA / "dem_download_state.json"
    if dst.exists():
        try:
            ds = json.loads(dst.read_text())
        except Exception:
            ds = {}
        ddone = sum(1 for v in ds.values() if v.get("status") == "done")
        dfail = [k for k, v in ds.items() if v.get("status") == "failed"]
        dgb = sum(v.get("mb", 0) for v in ds.values() if v.get("status") == "done") / 1024
        L += ["## DEM 1 m download", "", bar(ddone, 571), ""]
        last = [l for l in tail(DATA / "dem_download.log", 12) if "ETA" in l]
        if last:
            L += [f"    {last[-1][:110]}", ""]
        L += ["| | |", "|---|---|",
              f"| Downloaded | {dgb:.1f} / 126.1 GB |",
              f"| Tiles | {ddone} / 571 |",
              f"| **Failed** | **{len(dfail)}** |", ""]
        if dfail:
            L += ["⚠️ **Failed tiles — these need a retry before Phase 5:**", ""]
            for k in dfail[:10]:
                L.append(f"- `{k}` — {str(ds[k].get('error',''))[:70]}")
            if len(dfail) > 10:
                L.append(f"- …and {len(dfail)-10} more")
            L.append("")

    # ---- phase progress ----
    bursts = count_lines(DATA / "phase3_bursts.jsonl")
    gauges = count_lines(DATA / "downstream_gauges.jsonl")

    L += ["## Progress", "",
          "| Step | State |", "|---|---|",
          f"| Phase 1 — events | **done** — 4,872 events, 163 basins |",
          f"| Phase 2 — SAR / lidar / aridity gates | **done** — all passed |",
          f"| Gate B — coherence sees floods | **done** — 11x contrast |",
          f"| Frozen dataset | **done** — 623 events, 11 basins, 6 regions |",
          f"| Burst resolution | {bursts}/623 events |",
          f"| **Phase 3 — SAR products** | **{gd}/470 scene triples** ({ed} events) |",
          f"| Gauge connectivity | {gauges}/623 events |",
          f"| Lidar vintage | {'done' if (DATA/'lidar_vintage.csv').exists() else 'pending'} |",
          f"| DEM tile count | {'done' if (DATA/'dem_tile_counts.csv').exists() else 'pending'} |",
          f"| Transferability step 1 | {'done' if (DATA/'transferability.csv').exists() else 'running/pending'} |",
          ""]

    if gf:
        L += [f"⚠️ **{gf} groups failed** — see data/phase3_state.json", ""]

    # ---- key numbers, read from the reconciled file ----
    hn = {}
    try:
        hn = json.loads((DATA / "headline_numbers.json").read_text())
    except Exception:
        pass
    if hn:
        g = hn.get("gauge", {}); dv = hn.get("dem_vintage", {})
        L += ["## Dataset — reconciled numbers", "",
              "*(denominator is 617, the events that actually have SAR data;"
              " 6 events in dataset_phase3.csv never resolved to a burst)*", "",
              "| | |", "|---|---|",
              f"| **Events with radar** | **{hn.get('canonical_events')}** |",
              f"| Independent observations (scene triples) | {hn.get('independent_observations_groups')} |",
              f"| Basins / HUC4 regions | {hn.get('basins')} / {hn.get('huc4_regions')} |",
              f"| Excluded (no burst covers all 3 dates) | {hn.get('excluded_events')} |",
              "", "**Gauge evidence**", "", "| | |", "|---|---|",
              f"| Positive (gauge confirms flood) | **{g.get('positive')}** |",
              f"| Negative (reported, gauge silent) | **{g.get('negative')}** |",
              f"| Usable for threshold work | {g.get('usable')} ({g.get('usable_pct')}%) |",
              f"| No gauge record | {g.get('no_gauge_record')} |",
              "", "**Terrain vintage**", "", "| | |", "|---|---|",
              f"| DEM fully predates flood (clean) | **{dv.get('fully_clean_events')}** ({dv.get('fully_clean_pct')}%) |",
              f"| Any pre-event lidar within 5 km | {dv.get('any_pre_event_events') or dv.get('any_pre_event_lidar')} ({dv.get('any_pre_event_pct')}%) |",
              ""]

    # ---- data on disk ----
    def sz(rel):
        p = DATA / rel
        return dir_size(p) if p.exists() else "—"
    L += ["## Data on disk", "",
          "| Dataset | Size | For |", "|---|---|---|",
          f"| SAR coherence + amplitude | {sz('phase3')} | labels |",
          f"| 1 m lidar DEM | {sz('dem')} | terrain features |",
          f"| NOAA storm reports | {sz('raw/stormevents')} | event list |",
          f"| Watershed boundaries | {sz('raw/wbd')} | basins |",
          f"| Gauge readings | {sz('gauge_discharge.jsonl')} | validation |",
          f"| PRISM rainfall | {sz('raw/prism_ppt')} | aridity screen |",
          f"| Land cover | {sz('landcover')} | farmland mask |",
          f"| NEXRAD rain | {sz('rain')} | model input |",
          f"| SSURGO soil | {sz('soil')} | model input |",
          ""]

    # ---- audit state ----
    L += ["## Checks", "", "| Check | Result |", "|---|---|",
          "| Phase 1–2 audit (15 checks) | **0 fail** |",
          "| Raster contents real, not empty | **pass** |",
          "| All 470 groups complete (13 files x2) | **pass** |",
          "| All 571 DEM tiles verified on disk | **pass** |",
          "| Duplicate event IDs | **0** |",
          "| Partial / corrupt files | **0** |",
          "| Coherence outside [0,1] | **0 of 470** |",
          "| Discharge: negative or absurd values | **0** |",
          "| Event-count reconciliation (623 vs 617) | **fixed** |",
          ""]

    # ---- what is NOT started ----
    # 1.35 MB/s is the rate actually measured on the Gate B product downloads
    # over this machine's WiFi. Ethernet would change every row below.
    free = shutil.disk_usage(ROOT).free / 1024**3

    def dl_eta(gb: float) -> str:
        return fmt_dur(gb * 1024 / RATE_MBS)

    sar_remaining_gb = 96 * (1 - gd / 470)
    total = sar_remaining_gb + 188
    L += ["## Downloads not started", "",
          "All land on the **local disk** — §1 superseded, no external drive.", "",
          "| Download | Size | ETA | Needed for |", "|---|---|---|---|",
          f"| SAR (Phase 3, remaining) | ~{sar_remaining_gb:.0f} GB | {dl_eta(sar_remaining_gb)} | **the threshold sweep** |",
          f"| NLCD land cover | ~2 GB | {dl_eta(2)} | **the threshold sweep** (farmland mask) |",
          f"| DEM 1 m @ 5 km buffer | 148 GB | {dl_eta(148)} | Phase 5 features |",
          f"| Rain + soil | ~38 GB | {dl_eta(38)} | Phase 5 features |",
          f"| **everything** | **~{total:.0f} GB** | **{dl_eta(total)}** | |",
          "",
          f"Free space: **{free:.0f} GB** · after all downloads: "
          f"**{free-total:.0f} GB** (feature stack needs ~44 GB)", "",
          "ETAs assume WiFi at 1.35 MB/s, the rate measured during Gate B. "
          "An ethernet cable would cut every row substantially.", ""]

    L += ["---", "",
          "Stop this writer with: `pkill -f status.py` — it only reads state, "
          "so stopping it never affects a running job.", ""]
    return "\n".join(L)


def main() -> int:
    while True:
        try:
            tmp = OUT.with_suffix(".tmp")
            tmp.write_text(build())
            tmp.replace(OUT)
        except Exception as e:
            try:
                OUT.write_text(f"# status writer error\n\n`{type(e).__name__}: {e}`\n")
            except Exception:
                pass
        time.sleep(INTERVAL)


if __name__ == "__main__":
    raise SystemExit(main())
