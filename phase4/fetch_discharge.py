#!/usr/bin/env python3
"""
NEXT_STEPS Track B - pull the actual gauge readings.

We know where the gauges are and which are hydrologically downstream of each
event. This fetches what they RECORDED, which is the measurement the threshold
gets validated against.

Two series per event, because a flash flood is a spike, not a day:

  instantaneous (iv)  15-minute discharge, the real peak
  daily (dv)          daily values as a fallback where iv is unavailable

Daily MEAN is the wrong statistic here - a desert flood peaking for two hours
barely moves a 24-hour average - so daily MAX (statCd 00001) is requested first
and mean only as a last resort.

The window is the event window plus a margin either side, and a baseline period
30 days before, so "did discharge rise" can be judged against that gauge's own
normal rather than an absolute number that means nothing across basins.

Small files, no bulk download. Resumable via JSONL.

Output: data/gauge_discharge.jsonl
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
IV = "https://waterservices.usgs.gov/nwis/iv/"
DV = "https://waterservices.usgs.gov/nwis/dv/"
DISCHARGE = "00060"
PRE_H, POST_H = 12, 60      # window around the event, hours
BASELINE_D = 30             # days before, for that gauge's own normal


def rdb(text: str) -> pd.DataFrame:
    lines = [l for l in text.splitlines() if l and not l.startswith("#")]
    if len(lines) < 3:
        return pd.DataFrame()
    return pd.read_csv(io.StringIO("\n".join([lines[0]] + lines[2:])),
                       sep="\t", dtype=str, low_memory=False)


def get(session, url, params, retries=3):
    delay = 3
    for a in range(retries):
        try:
            r = session.get(url, params=params, timeout=120)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.text
        except Exception:
            if a == retries - 1:
                return None
            time.sleep(delay)
            delay *= 2
    return None


def series(session, site: str, start: str, end: str) -> tuple[list, str]:
    """Instantaneous values if available, else daily max, else daily mean."""
    txt = get(session, IV, {"format": "rdb", "sites": site,
                            "parameterCd": DISCHARGE,
                            "startDT": start, "endDT": end})
    if txt:
        df = rdb(txt)
        col = next((c for c in df.columns if c.endswith("_00060")), None)
        if col is not None and len(df):
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            if len(v):
                return v.tolist(), "iv_15min"
    for stat, label in (("00001", "dv_max"), ("00003", "dv_mean")):
        txt = get(session, DV, {"format": "rdb", "sites": site,
                                "parameterCd": DISCHARGE, "statCd": stat,
                                "startDT": start[:10], "endDT": end[:10]})
        if txt:
            df = rdb(txt)
            col = next((c for c in df.columns if c.endswith(f"_00060_{stat}")), None)
            if col is not None and len(df):
                v = pd.to_numeric(df[col], errors="coerce").dropna()
                if len(v):
                    return v.tolist(), label
    return [], "none"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/gauge_discharge.jsonl")
    args = ap.parse_args()

    ev = pd.read_csv(ROOT / "data/dataset_phase3.csv", low_memory=False)
    ev["BEGIN_UTC"] = pd.to_datetime(ev["BEGIN_UTC"])
    ev["END_UTC"] = pd.to_datetime(ev["END_UTC"]).fillna(ev["BEGIN_UTC"])

    dg = {}
    for line in (ROOT / "data/downstream_gauges.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            dg[r["EVENT_ID"]] = [s["site_no"] for s in r["sites"] if s.get("site_no")]

    out = ROOT / args.out
    done = set()
    if out.exists():
        for line in out.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["EVENT_ID"])
        print(f"resuming: {len(done)} events already done")

    todo = ev[ev["EVENT_ID"].isin([e for e in dg if dg[e]]) & ~ev["EVENT_ID"].isin(done)]
    print(f"{len(todo)} events with a downstream gauge to fetch\n", flush=True)

    session = requests.Session()
    n_peak = n_flat = n_nodata = 0
    with out.open("a") as fh:
        for i, r in enumerate(todo.itertuples(index=False), 1):
            sites = dg.get(r.EVENT_ID, [])[:3]      # nearest few are enough
            t0, t1 = r.BEGIN_UTC, r.END_UTC
            ws = (t0 - pd.Timedelta(hours=PRE_H)).strftime("%Y-%m-%dT%H:%MZ")
            we = (t1 + pd.Timedelta(hours=POST_H)).strftime("%Y-%m-%dT%H:%MZ")
            bs = (t0 - pd.Timedelta(days=BASELINE_D)).strftime("%Y-%m-%dT%H:%MZ")
            be = (t0 - pd.Timedelta(days=1)).strftime("%Y-%m-%dT%H:%MZ")

            recs = []
            for s in sites:
                ev_v, src = series(session, s, ws, we)
                bl_v, _ = series(session, s, bs, be)
                if not ev_v:
                    continue
                peak = max(ev_v)
                base = (sorted(bl_v)[len(bl_v) // 2] if bl_v else None)
                # A ratio alone is wrong here. Arid washes are EPHEMERAL: their
                # normal state is zero flow, so baseline == 0 is the common case,
                # not a missing value. Dividing by it discards precisely the
                # clearest floods - a channel going from 0 to 2030 cfs is the
                # strongest signal available, not an undefined one.
                if base is None:
                    ratio, kind = None, "no_baseline"
                elif base > 0:
                    ratio, kind = peak / base, "perennial"
                elif peak > 0:
                    ratio, kind = float("inf"), "ephemeral_dry_to_flowing"
                else:
                    ratio, kind = 0.0, "dry_throughout"
                recs.append({
                    "site_no": s, "source": src,
                    "event_peak": peak,
                    "baseline_median": base,
                    "peak_ratio": None if ratio == float("inf") else ratio,
                    "response": kind,
                    "n_obs": len(ev_v),
                })

            def strength(x):
                """Rank responses: dry-to-flowing beats any finite ratio."""
                if x["response"] == "ephemeral_dry_to_flowing":
                    return (2, x["event_peak"])
                return (1, x["peak_ratio"] or 0)

            best = max(recs, key=strength, default=None)
            rec = {"EVENT_ID": int(r.EVENT_ID), "huc8": str(r.huc8).zfill(8),
                   "begin_utc": str(t0), "gauges": recs,
                   "best_peak_ratio": best["peak_ratio"] if best else None,
                   "best_response": best["response"] if best else None,
                   "best_peak": best["event_peak"] if best else None,
                   "best_site": best["site_no"] if best else None}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()

            if not recs:
                n_nodata += 1
            elif best["response"] == "ephemeral_dry_to_flowing" or \
                 (best["peak_ratio"] or 0) >= 2:
                n_peak += 1
            else:
                n_flat += 1

            if i % 25 == 0:
                print(f"[{i}/{len(todo)}] clear rise {n_peak} | no rise {n_flat} | "
                      f"no data {n_nodata}", flush=True)
            time.sleep(0.15)

    print(f"\nclear discharge rise (>=2x baseline) : {n_peak}")
    print(f"no rise                              : {n_flat}")
    print(f"gauge returned nothing               : {n_nodata}")
    print(f"\nwrote {out}")
    print("\nThe 'no rise' pile is the important one - those are events where a")
    print("flood was reported but no gauge responded. Without them the threshold")
    print("can only be tuned to find floods, never to avoid inventing them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
