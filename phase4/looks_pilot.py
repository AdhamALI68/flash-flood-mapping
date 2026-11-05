#!/usr/bin/env python3
"""
Phase 4 - PILOT: does reprocessing at more looks fix the label?

    ./.venv/bin/python -u phase4/looks_pilot.py --submit --n 40
    ./.venv/bin/python -u phase4/looks_pilot.py --collect

THE CLAIM UNDER TEST
    Our label is coherence drop from HyP3 products at 5x1 looks. Coherence
    estimated from N looks carries noise of roughly (1-g^2)/sqrt(2N), so the
    DROP - a difference of two such estimates - carries about +/-0.09 at N=5.
    Our threshold is 0.15, which is therefore a 1.6 sigma test, and a 3 sigma
    cut fires on 0.00% of pixels.

    If that is the ceiling, reprocessing the SAME scene pairs at 20x4 looks
    (N=80, 16x more) should cut that noise about 4x and make the same threshold
    a ~7 sigma test.

WHY THIS COULD NOT BE TESTED IN SOFTWARE
    An earlier attempt box-car averaged the coherence AFTER it was computed and
    found no gain. That was not a valid proxy: real multilooking averages the
    complex interferogram BEFORE coherence is formed, which removes the
    estimator's BIAS as well as its variance. A coherence estimate from few
    looks is biased high, and averaging biased numbers does not remove bias.
    Only reprocessing settles it.

WHAT IS ORDERED
    The same scene triples we already have, at 10x2 (40 m) and 20x4 (80 m), for
    a sample of events in the held-out test regions. Two jobs per event per
    looks setting, so 4 jobs per event.

COST
    ~4 credits per event. 40 events is ~160 credits.

Output: data/looks_pilot_state.json, data/looks_pilot/
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "data/looks_pilot_state.json"
OUT = ROOT / "data/looks_pilot"
LOOKS = ["10x2", "20x4"]


def load():
    return json.loads(STATE.read_text()) if STATE.exists() else {"events": {}}


def save(s):
    tmp = STATE.with_suffix(".tmp"); tmp.write_text(json.dumps(s, indent=1))
    tmp.replace(STATE)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--collect", action="store_true")
    ap.add_argument("--n", type=int, default=40)
    args = ap.parse_args()

    import hyp3_sdk
    h = hyp3_sdk.HyP3()
    st = load()

    if args.submit:
        recs = [json.loads(l) for l in
                (ROOT / "data/phase3_bursts.jsonl").read_text().splitlines()
                if l.strip()]
        ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
        ev["EVENT_ID"] = ev.EVENT_ID.astype(int).astype(str)
        # held-out regions only: the pilot must speak to the headline number
        test_ev = set(ev[ev.huc4.astype(str).isin(["1606", "1809"])].EVENT_ID)
        recs = [r for r in recs if str(r.get("EVENT_ID")) in test_ev
                and r.get("pairs")]
        seen, chosen = set(), []
        for r in recs:
            k = json.dumps(r["pairs"], sort_keys=True)
            if k in seen:
                continue
            seen.add(k); chosen.append(r)
            if len(chosen) >= args.n:
                break
        print(f"{len(chosen)} distinct scene triples in the test regions")
        print(f"credits before: {h.my_info()['remaining_credits']}")
        print(f"about to order {len(chosen)*2*len(LOOKS)} jobs "
              f"(~{len(chosen)*2*len(LOOKS)} credits)\n", flush=True)

        for r in chosen:
            eid = str(r["EVENT_ID"])
            cur = st["events"].setdefault(eid, {})
            for lk in LOOKS:
                if cur.get(lk, {}).get("jobs"):
                    continue
                jobs = {}
                try:
                    for tag, (ref, sec) in r["pairs"].items():
                        b = h.submit_insar_isce_multi_burst_job(
                            reference=[ref], secondary=[sec],
                            name=f"looks{lk}-{eid}-{tag}",
                            apply_water_mask=False, looks=lk)
                        jobs[tag] = b.jobs[0].job_id
                    cur[lk] = {"jobs": jobs, "stage": "submitted"}
                    print(f"  {eid} {lk}: submitted {len(jobs)} jobs", flush=True)
                except Exception as e:
                    cur[lk] = {"stage": "failed", "error": str(e)[:200]}
                    print(f"  {eid} {lk}: FAILED {e}", flush=True)
            save(st)
        print(f"\ncredits after: {h.my_info()['remaining_credits']}")
        print("now wait, then run with --collect")
        return 0

    if args.collect:
        done = pend = fail = 0
        for eid, d in st["events"].items():
            for lk in LOOKS:
                j = d.get(lk, {})
                if not j.get("jobs"):
                    continue
                states = []
                for jid in j["jobs"].values():
                    try:
                        states.append(h.get_job_by_id(jid).status_code)
                    except Exception:
                        states.append("ERROR")
                if all(s == "SUCCEEDED" for s in states):
                    done += 1
                elif any(s in ("FAILED", "ERROR") for s in states):
                    fail += 1
                else:
                    pend += 1
        print(f"job sets: {done} succeeded, {pend} pending, {fail} failed")
        if pend:
            print("not ready yet - run --collect again later")
        return 0

    print("use --submit or --collect")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
