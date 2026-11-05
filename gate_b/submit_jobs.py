#!/usr/bin/env python3
"""
Gate B, step 2 - submit the baseline and co-event InSAR jobs.

Two jobs, same burst, same looks, same processing - the only difference is which
interval they span:

    baseline : pre2 -> pre1   (both before the flood)
    coevent  : pre1 -> post   (spans the flood)

Any coherence difference between them is attributable to the flood, because
everything else is held fixed. Matching the two intervals matters: if the
baseline spanned 24 days and the co-event 6, the comparison would be measuring
elapsed time rather than flooding.

apply_water_mask stays False on purpose - masking water would remove the signal
we are trying to detect.

Output: gate_b/<event>_jobs.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import hyp3_sdk

LOOKS = "5x1"  # 20 m, the finest coherence HyP3 produces


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bursts", required=True, help="path to <event>_bursts.json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = json.loads(Path(args.bursts).read_text())
    ev = spec["event_id"]
    pairs = spec["pairs"]

    print(f"event {ev}  {spec['basin']}  {spec['begin_utc']}")
    print(f"burst {spec['full_burst_id']}  looks={LOOKS} (20 m)")
    for tag, (a, b) in pairs.items():
        print(f"  {tag:<9} {a}\n            -> {b}")

    if args.dry_run:
        print("\ndry run, nothing submitted")
        return 0

    h = hyp3_sdk.HyP3()
    before = h.my_info()["remaining_credits"]
    print(f"\ncredits before: {before}")

    submitted = {}
    for tag, (ref, sec) in pairs.items():
        batch = h.submit_insar_isce_multi_burst_job(
            reference=[ref],
            secondary=[sec],
            name=f"gateb-{ev}-{tag}",
            apply_water_mask=False,
            looks=LOOKS,
        )
        job = batch.jobs[0]
        submitted[tag] = {"job_id": job.job_id, "name": job.name,
                          "reference": ref, "secondary": sec}
        print(f"  submitted {tag}: {job.job_id}")

    after = h.my_info()["remaining_credits"]
    print(f"credits after : {after}  (spent {before - after})")

    dest = Path(args.bursts).with_name(f"{ev}_jobs.json")
    dest.write_text(json.dumps({"event_id": ev, "looks": LOOKS,
                                "burst": spec["full_burst_id"],
                                "jobs": submitted}, indent=2))
    print(f"\nwrote {dest}")
    print("HyP3 products expire 14 days after processing - download promptly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
