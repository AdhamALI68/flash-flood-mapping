#!/usr/bin/env python3
"""
MRMS radar rainfall for every event and every negative control.

Three products per date, because a flood depends on more than one number:

  MultiSensor_QPE_24H_Pass2   rain that caused it. Pass2 is gauge-corrected,
                              which matters in deserts where radar alone
                              overestimates over dry ground
  MultiSensor_QPE_72H_Pass2   antecedent wetness - the same rain on saturated
                              ground behaves completely differently
  RadarOnly_QPE_01H           peak hourly intensity. Desert floods are driven by
                              short bursts exceeding the infiltration rate, not
                              by daily totals

NEGATIVES ARE INCLUDED. A control needs its rainfall too, or "flood vs quiet"
silently becomes "has rain data vs does not".

Resumable; a missing file is recorded, not retried forever - MRMS has genuine
gaps, especially in the early years.

Output: data/rain/<date>/*.grib2.gz, data/rain_state.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
BASE = "https://mtarchive.geol.iastate.edu"
DEST = ROOT / "data/rain"
STATE = ROOT / "data/rain_state.json"

# RadarOnly throughout, deliberately.
#
# MultiSensor_QPE_*_Pass2 is gauge-corrected and more accurate, but the archive
# only carries it from 2021. Events here span 2015-2026, so using MultiSensor
# where available and RadarOnly elsewhere would make the rainfall feature differ
# systematically by YEAR - and lidar vintage already tracks year, so the two
# biases would compound and be impossible to separate afterwards.
#
# A uniform, slightly less accurate product beats a mixed, more accurate one.
# The cost is stated: radar-only overestimates rainfall over dry desert ground,
# so absolute values are biased high. Relative comparisons between events, which
# is what the model uses, are unaffected.
# Only 01H and 24H exist across the whole 2015-2026 span; the 72H accumulation
# is absent from the early archive. Antecedent wetness is therefore built from
# the SAME 24H product at earlier offsets rather than a different product, which
# keeps one consistent measurement instead of two.
PRODUCTS = [
    ("RadarOnly_QPE_01H", "RadarOnly_QPE_01H_00.00", 0),    # peak intensity
    ("RadarOnly_QPE_24H", "RadarOnly_QPE_24H_00.00", 0),    # rain that caused it
    ("RadarOnly_QPE_24H", "RadarOnly_QPE_24H_00.00", -48),  # antecedent, 2 days
    ("RadarOnly_QPE_24H", "RadarOnly_QPE_24H_00.00", -72),  # antecedent, 3 days
]


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def fetch(session, url: str, out: Path) -> tuple[bool, str]:
    if out.exists() and out.stat().st_size > 1000:
        return True, "present"
    try:
        r = session.get(url, timeout=180, stream=True)
        if r.status_code == 404:
            return False, "not archived"
        r.raise_for_status()
        tmp = out.with_suffix(out.suffix + ".part")
        with tmp.open("wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
        tmp.replace(out)
        return True, "ok"
    except Exception as e:
        return False, f"{type(e).__name__}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    # Flood events and their quiet controls both need rainfall.
    ev = pd.read_csv(ROOT / "data/dataset_final.csv", low_memory=False)
    dates = set(pd.to_datetime(ev["BEGIN_UTC"]).dt.floor("h"))
    neg = pd.read_csv(ROOT / "data/negative_candidates.csv")
    dates |= set(pd.to_datetime(neg["quiet_date"]).dt.floor("h"))
    dates = sorted(dates)
    if args.limit:
        dates = dates[: args.limit]
    print(f"{len(dates)} unique date-hours (floods + negatives)")
    print(f"{len(PRODUCTS)} files each -> {len(dates)*len(PRODUCTS)} files, "
          f"~{len(dates)*len(PRODUCTS)*4/1024:.1f} GB\n", flush=True)

    st = load_state()
    session = requests.Session()
    DEST.mkdir(parents=True, exist_ok=True)

    ok = missing = 0
    for i, d in enumerate(dates, 1):
        key = d.strftime("%Y%m%d%H")
        if st.get(key, {}).get("done"):
            continue
        day = DEST / d.strftime("%Y%m%d")
        day.mkdir(parents=True, exist_ok=True)
        got = {}
        for folder, prefix, offset_h in PRODUCTS:
            t = d + pd.Timedelta(hours=offset_h)
            fn = f"{prefix}_{t.strftime('%Y%m%d-%H0000')}.grib2.gz"
            url = f"{BASE}/{t.strftime('%Y/%m/%d')}/mrms/ncep/{folder}/{fn}"
            good, msg = fetch(session, url, day / fn)
            got[f"{folder}{offset_h:+d}h"] = msg
            if good:
                ok += 1
            else:
                missing += 1
        st[key] = {"done": True, "products": got}
        if i % 25 == 0:
            save_state(st)
            print(f"  [{i}/{len(dates)}] files ok {ok}, missing {missing}", flush=True)
    save_state(st)

    total = sum(f.stat().st_size for f in DEST.rglob("*.grib2.gz"))
    print(f"\nfiles downloaded {ok}, unavailable {missing}")
    print(f"total size {total/1024**3:.2f} GB")
    print(f"date-hours covered: {len(st)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
