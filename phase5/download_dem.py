#!/usr/bin/env python3
"""
Download the selected 1 m tiles. Resumable, verified, safe to interrupt.

Reads data/dem_download_manifest.csv (written by select_dem_tiles.py) and pulls
each tile from the USGS public S3 bucket. No credentials needed.

Resume works two ways:
  - a tile whose file already matches the expected byte count is skipped
  - a partial file is continued with an HTTP Range request rather than restarted

Both matter at ~126 GB over WiFi: a dropped connection 90 GB in must not cost
the 90 GB. Downloads write to <name>.part and are renamed only once the size
matches, so an interrupted transfer can never be mistaken for a finished tile.

Output: data/dem/1m/<project>/<filename>.tif, data/dem_download_state.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "data/dem/1m"
STATE = ROOT / "data/dem_download_state.json"
CHUNK = 1 << 20
RETRIES = 5


def load_state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(st: dict) -> None:
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    tmp.replace(STATE)


def remote_size(session: requests.Session, url: str) -> int | None:
    try:
        r = session.head(url, timeout=60, allow_redirects=True)
        if r.status_code == 200:
            return int(r.headers.get("content-length", 0)) or None
    except Exception:
        pass
    return None


def fetch(session: requests.Session, url: str, out: Path) -> tuple[bool, str]:
    """Download one tile, resuming a partial file if present."""
    part = out.with_suffix(out.suffix + ".part")
    expect = remote_size(session, url)
    if expect and out.exists() and out.stat().st_size == expect:
        return True, "already complete"

    for attempt in range(1, RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with session.get(url, headers=headers, stream=True, timeout=180) as r:
                if r.status_code in (200, 206):
                    mode = "ab" if r.status_code == 206 and have else "wb"
                    if mode == "wb":
                        have = 0
                    with part.open(mode) as fh:
                        for chunk in r.iter_content(CHUNK):
                            if chunk:
                                fh.write(chunk)
                elif r.status_code == 416 and expect and have >= expect:
                    pass          # already fully fetched
                else:
                    raise RuntimeError(f"HTTP {r.status_code}")
        except Exception as e:
            if attempt == RETRIES:
                return False, f"{type(e).__name__}: {e}"
            time.sleep(min(5 * 2 ** (attempt - 1), 120))
            continue

        got = part.stat().st_size if part.exists() else 0
        if expect and got != expect:
            if attempt == RETRIES:
                return False, f"size mismatch {got} != {expect}"
            time.sleep(5)
            continue
        # Rename only on a verified size, so a .tif on disk is always complete.
        part.replace(out)
        return True, "ok"
    return False, "retries exhausted"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/dem_download_manifest.csv")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    man = pd.read_csv(ROOT / args.manifest)
    if args.limit:
        man = man.head(args.limit)
    st = load_state()
    session = requests.Session()

    todo = [r for r in man.itertuples(index=False)
            if st.get(r.grid_key, {}).get("status") != "done"]
    total_gb = man["size_mb"].sum() / 1024
    done_gb = sum(v.get("mb", 0) for v in st.values() if v.get("status") == "done") / 1024
    print(f"{len(man)} tiles, {total_gb:.1f} GB total")
    print(f"already done: {len(man) - len(todo)} ({done_gb:.1f} GB)")
    print(f"to fetch: {len(todo)}\n", flush=True)

    t0 = time.time()
    ok = fail = 0
    for i, r in enumerate(todo, 1):
        d = DEST / str(r.project)
        d.mkdir(parents=True, exist_ok=True)
        out = d / Path(r.url).name
        good, msg = fetch(session, r.url, out)
        if good:
            ok += 1
            st[r.grid_key] = {"status": "done", "project": r.project,
                              "survey_year": int(r.survey_year),
                              "mb": round(out.stat().st_size / 1048576, 1),
                              "path": str(out.relative_to(ROOT))}
        else:
            fail += 1
            st[r.grid_key] = {"status": "failed", "project": r.project, "error": msg}
            print(f"  FAILED {r.grid_key} {r.project}: {msg}", flush=True)
        save_state(st)

        if i % 5 == 0 or i == len(todo):
            el = time.time() - t0
            gb = sum(v.get("mb", 0) for v in st.values() if v.get("status") == "done") / 1024
            rate = (gb - done_gb) * 1024 / el if el > 0 else 0
            left = (total_gb - gb) * 1024 / rate if rate > 0 else 0
            print(f"[{i}/{len(todo)}] ok {ok} fail {fail} | {gb:.1f}/{total_gb:.1f} GB "
                  f"| {rate:.2f} MB/s | ETA {left/3600:.1f} h", flush=True)

    print(f"\ndone {ok}, failed {fail}")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
