#!/usr/bin/env python3
"""
Re-tile with the mask taken from the live channels, not the frozen valid.tif.

WHY v4 EXISTS
    v3 repaired HAND and changed nothing: still 5269 tiles, still zero from
    region 1810. The repair was cancelled by build_tiles ANDing in valid.tif,
    which build_features froze with the BROKEN basin_hand already baked in.
    Measured on 1810: valid.tif says 0.0% usable, the live channels say 18-36%.

    build_tiles now derives the mask from the channels it is actually tiling.
    Nothing is loosened - every layer valid.tif checked is still checked, just
    from the current arrays instead of a stale snapshot.

v3 is kept. It is the controlled comparison: same tiles, old HAND vs new.
"""
from __future__ import annotations
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from chain_v3 import PY, DATA, log, run, check_tiles      # noqa: E402

SHARED = "basin_v2,rain,soil,landcover,rain_routed"


def main() -> int:
    log("chain v4 started - mask from live channels")
    if not run("tiles_v4",
               [PY, "-u", "phase5/build_tiles.py", "--out", "data/tiles_v4",
                "--shared", SHARED],
               "build_tiles_v4.log", post=check_tiles("data/tiles_v4", True)):
        log("ABORT: tiling failed")
        return 1
    run("tiles_neg_v4",
        [PY, "-u", "phase5/build_tiles.py", "--features", "data/features_neg",
         "--out", "data/tiles_neg_v4", "--shared", SHARED],
        "build_tiles_neg_v4.log", post=check_tiles("data/tiles_neg_v4", False))
    run("baselines_v4",
        [PY, "-u", "phase6/baselines.py", "--tiles", "data/tiles_v4", "--rung", "10"],
        "baselines_v4.log")
    run("scale_tile_v4",
        [PY, "-u", "phase6/scale_and_tile_eval.py", "--tiles", "data/tiles_v4",
         "--rung", "10"],
        "scale_tile_v4.log")
    log("chain v4 finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
