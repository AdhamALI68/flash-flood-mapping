#!/bin/bash
# The full sweep. Winner is chosen on VALIDATION, then looked at on test ONCE -
# picking the best of 13 by test score would inflate whatever we report.
cd "$(dirname "$0")/.."
C="--tiles data/tiles_v4 --neg data/tiles_neg_v4 \
   --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
   --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel"

# --- the untried lever: terrain resolution -------------------------------
for R in 1 2 5 30; do
  echo "######## S-rung${R}: terrain at ${R} m ########"
  ./.venv/bin/python -u phase6/train.py $C --rung $R --tag sw_rung${R}
done
echo "######## S-multi: 1 m + 10 m + 30 m stacked ########"
./.venv/bin/python -u phase6/train.py $C --rung 1,10,30 --tag sw_multi
echo "######## S-multi5: all five resolutions ########"
./.venv/bin/python -u phase6/train.py $C --rung 1,2,5,10,30 --tag sw_multi5

# --- losses --------------------------------------------------------------
echo "######## S-focal ########"
./.venv/bin/python -u phase6/train.py $C --rung 10 --loss focal --tag sw_focal
echo "######## S-conf: per-pixel confidence weighting ########"
./.venv/bin/python -u phase6/train.py $C --rung 10 --loss confidence --tag sw_conf

# --- label threshold, never swept for the model --------------------------
for T in 0.10 0.20 0.25; do
  echo "######## S-thr${T} ########"
  ./.venv/bin/python -u phase6/train.py $C --rung 10 --threshold $T --tag sw_thr${T}
done

# --- ensemble seeds of the current best ----------------------------------
for S in 1 2 3; do
  echo "######## S-seed${S} ########"
  ./.venv/bin/python -u phase6/train.py $C --rung 10 --seed $S --tag sw_seed${S}
done
echo "######## SWEEP COMPLETE ########"
