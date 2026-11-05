#!/bin/bash
# R5/R6: drop near-empty tiles from TRAINING only. Test set untouched, so the
# numbers stay directly comparable to the 0.6950 baseline.
cd "$(dirname "$0")/.."
while pgrep -f "phase6/train\.py" >/dev/null 2>&1; do sleep 60; done
C="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
   --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
   --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel"
echo "######## R5: training tiles must be >=5% flooded ########"
./.venv/bin/python -u phase6/train.py $C --min-train-flood 0.05 --tag rec_min5
echo "######## R6: >=5% flooded AND class-weighted ########"
./.venv/bin/python -u phase6/train.py $C --min-train-flood 0.05 --pos-weight 10 --tag rec_min5pw
echo "######## R5/R6 COMPLETE ########"
