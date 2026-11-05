#!/bin/bash
# Re-run the headline at the label threshold the analysis actually chose.
# phase4/set_threshold.py swept cutoffs against the paired quiet controls and
# selected 0.075 (separation 0.8846). The pipeline then used 0.15 (0.8673) as a
# hard-coded default. This closes that gap.
cd "$(dirname "$0")/.."
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
  --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
  --groups terrain,basin,rain,rain_routed,soil,landcover \
  --select-on pixel --threshold 0.075 --tag thr075
echo "######## THR075 COMPLETE ########"
