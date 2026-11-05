#!/bin/bash
# Per-pixel flood prediction is now THE research question. These runs test the
# three things that were wrong for it:
#   1. the elevation prior was locked to "lower floods more" (below chance here)
#   2. the loss optimised within-tile ranking, a metric we no longer report
#   3. the best epoch was chosen on within-tile AUC, not on pixel AUC
cd "$(dirname "$0")/.."
C="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
   --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
   --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel"

echo "######## A: prior OFF, no ranking loss ########"
./.venv/bin/python -u phase6/train.py $C --prior off  --rank-weight 0 --tag pix_off
echo "######## B: prior FREE (sign learned), no ranking loss ########"
./.venv/bin/python -u phase6/train.py $C --prior free --rank-weight 0 --tag pix_free
echo "######## C: control - old settings, selected on pixel ########"
./.venv/bin/python -u phase6/train.py $C --prior downhill --rank-weight 0.3 --tag pix_ctrl
echo "######## PIXEL RUNS COMPLETE ########"
