#!/bin/bash
cd "$(dirname "$0")/.."
C="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
   --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
   --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel"
echo "######## R7: continuous target, keeps the magnitude ########"
./.venv/bin/python -u phase6/train.py $C --target continuous --tag rec_cont
echo "######## R7 COMPLETE ########"
