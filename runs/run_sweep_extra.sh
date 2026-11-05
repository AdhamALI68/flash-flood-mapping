#!/bin/bash
cd "$(dirname "$0")/.."
while pgrep -f "phase6/train\.py" >/dev/null 2>&1; do sleep 60; done
echo "######## S-eng: the team's engineered features ########"
./.venv/bin/python -u phase6/train.py --tiles data/tiles_v4 --neg data/tiles_neg_v4 \
  --rung 10 --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
  --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel \
  --engineered --tag sw_eng
echo "######## S-eng COMPLETE ########"
