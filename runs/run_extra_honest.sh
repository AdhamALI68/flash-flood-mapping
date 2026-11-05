#!/bin/bash
cd "$(dirname "$0")/.."
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_x --neg data/tiles_x_neg --rung 10 \
  --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
  --groups terrain,basin,rain,rain_routed,soil,landcover,extra \
  --tag x_extra_region
echo "######## EXTRA FEATURES, HONEST SPLIT, COMPLETE ########"
