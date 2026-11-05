#!/bin/bash
cd "$(dirname "$0")/.."
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_clean --neg data/tiles_clean_neg --rung 10 \
  --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
  --select-on pixel --groups terrain,basin,rain,rain_routed,soil,landcover \
  --tag clean_region
echo "######## CLEAN+REGION TRAINING COMPLETE ########"
