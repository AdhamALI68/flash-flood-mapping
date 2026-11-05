#!/bin/bash
cd "$(dirname "$0")/.."
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_clean --neg data/tiles_clean_neg --rung 10 \
  --split random --epochs 100 --patience 15 --select-on pixel \
  --groups terrain,basin,rain,rain_routed,soil,landcover --tag clean_random
echo "######## CLEAN+RANDOM TRAINING COMPLETE ########"
