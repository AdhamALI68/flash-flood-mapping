#!/bin/bash
cd "$(dirname "$0")/.."
echo "######## R8: RANDOM split - same ground, different day ########"
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
  --split random --epochs 100 --patience 15 --select-on pixel \
  --groups terrain,basin,rain,rain_routed,soil,landcover --tag rec_random
echo "######## R8 COMPLETE ########"
