#!/bin/bash
cd "$(dirname "$0")/.."
echo "######## building AGGRESSIVELY cleaned tiles (min_red=45, min_hole=60) ########"
./.venv/bin/python -u phase5/build_clean_tiles.py \
  --out-pos data/tiles_clean2 --out-neg data/tiles_clean2_neg \
  --min-red 45 --min-hole 60

echo "######## training BIGGER model (base=40) on aggressively-cleaned labels, honest region split ########"
./.venv/bin/python -u phase6/train.py \
  --tiles data/tiles_clean2 --neg data/tiles_clean2_neg --rung 10 \
  --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
  --select-on pixel --base 40 \
  --groups terrain,basin,rain,rain_routed,soil,landcover --tag clean2_region_big

echo "######## AGGRESSIVE CLEAN + BIGGER MODEL COMPLETE ########"
