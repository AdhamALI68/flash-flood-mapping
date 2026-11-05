#!/bin/bash
# Re-tile with sink depth + channel distance, then train on BOTH splits.
# region = held-out deserts (the honest transfer claim)
# random = same ground different day (easier, reported separately, never as the
#          headline - measured earlier at +0.13 AUC from spatial memorisation)
cd "$(dirname "$0")/.."
S="basin_v2,rain,soil,landcover,rain_routed,extra"
G="terrain,basin,extra,rain,rain_routed,soil,landcover"

./.venv/bin/python -u phase5/build_tiles.py --out data/tiles_x --shared "$S" || exit 1
./.venv/bin/python -u phase5/build_tiles.py --features data/features_neg \
   --out data/tiles_x_neg --shared "$S" || exit 1

for SP in region random; do
  echo "######## SPLIT: $SP ########"
  ./.venv/bin/python -u phase6/train.py \
    --tiles data/tiles_x --neg data/tiles_x_neg --rung 10 \
    --test-regions 1606,1809 --val-regions 1507 \
    --epochs 100 --patience 15 --select-on pixel \
    --groups "$G" --split $SP --tag x_$SP
done
echo "######## EXTRA RUNS COMPLETE ########"
