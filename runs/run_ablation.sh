#!/bin/bash
# PROJECT_PLAN's comparison set, one channel family at a time.
# Each run is independent; the elevation rule is re-scored inside every run on
# that run's own test tiles, so the comparison is always paired.
cd "$(dirname "$0")/.."
COMMON="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
        --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15"
for G in "terrain" \
         "terrain,basin" \
         "terrain,basin,rain" \
         "terrain,basin,rain,rain_routed" \
         "terrain,basin,rain,rain_routed,soil,landcover"; do
  echo "############ GROUPS: $G ############"
  ./.venv/bin/python -u phase6/train.py $COMMON --groups "$G" || echo "FAILED: $G"
done
echo "############ ABLATION COMPLETE ############"
