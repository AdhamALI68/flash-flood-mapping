#!/bin/bash
cd "$(dirname "$0")/.."
echo "######## waiting for the bigger-model job to finish ########"
while pgrep -f "phase6/train.py" >/dev/null 2>&1; do sleep 30; done

echo "######## ENSEMBLE: 2 more seeds, SAME size (base=24) as the original, on cleaned+region ########"
for SEED in 1 2; do
  ./.venv/bin/python -u phase6/train.py \
    --tiles data/tiles_clean --neg data/tiles_clean_neg --rung 10 \
    --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
    --select-on pixel --seed $SEED \
    --groups terrain,basin,rain,rain_routed,soil,landcover --tag clean_region_seed$SEED
done
echo "######## ENSEMBLE TRAINING COMPLETE ########"
