#!/bin/bash
# Chasing the fault, not the threshold. The model currently LOSES to slope alone
# on flood days (0.6351 vs 0.6595), which a model given slope as an input should
# never do. Each run tests one suspect.
cd "$(dirname "$0")/.."
C="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
   --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
   --groups terrain,basin,rain,rain_routed,soil,landcover --select-on pixel"

echo "######## R1: class-weighted loss (pos_weight 10) ########"
./.venv/bin/python -u phase6/train.py $C --pos-weight 10 --tag rec_pw10
echo "######## R2: train on flood days only - forces spatial learning ########"
./.venv/bin/python -u phase6/train.py $C --train-on flood_only --tag rec_floodonly
echo "######## R3: both, plus the ranking loss that targets placement ########"
./.venv/bin/python -u phase6/train.py $C --pos-weight 10 --train-on flood_only \
    --rank-weight 0.3 --tag rec_both
echo "######## R4: bigger model (base 40) ########"
./.venv/bin/python -u phase6/train.py $C --pos-weight 10 --base 40 --tag rec_big
echo "######## RECALL RUNS COMPLETE ########"
