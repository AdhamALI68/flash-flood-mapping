#!/bin/bash
cd "$(dirname "$0")/.."
COMMON="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
        --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
        --groups terrain,basin,rain,rain_routed,soil,landcover"
./.venv/bin/python -u phase6/train.py $COMMON --seed 1 --tag headline_seed1
./.venv/bin/python -u phase6/train.py $COMMON --seed 2 --tag headline_seed2
echo "######## HEADLINE ENSEMBLE SEEDS COMPLETE ########"
