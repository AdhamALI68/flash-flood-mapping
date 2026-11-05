#!/bin/bash
cd "$(dirname "$0")/.."
COMMON="--tiles data/tiles_v4 --neg data/tiles_neg_v4 --rung 10 \
        --test-regions 1606,1809 --epochs 100 --patience 15 \
        --groups terrain,basin,rain,rain_routed,soil,landcover"
./.venv/bin/python -u phase6/train.py $COMMON --val-regions 1501 --tag headline_foldA
./.venv/bin/python -u phase6/train.py $COMMON --val-regions 1810 --tag headline_foldB
echo "######## FOLD-DIVERSITY ENSEMBLE MEMBERS COMPLETE ########"
