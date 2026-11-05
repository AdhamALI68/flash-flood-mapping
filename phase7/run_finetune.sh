#!/usr/bin/env bash
# Start from the radar-label model (5,887 tiles) and train it a little more on
# clean photo-trace floods, then score it next to the starting model on the
# same photo answer key.
#
#   setsid nohup phase7/run_finetune.sh rules_v1 > data/finetune_rules_v1.log 2>&1 < /dev/null & disown
set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
NAME=${1:?labels name, for example rules_v1 or checked_v1}
TILES=data/tiles_trace_$NAME
HEAD="terrain+basin+rain+rain_routed+soil+landcover"
TAG=ft_${NAME}
[ -f "$TILES/summary.json" ] || { echo "no tiles at $TILES"; exit 1; }
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 6000 ]; do sleep 60; done
$PY -u phase6/train.py --tiles "$TILES" --neg data/tiles_direct_neg --rung 10 \
    --test-regions 1606,1809 --val-regions 1507 --epochs 30 --patience 8 --lr 5e-5 \
    --groups terrain,basin,rain,rain_routed,soil,landcover --init-from "$HEAD" --tag "$TAG" \
    > "data/train_$TAG.log" 2>&1 || { echo "FAILED training, see data/train_$TAG.log"; exit 1; }
$PY -u phase5/compare_on_photo_ruler.py --tags "$HEAD,trace_rules_v1,$TAG" \
    --tiles "$TILES" --neg data/tiles_direct_neg --out "data/compare_$TAG.json" \
    > "data/compare_$TAG.log" 2>&1 || { echo "FAILED scoring"; exit 1; }
echo "FINETUNE DONE"
