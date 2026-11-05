#!/usr/bin/env bash
# The automatic end of the flood-trace plan, once the rules are settled:
# build tiles from the kept floods -> train -> score.
# Each step updates data/trace_v3/plan.json, which STATUS.md shows live.
#
#   setsid nohup phase7/run_final.sh rules_v1 > data/trace_v3_final.log 2>&1 < /dev/null & disown

set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
RULES=${1:?give the rules name, for example rules_v1}
LABELS=data/trace_v3/labels/$RULES
TAG=trace_$RULES
TILES=data/tiles_trace_$RULES
plan() { $PY phase7/plan.py set "$@" > /dev/null; }
fail() { plan "$1" failed "$2"; echo "FAILED at $1: $2"; exit 1; }

[ -f "$LABELS/summary.json" ] || fail tiles "no labels at $LABELS, run apply_rules.py first"

plan tiles running "from $RULES"
$PY -u phase7/build_trace_tiles.py --labels "$LABELS" --out "$TILES" \
    > data/trace_v3_tiles_$RULES.log 2>&1 || fail tiles "see data/trace_v3_tiles_$RULES.log"
plan tiles done "$($PY -c "import json;s=json.load(open('$TILES/summary.json'));print(f\"{s['tiles']} tiles from {s['clean_floods']} floods\")")"

# shared GPU: wait for room, never touch anyone else's job
plan train "waiting for GPU" "needs 6 GB free"
until [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -ge 6000 ]; do
    sleep 60
done
plan train running "tag $TAG"
$PY -u phase6/train.py --tiles "$TILES" --neg data/tiles_direct_neg --rung 10 \
    --test-regions 1606,1809 --val-regions 1507 --epochs 100 --patience 15 \
    --groups terrain,basin,rain,rain_routed,soil,landcover --tag "$TAG" \
    > data/trace_v3_train_$RULES.log 2>&1 || fail train "see data/trace_v3_train_$RULES.log"
plan train done ""

plan score running ""
$PY -u phase5/make_result_folder.py --tag "$TAG" --tiles "$TILES" \
    --neg data/tiles_direct_neg --out "handoff/results_$TAG" --base 24 --split region --n 8 \
    > data/trace_v3_score_$RULES.log 2>&1 || fail score "see data/trace_v3_score_$RULES.log"
$PY -u phase5/compare_on_photo_ruler.py \
    --tags "terrain+basin+rain+rain_routed+soil+landcover,$TAG" \
    --tiles "$TILES" --neg data/tiles_direct_neg --out data/compare_trace_ruler_$RULES.json \
    >> data/trace_v3_score_$RULES.log 2>&1 || fail score "see data/trace_v3_score_$RULES.log"
plan score done "$($PY -c "import json;d=json.load(open('data/${TAG}_final.json'));print(f\"recall {d['recall']:.3f}, precision {d['precision']:.3f}, AUC {d['auc']:.3f}\")")"
echo "PIPELINE COMPLETE"
