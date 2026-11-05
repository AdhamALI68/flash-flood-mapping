#!/usr/bin/env bash
# First training try on the photo flood-trace labels, without the hand check:
# the loose candidate list (63 floods) and the strict list (28 floods), one
# after the other, then every model scored on the same answer keys.
#
#   setsid nohup phase7/run_both.sh > data/trace_v3_both.log 2>&1 < /dev/null & disown

set -u
cd "$(dirname "$0")/.."
PY=./.venv/bin/python
$PY phase7/plan.py set rules done "no hand check this time: loose 63 floods and strict 28 floods" > /dev/null

for RULES in candidates_v1 rules_v1; do
    echo "=== $RULES ==="
    phase7/run_final.sh "$RULES" || { echo "FAILED on $RULES"; exit 1; }
done

$PY phase7/plan.py set score running "comparing all models on the same answer keys" > /dev/null
HEAD="terrain+basin+rain+rain_routed+soil+landcover"
for KEY in rules_v1 candidates_v1; do
    $PY -u phase5/compare_on_photo_ruler.py --tags "$HEAD,trace_candidates_v1,trace_rules_v1" \
        --tiles "data/tiles_trace_$KEY" --neg data/tiles_direct_neg \
        --out "data/compare_all_on_$KEY.json" > "data/compare_all_on_$KEY.log" 2>&1 \
        || { $PY phase7/plan.py set score failed "see data/compare_all_on_$KEY.log" > /dev/null; exit 1; }
done
$PY phase7/plan.py set score done "both trainings scored, see data/compare_all_on_rules_v1.log" > /dev/null
echo "BOTH DONE"
