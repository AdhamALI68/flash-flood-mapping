#!/bin/bash
cd "$(dirname "$0")/.."
./.venv/bin/python -u phase5/route_rainfall.py
echo "=== NEGATIVES ==="
./.venv/bin/python -u phase5/route_rainfall.py --negatives --out data/rain_routed_neg
echo "=== ALL ROUTING DONE ==="
