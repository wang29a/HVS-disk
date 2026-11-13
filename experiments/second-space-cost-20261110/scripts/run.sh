#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
ulimit -c 0
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1
exec >> logs/queue.log 2>&1
exec nice -n 10 ionice -c2 -n7 numactl --cpunodebind=3 --membind=3 python3 -u scripts/endpoint.py
