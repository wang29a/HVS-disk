#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
ulimit -c 0
exec >> queue.log 2>&1
export LC_ALL=C
exec nice -n 10 ionice -c2 -n7 numactl --cpunodebind=3 --membind=3 python3 -u scripts/run.py
