#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
exec nice -n 10 ionice -c 2 -n 7 numactl --cpunodebind=3 --membind=3 \
  python3 "$ROOT/scripts/run_queue.py" >> "$ROOT/logs/queue.log" 2>&1
