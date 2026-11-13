#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mkdir -p "$ROOT/logs"
exec nice -n 10 ionice -c 2 -n 7 numactl --physcpubind=0-47,96-111 --interleave=0,1 \
  python3 "$ROOT/scripts/run_queue.py" >> "$ROOT/logs/queue.log" 2>&1
