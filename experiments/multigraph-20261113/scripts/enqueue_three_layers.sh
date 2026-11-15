#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
session=hvs-three-layer-20261114
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' "Already running: $session"
  exit 0
fi
mkdir -p "$root/runtime/three-layer-20261114"
export LC_ALL=C
export PYTHONPATH="$root/runtime/python${PYTHONPATH:+:$PYTHONPATH}"
tmux new-session -d -s "$session" "cd '$root' && LC_ALL=C OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2 numactl --all --physcpubind=80,81 --membind=3 python3 scripts/run_three_layers.py >> runtime/three-layer-20261114/coordinator.log 2>&1"
printf '%s\n' "Started: $session"
