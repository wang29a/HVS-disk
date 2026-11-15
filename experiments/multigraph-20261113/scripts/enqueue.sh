#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "$0")/.." && pwd)
session=hvs-multigraph-20261113
if tmux has-session -t "$session" 2>/dev/null; then
  printf '%s\n' "Already queued: $session"
  exit 0
fi
export LC_ALL=C
export PYTHONPATH="$root/runtime/python${PYTHONPATH:+:$PYTHONPATH}"
python3 -c 'import json,sys,matplotlib; from pathlib import Path; p=Path(sys.argv[1]); assert json.loads((p/"runtime/validation/tiny-gates.json").read_text())["status"]=="PASS"' "$root"
tmux new-session -d -s "$session" "cd '$root' && LC_ALL=C python3 scripts/run_queue.py >> runtime/coordinator.log 2>&1"
printf '%s\n' "Queued: $session"
