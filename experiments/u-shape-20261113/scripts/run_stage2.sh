#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
mkdir -p "$ROOT/stage2/logs"
# The build and controller use one shared global experiment lock. The controller
# waits for the source graph, then runs only one measured process at a time.
# A lost builder, six-hour lock timeout, semantic error, or stage gate is explicit.
exec python3 -u "$ROOT/scripts/stage2_queue.py" >>"$ROOT/stage2/logs/coordinator.log" 2>&1
