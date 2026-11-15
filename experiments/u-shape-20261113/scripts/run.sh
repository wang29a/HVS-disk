#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT=/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m
mkdir -p "$ROOT/logs" "$OUT"
export LC_ALL=C PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
export OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 OMP_NUM_THREADS=2 OMP_THREAD_LIMIT=2
export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:${LD_LIBRARY_PATH:-}"
set +e
timeout --signal=TERM --kill-after=30s 24h \
  nice -n 15 ionice -c 2 -n 7 \
  numactl --all --physcpubind=76,77 --membind=3 \
  python3 -u "$ROOT/scripts/prepare_data.py" \
    --source /mnt/nvme2/wz/disk/dataset/Laion \
    --out "$OUT" \
    --split /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/laion-six-cell-20261111/workloads/alpha-0.5/analysis/tuning-split-v2.json \
    --threads 2 --count 2000000 --seed 20261113 \
    --queue-lock /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/budget-reorder-20261103/queue.lock \
    >>"$ROOT/logs/prepare.log" 2>&1
code=$?
set -e
python3 - "$OUT" "$code" <<'PY'
from pathlib import Path
import json,sys,time
root=Path(sys.argv[1])
value=dict(exit_code=int(sys.argv[2]),ended_unix=time.time(),
           note='124 means the 24-hour outer bound expired; partial checkpoints remain resumable')
(root/'runner-status.json').write_text(json.dumps(value,indent=2)+'\n')
PY
exit "$code"
