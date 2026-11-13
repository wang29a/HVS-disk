#!/usr/bin/env bash
set -euo pipefail
cd /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/concurrency-alpha
mkdir -p logs
exec > >(tee -a logs/queue.log) 2>&1
date -Is
python3 -u scripts/pipeline.py
