#!/usr/bin/env bash
set -euo pipefail

HVS=/mnt/nvme3/wz/hvs-disk-paper-20260920
OUT="$HVS/experiments/gated-a0-a2"
VALIDATOR="$HVS/experiments/validate_gated_a0_a2.py"

while tmux has-session -t hvs-gated-convert 2>/dev/null; do
  sleep 15
done

exec > >(tee -a "$OUT/validation.log") 2>&1

validate_pair() {
  local dataset="$1" threshold="$2" merge_meta="$3" merge_topo="$4"
  echo "START validation dataset=$dataset threshold=$threshold time=$(date -Is)"
  python3 "$VALIDATOR" --merge-meta "$merge_meta" --merge-topo "$merge_topo" \
    --prefix "$OUT/$dataset/gated-$threshold"
  echo "DONE validation dataset=$dataset threshold=$threshold time=$(date -Is)"
}

for threshold in 0.20 0.30; do
  validate_pair openimage "$threshold" \
    "$HVS/experiments/formal-32/openimage/merge/_disk.index" \
    "$HVS/experiments/formal-32/openimage/merge/disk_index_graph.original"
done

for threshold in 0.20 0.30; do
  validate_pair cc3m "$threshold" \
    "$HVS/experiments/alpha-characterization/cc3m/_disk.index.merge" \
    "$HVS/experiments/alpha-characterization/cc3m/disk_index_graph.merge"
done

for threshold in 0.20 0.30; do
  validate_pair howto100m "$threshold" \
    "$HVS/experiments/formal-32/howto100m/merge/_disk.index" \
    "$HVS/experiments/formal-32/howto100m/merge/disk_index_graph.original"
done

echo "gated A0/A2 validation queue complete time=$(date -Is)"
