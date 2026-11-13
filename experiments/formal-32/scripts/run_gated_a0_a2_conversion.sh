#!/usr/bin/env bash
set -euo pipefail

HVS=/mnt/nvme3/wz/hvs-disk-paper-20260920
OUT="$HVS/experiments/gated-a0-a2"
BUILDER="$HVS/experiments/build_gated_a0_a2.py"
mkdir -p "$OUT"/{openimage,cc3m,howto100m}
exec > >(tee -a "$OUT/queue.log") 2>&1

run_one() {
  local dataset="$1" threshold="$2" full_meta="$3" full_topo="$4"
  local merge_meta="$5" merge_topo="$6" query_alpha="$7"
  local prefix="$OUT/$dataset/gated-${threshold}"
  echo "START dataset=$dataset threshold=$threshold time=$(date -Is)"
  python3 "$BUILDER" \
    --full-meta "$full_meta" --full-topo "$full_topo" \
    --merge-meta "$merge_meta" --merge-topo "$merge_topo" \
    --query-alpha "$query_alpha" --threshold "$threshold" \
    --output-prefix "$prefix"
  echo "DONE dataset=$dataset threshold=$threshold time=$(date -Is)"
}

for threshold in 0.20 0.30; do
  run_one openimage "$threshold" \
    "$HVS/experiments/formal-32/openimage/full/_disk.index" \
    "$HVS/experiments/formal-32/openimage/full/disk_index_graph.original" \
    "$HVS/experiments/formal-32/openimage/merge/_disk.index" \
    "$HVS/experiments/formal-32/openimage/merge/disk_index_graph.original" \
    /mnt/nvme3/wz/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs
done

for threshold in 0.20 0.30; do
  run_one cc3m "$threshold" \
    /mnt/nvme3/wz/Index/CC3M-shard/_disk.index \
    /mnt/nvme3/wz/Index/CC3M-shard/disk_index_graph \
    "$HVS/experiments/alpha-characterization/cc3m/_disk.index.merge" \
    "$HVS/experiments/alpha-characterization/cc3m/disk_index_graph.merge" \
    /mnt/nvme3/wz/dataset/CC3M_disk/base_gt_round_00/query_alpha.fvecs
done

for threshold in 0.20 0.30; do
  run_one howto100m "$threshold" \
    "$HVS/experiments/formal-32/howto100m/full/_disk.index" \
    "$HVS/experiments/formal-32/howto100m/full/disk_index_graph.original" \
    "$HVS/experiments/formal-32/howto100m/merge/_disk.index" \
    "$HVS/experiments/formal-32/howto100m/merge/disk_index_graph.original" \
    /mnt/nvme3/wz/dataset/Howto100M_disk/base_gt_round_00/query_alpha.fvecs
done

echo "gated A0/A2 conversion queue complete time=$(date -Is)"
