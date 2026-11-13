#!/usr/bin/env bash
set -euo pipefail

HVS=/mnt/nvme3/wz/hvs-disk-paper-20260920
OUT="$HVS/experiments/alpha-characterization"
ANALYZER="$HVS/experiments/analyze_alpha_metadata.py"
mkdir -p "$OUT"/{openimage,cc3m,howto100m}
exec > >(tee -a "$OUT/queue.log") 2>&1
date -Is

CCSRC=/mnt/nvme3/wz/Index/CC3M-shard
if [[ ! -s "$OUT/cc3m/_disk.index.merge" || ! -s "$OUT/cc3m/disk_index_graph.merge" ]]; then
  "$HVS/build-paper/tests/merge_alpha" \
    "$CCSRC/_disk.index" "$CCSRC/disk_index_graph" \
    "$OUT/cc3m/_disk.index.merge" "$OUT/cc3m/disk_index_graph.merge" 2
fi

python3 "$ANALYZER" --dataset openimage \
  --full-meta "$HVS/experiments/formal-32/openimage/full/_disk.index" \
  --full-topo "$HVS/experiments/formal-32/openimage/full/disk_index_graph.original" \
  --merge-meta "$HVS/experiments/formal-32/openimage/merge/_disk.index" \
  --merge-topo "$HVS/experiments/formal-32/openimage/merge/disk_index_graph.original" \
  --query-alpha /mnt/nvme3/wz/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
  --output "$OUT/openimage/summary.json"

python3 "$ANALYZER" --dataset cc3m \
  --full-meta "$CCSRC/_disk.index" --full-topo "$CCSRC/disk_index_graph" \
  --merge-meta "$OUT/cc3m/_disk.index.merge" --merge-topo "$OUT/cc3m/disk_index_graph.merge" \
  --query-alpha /mnt/nvme3/wz/dataset/CC3M_disk/base_gt_round_00/query_alpha.fvecs \
  --output "$OUT/cc3m/summary.json"

python3 "$ANALYZER" --dataset howto100m \
  --full-meta "$HVS/experiments/formal-32/howto100m/full/_disk.index" \
  --full-topo "$HVS/experiments/formal-32/howto100m/full/disk_index_graph.original" \
  --merge-meta "$HVS/experiments/formal-32/howto100m/merge/_disk.index" \
  --merge-topo "$HVS/experiments/formal-32/howto100m/merge/disk_index_graph.original" \
  --query-alpha /mnt/nvme3/wz/dataset/Howto100M_disk/base_gt_round_00/query_alpha.fvecs \
  --output "$OUT/howto100m/summary.json"

date -Is
echo "alpha characterization complete"
