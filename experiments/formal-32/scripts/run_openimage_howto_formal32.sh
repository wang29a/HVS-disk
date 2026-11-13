#!/usr/bin/env bash
set -euo pipefail

HVS=/mnt/nvme3/wz/hvs-disk-paper-20260920
ROOT="$HVS/experiments/formal-32"
STAR=/mnt/nvme3/wz/starling-wang29a
export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:$STAR/release-gcc11-abi0/gnu_11.4_cxx17_64_release:${LD_LIBRARY_PATH:-}"

while tmux ls 2>/dev/null | grep -q '^hvs-formal32:'; do
  sleep 30
done
if ! tail -5 "$ROOT/cc3m/logs/queue.log" | grep -q "queue complete"; then
  echo "CC3M queue did not finish cleanly; refusing to mix incomplete runs" >&2
  exit 1
fi

mkdir -p "$ROOT/logs"
exec > >(tee -a "$ROOT/logs/openimage-howto-queue.log") 2>&1
date -Is
echo "Starting OpenImages and HowTo100M after successful CC3M completion"

cd "$HVS"
export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build-paper -DBUILD_WITH_PQ=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-paper -j32 --target create_single_file_index merge_alpha remove_alpha pad_partition
unset ADDITIONAL_DEFINITIONS

starling_layout() {
  local dir="$1" ep="$2"
  if [[ ! -s "$dir/partition.bin" ]]; then
    cd "$STAR"
    ./release-gcc11-abi0/graph_partition/partitioner \
      --data_type float --gp_file "$dir/partition.bin" \
      --custom_graph 1 --graph_file "$dir/disk_index_graph.original" \
      --meta_file "$dir/_disk.index" --ep_size "$ep" -T 32 -L 8
  fi
  if [[ ! -s "$dir/disk_index_graph.relayout" ]]; then
    cd "$STAR"
    ./release-gcc11-abi0/tests/utils/index_relayout_custom --custom \
      "$dir/disk_index_graph.original" "$dir/_disk.index" \
      "$dir/partition.bin" "$dir/disk_index_graph.relayout"
  fi
  cd "$HVS"
  if [[ ! -s "$dir/partition.bin.aligned" ]]; then
    "$HVS/build-paper/tests/pad_partition" "$dir/partition.bin"
  fi
  ln -sfn "$dir/disk_index_graph.relayout" "$dir/disk_index_graph"
  ln -sfn "$dir/partition.bin.aligned" "$dir/_partition.bin.aligned"
}

prepare_dataset() {
  local name="$1" src="$2" data="$3" ep="$4" single="$5"
  local out="$ROOT/$name"
  mkdir -p "$out"/{baseline,full,merge,no-alpha,logs}

  ln -sfn "$single" "$out/baseline/single_index"
  for f in emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
    ln -sfn "$src/$f" "$out/baseline/$f"
  done

  ln -sfn "$src/_disk.index" "$out/full/_disk.index"
  ln -sfn "$src/disk_index_graph" "$out/full/disk_index_graph.original"

  if [[ ! -s "$out/merge/_disk.index" || ! -s "$out/merge/disk_index_graph.original" ]]; then
    "$HVS/build-paper/tests/merge_alpha" \
      "$src/_disk.index" "$src/disk_index_graph" \
      "$out/merge/_disk.index" "$out/merge/disk_index_graph.original" 2
  fi
  if [[ ! -s "$out/no-alpha/_disk.index" || ! -s "$out/no-alpha/disk_index_graph.original" ]]; then
    "$HVS/build-paper/tests/remove_alpha" \
      "$src/_disk.index" "$src/disk_index_graph" \
      "$out/no-alpha/_disk.index" "$out/no-alpha/disk_index_graph.original"
  fi

  starling_layout "$out/full" "$ep"
  starling_layout "$out/merge" "$ep"
  starling_layout "$out/no-alpha" "$ep"

  for kind in full merge no-alpha; do
    for f in disk_index_data emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
      ln -sfn "$src/$f" "$out/$kind/$f"
    done
  done
}

run_dataset() {
  local name="$1"
  local data="$2"
  local out="$ROOT/$name"
  local log="$out/logs"
  local q1="$data/query_img_emb.fvecs" q2="$data/query_text_emb.fvecs"
  local alpha="$data/base_gt_round_00/query_alpha.fvecs"
  local gt="$data/base_gt_round_00/top10_results.ivecs"
  local ls=(20 40 60 80 100 120 160 200 300 400)
  export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores

  run_one() {
    local rep="$1" kind="$2" mode strategy binary
    if [[ "$kind" == baseline ]]; then
      binary="$HVS/build-single-baseline/tests/search_disk_index"
      mode=0; strategy=0
    else
      binary="$HVS/build-paper/tests/search_disk_index"
      mode=6; strategy=17
    fi
    echo "RUN dataset=$name rep=$rep kind=$kind start=$(date -Is)"
    /usr/bin/time -v numactl --cpunodebind=0,1 --interleave=0,1 \
      "$binary" float "$out/$kind/" 32 4 "$q1" "$q2" "$alpha" "$gt" \
      10 l2 "$mode" 0 "$strategy" "${ls[@]}" > "$log/rep${rep}-${kind}.log" 2>&1
    echo "DONE dataset=$name rep=$rep kind=$kind end=$(date -Is)"
  }
  for kind in baseline full merge no-alpha; do run_one 1 "$kind"; done
  for kind in full merge no-alpha baseline; do run_one 2 "$kind"; done
  for kind in merge no-alpha baseline full; do run_one 3 "$kind"; done
}

OPEN_SRC=/mnt/nvme3/wz/Index/Openimage
OPEN_DATA=/mnt/nvme3/wz/dataset/Openimage_disk
OPEN_SINGLE="$ROOT/openimage/baseline/single_index.generated"
mkdir -p "$ROOT/openimage/baseline" "$ROOT/openimage/logs"
if [[ ! -s "$OPEN_SINGLE" ]]; then
  "$HVS/build-paper/tests/create_single_file_index" \
    "$OPEN_SRC/mem.index" "$OPEN_DATA/base_img_emb.fvecs" \
    "$OPEN_DATA/base_text_emb.fvecs" "$OPEN_SRC/medoids.bin" "$OPEN_SINGLE" 0
fi
python3 "$HVS/experiments/validate_single_full_alpha.py" \
  --mem "$OPEN_SRC/mem.index" --single "$OPEN_SINGLE" \
  --base1 "$OPEN_DATA/base_img_emb.fvecs" --base2 "$OPEN_DATA/base_text_emb.fvecs" \
  > "$ROOT/openimage/logs/baseline-validation.log" 2>&1
prepare_dataset openimage "$OPEN_SRC" "$OPEN_DATA" 12 "$OPEN_SINGLE"
run_dataset openimage "$OPEN_DATA"

HOWTO_SRC=/mnt/nvme3/wz/Index/Howto100MM-shard
HOWTO_DATA=/mnt/nvme3/wz/dataset/Howto100M_disk
prepare_dataset howto100m "$HOWTO_SRC" "$HOWTO_DATA" 11 "$HOWTO_SRC/single_index"
run_dataset howto100m "$HOWTO_DATA"

date -Is
echo "OpenImages and HowTo100M formal 32-thread queues complete"
