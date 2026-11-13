#!/usr/bin/env bash
set -euo pipefail

HVS=/mnt/nvme3/wz/hvs-disk-paper-20260920
EXP="$HVS/experiments/formal-32/cc3m"
SRC=/mnt/nvme3/wz/Index/CC3M-shard
DATA=/mnt/nvme3/wz/dataset/CC3M_disk
STAR=/mnt/nvme3/wz/starling-wang29a
LOG="$EXP/logs"
mkdir -p "$LOG" "$EXP/baseline" "$EXP/full" "$EXP/merge" "$EXP/no-alpha"
exec > >(tee -a "$LOG/queue.log") 2>&1

date -Is
echo "commit=d2a39e2 threads=32 beam=4 numa=0,1 repeats=3"
lscpu | grep -E 'Model name|Socket|NUMA|CPU\(s\)'
numactl --hardware

echo "[1/5] Build the strict single-file baseline reader"
cd "$HVS"
export ADDITIONAL_DEFINITIONS="-DNO_MAPPING"
cmake -S . -B build-single-baseline -DBUILD_WITH_PQ=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-single-baseline -j32 --target search_disk_index
unset ADDITIONAL_DEFINITIONS

echo "[2/5] Assemble baseline, Full, and Merge search roots"
ln -sfn "$SRC/single_index" "$EXP/baseline/single_index"
for f in emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
  ln -sfn "$SRC/$f" "$EXP/baseline/$f"
done

ln -sfn "$SRC/_disk.index" "$EXP/full/_disk.index"
ln -sfn "$SRC/disk_index_graph.relayout" "$EXP/full/disk_index_graph"
ln -sfn "$SRC/_part.bin.aligned" "$EXP/full/_partition.bin.aligned"
ln -sfn "$SRC/_disk.index.ma" "$EXP/merge/_disk.index"
ln -sfn "$SRC/disk_index_graph.relayout.ma" "$EXP/merge/disk_index_graph"
ln -sfn "$SRC/_part.bin.ma.aligned" "$EXP/merge/_partition.bin.aligned"
for kind in full merge; do
  for f in disk_index_data emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
    ln -sfn "$SRC/$f" "$EXP/$kind/$f"
  done
done

echo "[3/5] Build No-alpha from the same Full topology"
if [[ ! -s "$EXP/no-alpha/_disk.index" || ! -s "$EXP/no-alpha/disk_index_graph.original" ]]; then
  "$HVS/build-paper/tests/remove_alpha" \
    "$SRC/_disk.index" "$SRC/disk_index_graph" \
    "$EXP/no-alpha/_disk.index" "$EXP/no-alpha/disk_index_graph.original"
fi

if [[ ! -s "$EXP/no-alpha/partition.bin" ]]; then
  export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:$STAR/release-gcc11-abi0/gnu_11.4_cxx17_64_release:${LD_LIBRARY_PATH:-}"
  cd "$STAR"
  ./release-gcc11-abi0/graph_partition/partitioner \
    --data_type float --gp_file "$EXP/no-alpha/partition.bin" \
    --custom_graph 1 --graph_file "$EXP/no-alpha/disk_index_graph.original" \
    --meta_file "$EXP/no-alpha/_disk.index" --ep_size 13 -T 32 -L 8
fi

if [[ ! -s "$EXP/no-alpha/disk_index_graph.relayout" ]]; then
  export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:$STAR/release-gcc11-abi0/gnu_11.4_cxx17_64_release:${LD_LIBRARY_PATH:-}"
  cd "$STAR"
  ./release-gcc11-abi0/tests/utils/index_relayout_custom --custom \
    "$EXP/no-alpha/disk_index_graph.original" "$EXP/no-alpha/_disk.index" \
    "$EXP/no-alpha/partition.bin" "$EXP/no-alpha/disk_index_graph.relayout"
fi

cd "$HVS"
if [[ ! -s "$EXP/no-alpha/partition.bin.aligned" ]]; then
  "$HVS/build-paper/tests/pad_partition" "$EXP/no-alpha/partition.bin"
fi
ln -sfn "$EXP/no-alpha/disk_index_graph.relayout" "$EXP/no-alpha/disk_index_graph"
ln -sfn "$EXP/no-alpha/partition.bin.aligned" "$EXP/no-alpha/_partition.bin.aligned"
for f in disk_index_data emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
  ln -sfn "$SRC/$f" "$EXP/no-alpha/$f"
done

echo "[4/5] Record final artifact identities"
for kind in baseline full merge no-alpha; do
  find -L "$EXP/$kind" -maxdepth 1 -type f -printf "$kind %f %s\n" | sort
done

QUERY1="$DATA/query_img_emb.fvecs"
QUERY2="$DATA/query_text_emb.fvecs"
ALPHA="$DATA/base_gt_round_00/query_alpha.fvecs"
GT="$DATA/base_gt_round_00/top10_results.ivecs"
LS=(20 40 60 80 100 120 160 200 300 400)
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores

run_one() {
  local rep="$1" kind="$2" mode strategy binary
  if [[ "$kind" == baseline ]]; then
    binary="$HVS/build-single-baseline/tests/search_disk_index"
    mode=0
    strategy=0
  else
    binary="$HVS/build-paper/tests/search_disk_index"
    mode=6
    strategy=17
  fi
  echo "RUN rep=$rep kind=$kind start=$(date -Is)"
  /usr/bin/time -v numactl --cpunodebind=0,1 --interleave=0,1 \
    "$binary" float "$EXP/$kind/" 32 4 \
    "$QUERY1" "$QUERY2" "$ALPHA" "$GT" \
    10 l2 "$mode" 0 "$strategy" "${LS[@]}" \
    > "$LOG/rep${rep}-${kind}.log" 2>&1
  echo "DONE rep=$rep kind=$kind end=$(date -Is)"
}

echo "[5/5] Run three serialized, rotated repetitions"
for kind in baseline full merge no-alpha; do run_one 1 "$kind"; done
for kind in full merge no-alpha baseline; do run_one 2 "$kind"; done
for kind in merge no-alpha baseline full; do run_one 3 "$kind"; done

date -Is
echo "CC3M formal 32-thread four-scheme queue complete"
