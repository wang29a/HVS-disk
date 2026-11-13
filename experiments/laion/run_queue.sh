#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/nvme3/wz/hvs-disk-build-20261027
EXP="$ROOT/experiments/laion"
SRC=/mnt/nvme3/wz/Index/Laion-shard
DATA=/mnt/nvme2/wz/disk/dataset/Laion
STAR=/mnt/nvme3/wz/starling-wang29a
BIN="$ROOT/build-separated/tests"
VALIDATOR="$EXP/scripts/validate_laion.py"
mkdir -p "$EXP/logs" "$EXP/analysis" "$EXP/full" "$EXP/merge" "$EXP/no-alpha" "$EXP/baseline"
test ! -e "$EXP/queue.started-at"
date -Is > "$EXP/queue.started-at"
exec > >(tee -a "$EXP/logs/queue.log") 2>&1
finish() {
  local result=$?
  printf '%s\n' "$result" > "$EXP/queue.exit-code"
  date -Is > "$EXP/queue.finished-at"
  if [[ "$result" == 0 ]]; then echo LAION_QUEUE_COMPLETE; else echo "LAION_QUEUE_FAILED exit=$result stage=$(cat "$EXP/current-stage.txt")"; fi
}
trap finish EXIT
phase() { printf '%s\n' "$1" > "$EXP/current-stage.txt"; echo "STAGE=$1 time=$(date -Is)"; }
bound() { numactl --cpunodebind=0,1 --interleave=0,1 "$@"; }
link_input() {
  local target="$1" destination="$2"
  test -f "$target"
  if [[ -e "$destination" || -L "$destination" ]]; then
    test "$(readlink -f "$destination")" = "$(readlink -f "$target")"
  else
    ln -s "$target" "$destination"
  fi
}
export LD_LIBRARY_PATH="/opt/gcc-11.4/lib64:$STAR/release-gcc11-abi0/gnu_11.4_cxx17_64_release:/mnt/nvme2/wz/disk/local/openblas/lib:${LD_LIBRARY_PATH:-}"
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export OMP_NUM_THREADS=32 OMP_PROC_BIND=close OMP_PLACES=cores

phase provenance
cd "$ROOT/source"
test "$(git rev-parse HEAD)" = d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e
test -z "$(git status --porcelain)"
git log -1 --format=fuller
cd "$STAR"
test "$(git rev-parse HEAD)" = 7437c4848a83e7bf62558c5eab1eb999feed8537
git status --short > "$EXP/logs/starling-status.txt"
git diff > "$EXP/logs/starling-root.diff"
git submodule status > "$EXP/logs/starling-submodules.txt"
if [[ -d "$STAR/graph_partition/.git" || -f "$STAR/graph_partition/.git" ]]; then
  (cd "$STAR/graph_partition" && git diff) > "$EXP/logs/starling-partitioner.diff"
fi
uname -a
lscpu
numactl --hardware
df -h "$ROOT" "$DATA"
env | grep -E '^(LD_LIBRARY_PATH|CPATH|LIBRARY_PATH|OPENBLAS_NUM_THREADS|MKL_NUM_THREADS|OMP_NUM_THREADS|OMP_PROC_BIND|OMP_PLACES)='
sha256sum "$BIN/search_disk_index" "$BIN/merge_alpha" "$BIN/remove_alpha" "$BIN/pad_partition" \
  "$ROOT/build-single-baseline/tests/search_disk_index" \
  "$STAR/release-gcc11-abi0/graph_partition/partitioner" \
  "$STAR/release-gcc11-abi0/tests/utils/index_relayout_custom" \
  "$EXP/scripts/validate_laion.py" "$EXP/scripts/run_queue.sh" > "$EXP/analysis/tool-sha256.txt"
python3 "$EXP/scripts/record_manifest.py" --root "$EXP" --source "$SRC" --data "$DATA" \
  > "$EXP/logs/manifest.log" 2>&1

phase source-validation
bound python3 "$VALIDATOR" source --meta "$SRC/_disk.index.bp" \
  --topo "$SRC/disk_index_graph.bp" --coord "$SRC/disk_index_data" \
  --single "$SRC/single_index" --base1 "$DATA/base_img_emb.fvecs" \
  --base2 "$DATA/base_text_emb.fvecs" --output "$EXP/analysis/source-validation.json" \
  > "$EXP/logs/01-source-validation.log" 2>&1

phase ground-truth-validation
bound python3 "$VALIDATOR" queries --base1 "$DATA/base_img_emb.fvecs" \
  --base2 "$DATA/base_text_emb.fvecs" --query1 "$DATA/query_img_emb.fvecs" \
  --query2 "$DATA/query_text_emb.fvecs" --alpha "$DATA/base_gt_round_00/query_alpha.fvecs" \
  --gt "$DATA/base_gt_round_00/top10_results.ivecs" \
  --output "$EXP/analysis/query-validation.json" > "$EXP/logs/02-query-validation.log" 2>&1

phase conversion
link_input "$SRC/_disk.index.bp" "$EXP/full/_disk.index"
link_input "$SRC/disk_index_graph.bp" "$EXP/full/disk_index_graph.original"
link_input "$SRC/single_index" "$EXP/baseline/single_index"
bound "$BIN/merge_alpha" "$EXP/full/_disk.index" "$EXP/full/disk_index_graph.original" \
  "$EXP/merge/_disk.index" "$EXP/merge/disk_index_graph.original" 2 \
  > "$EXP/logs/03-merge.log" 2>&1
bound "$BIN/remove_alpha" "$EXP/full/_disk.index" "$EXP/full/disk_index_graph.original" \
  "$EXP/no-alpha/_disk.index" "$EXP/no-alpha/disk_index_graph.original" \
  > "$EXP/logs/04-remove-alpha.log" 2>&1
bound python3 "$VALIDATOR" converted --root "$EXP" \
  --output "$EXP/analysis/conversion-validation.json" > "$EXP/logs/05-conversion-validation.log" 2>&1

for kind in full merge no-alpha; do
  phase "starling-$kind"
  bound "$STAR/release-gcc11-abi0/graph_partition/partitioner" \
    --data_type float --gp_file "$EXP/$kind/partition.bin" \
    --custom_graph 1 --graph_file "$EXP/$kind/disk_index_graph.original" \
    --meta_file "$EXP/$kind/_disk.index" --ep_size 22 -T 32 -L 8 \
    > "$EXP/logs/06-starling-$kind.log" 2>&1
  bound "$STAR/release-gcc11-abi0/tests/utils/index_relayout_custom" --custom \
    "$EXP/$kind/disk_index_graph.original" "$EXP/$kind/_disk.index" \
    "$EXP/$kind/partition.bin" "$EXP/$kind/disk_index_graph.relayout" \
    > "$EXP/logs/07-relayout-$kind.log" 2>&1
  "$BIN/pad_partition" "$EXP/$kind/partition.bin" > "$EXP/logs/08-pad-$kind.log" 2>&1
  phase "validate-relayout-$kind"
  bound python3 "$VALIDATOR" relayout --root "$EXP" --kind "$kind" \
    --output "$EXP/analysis/relayout-$kind.json" > "$EXP/logs/09-validate-$kind.log" 2>&1
  link_input "$EXP/$kind/disk_index_graph.relayout" "$EXP/$kind/disk_index_graph"
  link_input "$EXP/$kind/partition.bin.aligned" "$EXP/$kind/_partition.bin.aligned"
  link_input "$SRC/disk_index_data" "$EXP/$kind/disk_index_data"
done
for kind in baseline full merge no-alpha; do
  for file in emb_pq_compressed.bin emb_pq_pivots.bin loc_pq_compressed.bin loc_pq_pivots.bin; do
    link_input "$SRC/$file" "$EXP/$kind/$file"
  done
done

run_search() {
  local threads="$1" kind="$2" log="$3"
  shift 3
  local binary mode strategy
  if [[ "$kind" == baseline ]]; then
    binary="$ROOT/build-single-baseline/tests/search_disk_index"; mode=0; strategy=0
  else
    binary="$BIN/search_disk_index"; mode=6; strategy=17
  fi
  export OMP_NUM_THREADS="$threads"
  echo "RUN threads=$threads kind=$kind log=$log start=$(date -Is)"
  /usr/bin/time -v numactl --cpunodebind=0,1 --interleave=0,1 \
    "$binary" float "$EXP/$kind/" "$threads" 4 \
    "$DATA/query_img_emb.fvecs" "$DATA/query_text_emb.fvecs" \
    "$DATA/base_gt_round_00/query_alpha.fvecs" "$DATA/base_gt_round_00/top10_results.ivecs" \
    10 l2 "$mode" 0 "$strategy" "$@" > "$log" 2>&1
  echo "DONE threads=$threads kind=$kind end=$(date -Is)"
}

phase smoke
for kind in baseline full merge no-alpha; do
  run_search 32 "$kind" "$EXP/logs/smoke-$kind.log" 40 100
done
python3 "$EXP/scripts/summarize_laion.py" --root "$EXP" --check-smoke

LS=(20 40 60 80 100 120 160 200 300 400)
for threads in 32 64; do
  mkdir -p "$EXP/logs/threads-$threads"
  for rep in 1 2 3; do
    case "$rep" in
      1) kinds=(baseline full merge no-alpha) ;;
      2) kinds=(full merge no-alpha baseline) ;;
      3) kinds=(merge no-alpha baseline full) ;;
    esac
    for kind in "${kinds[@]}"; do
      phase "search-T${threads}-rep${rep}-$kind"
      run_search "$threads" "$kind" "$EXP/logs/threads-$threads/rep${rep}-$kind.log" "${LS[@]}"
    done
  done
done

phase characterization
bound python3 "$EXP/scripts/analyze_alpha_metadata.py" --dataset laion \
  --full-meta "$EXP/full/_disk.index" --full-topo "$EXP/full/disk_index_graph.original" \
  --merge-meta "$EXP/merge/_disk.index" --merge-topo "$EXP/merge/disk_index_graph.original" \
  --query-alpha "$DATA/base_gt_round_00/query_alpha.fvecs" \
  --output "$EXP/analysis/alpha-characterization.json" > "$EXP/logs/10-characterization.log" 2>&1
phase summarize
python3 "$EXP/scripts/summarize_laion.py" --root "$EXP"
phase complete
