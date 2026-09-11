# hvs-disk Agent Guide

hvs-disk is a C++17 implementation of SSD-resident dual-space graph ANN search. A query uses `alpha` to blend two distance spaces. The disk topology supports Full-alpha, Merge-alpha, and No-alpha layouts. Read `README.md` before running experiments; it is the authoritative source for binary formats, the verified Starling revision, node3 paths, parameter meanings, and expected results.

## Build

Build and run experiments on Linux. After changing C++ code, rebuild and run a focused validation:

```bash
export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build -DBUILD_WITH_PQ=ON -DCMAKE_CXX_COMPILER=g++-11
cmake --build build -j8
```

## Build the three index layouts

The two base matrices must use the headered binary format documented in `README.md`, contain the same number of rows, and use the same row-to-ID ordering. First build the Full-alpha index in separated-file mode (`single_file_index=0`):

```bash
mkdir -p "$EXP_ROOT/full" "$EXP_ROOT/merge" "$EXP_ROOT/no-alpha"

./build/tests/build_hybrid_disk_index \
  float "$BASE_SPACE_1" "$BASE_SPACE_2" "$EXP_ROOT/full/" \
  40 100 64 64 16 l2 0
```

Then derive Merge-alpha and No-alpha from that same Full-alpha metadata and topology:

```bash
./build/tests/merge_alpha \
  "$EXP_ROOT/full/_disk.index" "$EXP_ROOT/full/disk_index_graph" \
  "$EXP_ROOT/merge/_disk.index" "$EXP_ROOT/merge/disk_index_graph" 2

./build/tests/remove_alpha \
  "$EXP_ROOT/full/_disk.index" "$EXP_ROOT/full/disk_index_graph" \
  "$EXP_ROOT/no-alpha/_disk.index" "$EXP_ROOT/no-alpha/disk_index_graph"
```

Run the external Starling implementation pinned in `README.md` separately for `full`, `merge`, and `no-alpha`. For each layout, use its own metadata and topology to produce its own `partition.bin`, relayout topology, and `partition.bin.aligned`. Never reuse a partition or reorder mapping across layouts: their record widths and nodes-per-page differ. After relayout, retain the original 8192-byte metadata file; do not install Starling's 4096-byte generated metadata copy.

Each layout's `search/` directory must expose its own `_disk.index`, relayout topology as `disk_index_graph`, and aligned partition as `_partition.bin.aligned`. It may share `disk_index_data`, both `*_pq_compressed.bin` files, and both `*_pq_pivots.bin` files from the Full build. The exact Starling and symlink commands are in `README.md` Section 7.3.

## Search

Use the Starling page-rerank path for the three-way comparison. Query matrices, the one-column alpha matrix, and ground truth must have matching query counts and correspond to the same base dataset:

```bash
./build/tests/search_disk_index \
  float "$EXP_ROOT/full/search/" \
  8 4 \
  "$QUERY_SPACE_1" "$QUERY_SPACE_2" "$QUERY_ALPHA" "$GROUND_TRUTH" \
  10 l2 6 0 17 \
  20 40 80 100 200
```

Repeat the command with `merge/search/` and `no-alpha/search/`. Here `mode=6`, `mem_L=0`, and `strategy=17` select the verified Starling page-rerank path. No-alpha has `max_alpha_range_len=0`, which means every neighbor is eligible for expansion; it must not be treated as an empty active range. At the same search depth, No-alpha normally performs more distance comparisons than Full/Merge.

Before accepting an index, require a complete partition bijection, valid neighbor IDs and degrees, correct interval bounds, and zero full-record mismatches between the original and relayout topology. Compare performance at matched Recall and record the commit, index identity, all search parameters, hardware, and raw logs.

## Repository scripts

The scripts under `scripts/` are inherited experiment drivers, not the canonical Full/Merge/No-alpha reproduction workflow. Use the explicit commands above and `README.md` Section 7.3 for the handoff validation. Do not use `shell/` or `_scripts/` as a substitute; those directories contain older PipeANN/OdinANN workflows and layouts.

Before using a repository script, read the entire file and inspect its changes:

```bash
sed -n '1,240p' scripts/config.sh
sed -n '1,260p' scripts/run_exp1.sh  # replace with the intended script
git status --short
```

`scripts/config.sh` defines `PROJECT_PATH`, `DATA_PATH`, `DISKANN_INDEX_PATH`, `OUTPUT_PATH`, and `LOG_PATH`. Its checked-in values are historical placeholders. Copy the script to a private experiment workspace or update a local, uncommitted copy with valid Linux paths before running any driver. Ensure output and log directories are separate from datasets and committed source.

`scripts/preprocess.sh` is a legacy single-space DiskANN preprocessing driver. It invokes `build_disk_index` and the obsolete 4096-byte `split_index` path, so it must not be used to create the current dual-space alpha indexes. Use `build_hybrid_disk_index`, `merge_alpha`, and `remove_alpha` as shown above instead.

`scripts/run_exp1.sh` through `scripts/run_exp5.sh` reproduce older update/performance experiments for specific datasets and systems. They contain fixed dataset names, strategies, compile definitions, copy/remove operations, and in some cases `sed -i` edits to C++ source before rebuilding. Run one only after adapting every path and parameter, recording the starting commit, and confirming that its search mode matches the intended index layout:

```bash
N=1  # choose 1, 2, 3, 4, or 5
bash -n scripts/config.sh "scripts/run_exp${N}.sh"
git diff --exit-code
bash "scripts/run_exp${N}.sh" 2>&1 | tee "$LOG_PATH/run_exp${N}.log"
git status --short
```

Do not proceed when `git diff --exit-code` reports pre-existing source changes unless those changes are understood and intentionally part of the experiment. After a script finishes, inspect `git diff` because some legacy drivers modify source code. A successful shell exit or high QPS alone is not validation; check Recall, Mean Cmps, Mean I/O, and the index-format invariants.

## Working rules

Use `rg` for code discovery and preserve unrelated user changes. An 8192-byte page and fixed-width topology records are format invariants; layout changes must update metadata, builders, converters, offset helpers, all search readers, and Starling handling together. Do not commit datasets, generated indexes, build outputs, logs, or paper material. Do not run `git add`, `git commit`, `git push`, or destructive Git commands unless explicitly requested.
