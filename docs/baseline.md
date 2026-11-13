# 单文件 Disk+PQ baseline

Baseline 把一个节点的两份精确向量和完整 alpha 拓扑放在同一个对齐记录中，不做拓扑／坐标解耦，也不经过 Starling。它与分离式 Full 使用同一类构图语义，但 writer、物理布局和 reader 不同，不能仅把 `mode=6` 改成 `0` 就复用同一个搜索二进制。

这里描述根目录的兼容实现。近期统一距离目标的三臂实验使用另一个隔离搜索器，入口见[预实验说明](../experiments/README.md)。

## 准备与构建索引

先按[首页](../README.md)生成 `build/`，再在仓库根目录设置路径。示例使用 node3 的全量 OpenImages 数据和对应 GT：

```bash
export HVS_ROOT="$PWD"
export DATA_ROOT=/mnt/nvme3/wz/dataset/Openimage_disk
export BASELINE_ROOT="$HOME/hvs-runs/openimages-baseline"
mkdir -p "$BASELINE_ROOT/index"

"$HVS_ROOT/build/tests/build_hybrid_disk_index" \
  float "$DATA_ROOT/base_img_emb.fvecs" "$DATA_ROOT/base_text_emb.fvecs" \
  "$BASELINE_ROOT/index/" \
  40 100 64 64 8 l2 1
```

最后的 `1` 表示输出单文件索引。目录中应有 `single_index`、两套 PQ 文件、`medoids.bin` 和 `mem.index`。不压缩的 writer 保留完整 alpha ranges，不应拿旧的、已经截成两段区间的 single_index 替代 Full baseline。

## 单独编译搜索器

```bash
export ADDITIONAL_DEFINITIONS="-DNO_MAPPING"
cmake -S "$HVS_ROOT" -B "$HVS_ROOT/build-single-baseline" \
  -DBUILD_WITH_PQ=ON \
  -DCMAKE_CXX_COMPILER=g++-11
cmake --build "$HVS_ROOT/build-single-baseline" -j8 --target search_disk_index
```

这个构建不要定义 `USE_TOPO_DISK` 或 `REORDER_COMPUTE_PQ`。单文件 metadata 的 `nnodes_per_sector=0` 表示独立对齐的节点记录，`NO_MAPPING` 让节点 ID 对应其物理记录；分离式 reader 会按不同格式计算偏移，不能读取它。之后若重新配置分离式 `build/`，记得恢复首页的 `ADDITIONAL_DEFINITIONS`。

## 执行搜索

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$HVS_ROOT/build-single-baseline/tests/search_disk_index" \
  float "$BASELINE_ROOT/index/" \
  4 4 \
  "$DATA_ROOT/query_img_emb.fvecs" \
  "$DATA_ROOT/query_text_emb.fvecs" \
  "$DATA_ROOT/base_gt_round_00/query_alpha.fvecs" \
  "$DATA_ROOT/base_gt_round_00/top10_results.ivecs" \
  10 l2 0 0 0 \
  20 40 80 100 200
```

这里是四线程、beam width 4、top-10，使用 `mode=0, mem_L=0, strategy=0`。同样要校验数据和 GT 的对应关系，并在等召回下比较性能。

## 已有验收记录

2026-09-12 的 node3 10K OpenImages／100-query 测试中，单文件与 `mem.index` 的 degree、邻居 ID 和全部 alpha range 比较均为零 mismatch，实际 header 为 degree 54、5 个 range slots。该批 `L=20/40/80/100/200` 的 Recall@10 为 `77.20/87.40/93.20/94.80/98.20%`；这些数字只用于定位这批历史验收，不代表本文全量示例的预期曲线。

记录在 `/mnt/nvme3/wz/hvs-disk-validation-20260910/single-file-full-alpha-validation-20260912/`，详见[历史交接记录](archive/handoff-notes-2026-09.md)。
