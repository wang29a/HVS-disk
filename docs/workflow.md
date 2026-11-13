# 从数据到三方案搜索

本页使用根目录构建的 `build_hybrid_disk_index`、`merge_alpha`、`remove_alpha` 和 `search_disk_index`，完成分离式 Full/Merge/No-alpha 的流程。复现近期修正距离目标后的研究结果，请走[预实验入口](../experiments/README.md)，两者的二进制和图不能混用。

## 设置本次运行的路径

先按[首页](../README.md)完成编译，再在仓库根目录设置路径。以下数据路径是 node3 的 OpenImages 示例，输出放到自己的新目录；不会自动获取数据或生成配套 GT。

```bash
export HVS_ROOT="$PWD"
export DATA_ROOT=/mnt/nvme3/wz/dataset/Openimage_disk
export EXP_ROOT="$HOME/hvs-runs/openimages"
export STARLING_ROOT=/mnt/nvme3/wz/starling-wang29a

export BASE_X="$DATA_ROOT/base_img_emb.fvecs"
export BASE_Y="$DATA_ROOT/base_text_emb.fvecs"
export QUERY_X="$DATA_ROOT/query_img_emb.fvecs"
export QUERY_Y="$DATA_ROOT/query_text_emb.fvecs"
export QUERY_ALPHA="$DATA_ROOT/base_gt_round_00/query_alpha.fvecs"
export GT="$DATA_ROOT/base_gt_round_00/top10_results.ivecs"

mkdir -p "$EXP_ROOT/full" "$EXP_ROOT/merge" "$EXP_ROOT/no-alpha"
```

这些文件必须满足[数据格式](data-format.md)中的对应关系。如果改用 base 子集，必须同时使用该子集对应的 GT，不能沿用全库 GT。

## 构建 Full-alpha

```bash
"$HVS_ROOT/build/tests/build_hybrid_disk_index" \
  float "$BASE_X" "$BASE_Y" "$EXP_ROOT/full/" \
  40 100 64 64 8 l2 0
```

参数顺序为：

```text
数据类型、Base X、Base Y、输出前缀、R、L_build、B、M、构图线程数、距离、single_file_index
```

示例的 `R=40` 是构图目标度数，`L_build=100` 是构图候选长度；`B=64`、`M=64` 是 CLI 的内存预算参数（GB），**不是两空间各 64 字节 PQ 的意思**。实际 PQ 设置以实现和构建日志为准。最后的 `0` 选择拓扑／坐标分离，`1` 则生成[单文件 baseline](baseline.md)。索引前缀需保留结尾的 `/`。

完成后，`full/` 中应有 `_disk.index`、`disk_index_graph`、`disk_index_data`、两套 `emb/loc` PQ 文件，以及构图产物。实际 `max_nbr_len`、区间槽数和 nodes/page 从 metadata 读取；历史根目录构图器的合并图可能超过名义 `R`，不要在转换工具中硬编码 40。

## 派生 Merge 和 No-alpha

从同一份、尚未 Starling 重排的 Full topology 派生两个方案：

```bash
"$HVS_ROOT/build/tests/merge_alpha" \
  "$EXP_ROOT/full/_disk.index" "$EXP_ROOT/full/disk_index_graph" \
  "$EXP_ROOT/merge/_disk.index" "$EXP_ROOT/merge/disk_index_graph" 2

"$HVS_ROOT/build/tests/remove_alpha" \
  "$EXP_ROOT/full/_disk.index" "$EXP_ROOT/full/disk_index_graph" \
  "$EXP_ROOT/no-alpha/_disk.index" "$EXP_ROOT/no-alpha/disk_index_graph"
```

`merge_alpha` 的 method 2 贪心合并最小间隙，method 3 使用 max-coverage；输出均为至多两段区间。`remove_alpha` 只保留邻居 ID。两者改变记录宽度，必须完整重写 topology，不能只改 metadata 头。

## 分别用 Starling 重排

先按 [Starling 使用说明](starling.md)安装并构建外部工具，再对三种方案分别运行 partition、custom relayout 和 `pad_partition`。每种方案都使用自己的 metadata、记录宽度和 partition，不能复用另一方案的顺序。后续只重排拓扑，坐标和 PQ 共享 Full 构建产物。

## 组装搜索目录

Starling 完成后，每种方案都应得到 `disk_index_graph.relayout` 和 `partition.bin.aligned`。下面为每种方案安装 loader 需要的统一名称，命令面向新的 `search/` 目录：

```bash
for kind in full merge no-alpha; do
  mkdir -p "$EXP_ROOT/$kind/search"
  ln -s ../_disk.index "$EXP_ROOT/$kind/search/_disk.index"
  ln -s ../disk_index_graph.relayout "$EXP_ROOT/$kind/search/disk_index_graph"
  ln -s ../partition.bin.aligned "$EXP_ROOT/$kind/search/_partition.bin.aligned"
  for name in disk_index_data emb_pq_compressed.bin emb_pq_pivots.bin \
              loc_pq_compressed.bin loc_pq_pivots.bin; do
    ln -s "../../full/$name" "$EXP_ROOT/$kind/search/$name"
  done
done
```

搜索目录共八个文件名：该方案的 metadata、重排 topology、aligned partition，以及共享坐标和四份 PQ 文件。Metadata 继续使用原 8192 字节文件，不能用 Starling 输出的 4096 字节副本替换。

## 执行搜索

```bash
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

"$HVS_ROOT/build/tests/search_disk_index" \
  float "$EXP_ROOT/full/search/" \
  4 4 \
  "$QUERY_X" "$QUERY_Y" "$QUERY_ALPHA" "$GT" \
  10 l2 6 0 17 \
  20 40 80 100 200
```

依次将 `full/search/` 换成 `merge/search/`、`no-alpha/search/`。第一组 `4 4` 分别是查询线程数和 beam/I/O width，`10` 是 top-k，末尾列表是搜索深度 L。`mode=6` 选 Starling page-rerank，`mem_L=0` 不启用可选入口内存索引，`strategy=17` 是这条历史验证路径的配置；它本身不能证明跨查询共享缓存已生效，缓存实验需使用专门驱动并记录实际预算与命中。

命令行的完整顺序为：

```text
数据类型、索引前缀、线程数、beam width、Query X、Query Y、Alpha、GT、K、距离、mode、mem_L、strategy、L...
```

`search_disk_index` 内部 Usage 仍有旧单查询描述，参数以此处及 `tests/search_disk_index.cpp` 的解析代码为准。

## 如何确认跑对了

程序退出成功和 QPS 非零还不够。先核对 metadata 的节点数、维度和记录宽度，再检查 partition 是全部节点 ID 的双射，重排前后的完整记录一致，邻居 ID、degree、区间端点和末尾 padding 合法，最后用匹配 GT 做 Recall 冒烟测试。

`tests/check_reorder_topo.cpp` 仍含旧的 4096B／132B 布局硬编码，不能直接验证本格式。已有新实验的校验器见[代码与索引参考](implementation.md)。历史 10K 验收和旧索引兼容测试保留在[交接记录存档](archive/handoff-notes-2026-09.md)，其中数字属于对应的历史数据与配置，不是所有运行都应复现的固定 Recall。

长实验放在 tmux 中串行运行，先确认主机使用约束，设置线程数及 NUMA。近期预实验是四线程、NUMA 3；具体 CPU 编号是 node3 配置，不能照搬到另一台机器。比较性能使用等召回工作点，同时记录 L、精排深度、缓存、PQ、线程、I/O 和代码／索引身份。
