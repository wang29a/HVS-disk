# hvs-disk 代码交接说明

本仓库是 hvs-disk 的代码交接版本。hvs-disk 面向 **SSD 常驻的双空间图近似最近邻搜索**：查询通过 `alpha` 融合两个距离空间，并可根据边上的 alpha 有效区间决定是否扩展该边。

这份交接包只保留实现、构建文件和运行脚本，不包含论文草稿、写作材料、实验图片、日志、数据集、索引文件和编译产物。Git 历史已压缩为一个交接基线提交；后续开发可以直接在此基础上创建分支和提交。

当前交接版本为 `ea4ac09`。第一次接手时，先在 Linux/node3 上运行 `git status --short --branch` 和 `git log -1 --oneline` 确认版本，再按这条唯一主线执行：第 4 节编译并构建 Full-alpha，第 7.1/7.2 节转换 No-alpha/Merge-alpha，第 7.3 节让三种格式分别执行 Starling，最后使用第 7.3.5 节的 `mode=6、mem_L=0、strategy=17` 命令搜索。第 6 节只用于兼容性检查历史 OpenImages 原始索引，不是三方案端到端实验的主入口。

## 外部 Starling 仓库

hvs-disk 的 page-aware graph partition 与 topology relayout 依赖外部 Starling，实现不包含在本交接仓库中。统一使用：

- 仓库：[wang29a/starling](https://github.com/wang29a/starling)
- 克隆地址：`https://github.com/wang29a/starling.git`
- node3 已验证提交：`7437c4848a83e7bf62558c5eab1eb999feed8537`
- node3 安装路径：`/mnt/nvme3/wz/starling-wang29a`

不要使用其他同名 Starling 仓库代替，也不要把 Starling 源码复制进 hvs-disk 后一起提交。完整的 node3 编译、Full/Merge/No-alpha partition、relayout、metadata 注意事项和搜索验证流程见 [7.3 Starling relayout](#73-starling-relayout)。

node3 另有一份面向后续 agent 的完整实验执行手册：

```text
/mnt/nvme3/wz/hvs-disk-workflow/Openimage/full-starling-7437c484/EXPERIMENT_WORKFLOW.md
/mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902/E2E_FULL_MERGE_RUN.md
```

第一份是通用操作手册；第二份是已经实际跑通的“构建 Full → 转换 Merge → 两套 Starling reorder → 两套搜索”逐命令记录。配套输入、ground truth、索引与日志均在对应稳定目录，不依赖 `/tmp`。

## 1. 建议先了解的内容

核心距离定义为：

```text
combined_dist = alpha * dist_space_1 + (1 - alpha) * dist_space_2
```

- `alpha = 0` 时只使用空间 2。
- `alpha = 1` 时只使用空间 1。
- 粗排阶段直接融合两个空间的 PQ 距离。
- 精排阶段当前使用 `alpha * sqrt(exact_dist_1) + (1-alpha) * sqrt(exact_dist_2)`，不要把它与 PQ 粗排公式混为一谈。
- 每条边可以有一个或多个有效区间；查询 alpha 落入任一区间时，该边才参与扩展。

建议按以下顺序阅读：

1. `include/neighbor.h`：构建期和序列化后的邻居表示。
2. `include/ssd_index.h`、`src/ssd_index.cpp`：SSD 索引格式、元数据与加载逻辑。
3. `src/hybrid_index.cpp`：alpha 区间构建和邻居裁剪。
4. `src/search/page_search.cpp`：带 alpha 过滤的页级搜索。
5. `src/search/starling_page_rerank_search.cpp`：重排、缓存和页级精排路径。
6. `tests/build_hybrid_disk_index.cpp`、`tests/search_disk_index.cpp`：主要命令行入口。

## 2. 目录结构

```text
include/                 核心头文件、索引接口和查询缓冲区
src/                     索引构建、加载、搜索、更新、距离和 I/O 实现
src/search/              Beam、page、pipeline、coroutine、rerank 等搜索路径
src/buffer/              页/块缓存与序列化
src/update/              动态插入、删除与合并
tests/                   构建、搜索、格式转换和检查工具的可执行程序
tests/utils/             数据格式转换、采样与 ground truth 工具
scripts/                 当前实验和预处理脚本
shell/、_scripts/        历史或替代脚本，使用前必须核对参数和路径
python/                  Python 绑定或辅助代码
third_party/liburing/    随仓库提供的 liburing 源码
```

`src/search/` 中存在多个独立的邻居解析和搜索实现。修改磁盘记录格式或邻居布局时，必须逐个检查这些实现，不能只修改其中一个入口。

## 3. 开发环境

项目以 Linux 为目标环境，不建议在 macOS 上判断构建是否成功。需要：

- CMake 和支持 C++17 的 GCC/Clang；
- OpenMP；
- BLAS 或 MKL；
- jemalloc；
- 默认使用 Linux AIO（`libaio`）；也可以设置 `USE_AIO=OFF` 使用 liburing；
- 支持时启用 AVX2 或 AVX-512。

Ubuntu 环境可参考安装：

```bash
sudo apt install make cmake g++ libaio-dev libjemalloc-dev \
  libboost-all-dev libmkl-full-dev clang-format
```

不同服务器的 MKL/BLAS 安装方式可能不同。当前顶层 `CMakeLists.txt` 包含 `/usr/include/mkl`，如果服务器路径不同，需要按实际环境调整。

## 4. 编译

常规构建：

```bash
cmake -S . -B build -DBUILD_WITH_PQ=ON
cmake --build build -j
```

历史上常用的 hvs-disk 配置：

```bash
export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build \
  -DBUILD_WITH_PQ=ON \
  -DCMAKE_CXX_COMPILER=g++-11
cmake --build build -j
```

顶层 CMake 的重要选项和默认宏：

- `USE_AIO=ON`：使用 Linux AIO；设为 `OFF` 时链接 liburing。
- `BUILD_WITH_PQ=ON`：编译 PQ 生成逻辑。
- 默认定义 `NDEBUG`、`DELTA_PRUNING`、`OVERLAP_INIT` 和 `DYN_PIPE_WIDTH`。
- 环境变量 `ADDITIONAL_DEFINITIONS` 用于注入实验配置宏。
- Debug 构建会启用 AddressSanitizer。

不要盲目运行 `build.sh`：它固定使用 `g++-11`，并且旧写法假设 `build/` 不存在。优先使用上面的 `cmake -S/-B` 命令。

### 4.1 从原始双空间向量构建：当前验证结论

当前代码已修复分离式构建：`single_file_index=0` 会输出 Full-alpha 的 `_disk.index`、`disk_index_graph` 和 `disk_index_data`，而不是错误地始终生成 Merge-alpha `single_index`。Full 构建完成后，再使用 `merge_alpha` 生成 Merge-alpha；两种格式分别执行 Starling，不能共享 reorder 顺序。

2026-09-02 在 node3 使用 10,000 条 OpenImages、两侧各 768 维的子集进行验证：

```bash
./build/tests/build_hybrid_disk_index \
  float \
  /mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902/input/base_img_10000.bin \
  /mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902/input/base_text_10000.bin \
  /mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902/full/ \
  40 100 64 64 16 l2 0
```

构建程序成功生成两套 PQ 文件、`medoids.bin`、`mem.index`、8KB metadata、分离 coord 和 Full-alpha topology。构建参数 `R=40` 不保证最终序列化宽度仍为 40；本次合并图实际 `max_nbr_len=54`，后续工具均从 metadata 读取实际值。

本次 Full-alpha metadata：

```text
node_num=10000, dims=768+768, max_nbr_len=54
max_alpha_range_len=5, fixed_topo_size=760B, nodes/page=10
topology size=8,192,000B
```

随后使用 greedy method 2 转换出的 Merge-alpha metadata：

```text
node_num=10000, dims=768+768, max_nbr_len=54
max_alpha_range_len=2, fixed_topo_size=436B, nodes/page=18
topology size=4,554,752B
```

两种格式均完成 8 轮 Starling、relayout、aligned partition、全量 reorder 字节校验和带匹配 ground truth 的搜索，详细结果见 7.3.4。`single_file_index=1` 的旧 mode 0 路径仍不是当前交接主流程；`tests/split_index.cpp` 仍是不能处理本格式的 4096B 遗留工具。

### 4.2 单文件 Disk+PQ baseline（兼容路径）

仓库仍保留旧的单文件搜索路径，可用于检查“不做 topology/coordinate 解耦、不经过 Starling reorder”的 Disk+PQ baseline。它不是第 7 节 Full/Merge/No-alpha 三方案实验的主入口，必须使用独立构建目录和独立编译条件，不能复用带 `USE_TOPO_DISK` 的搜索二进制。

构建单文件索引时，将最后一个参数设为 `1`：

```bash
./build/tests/build_hybrid_disk_index \
  float "$BASE_SPACE_1" "$BASE_SPACE_2" "$BASELINE_ROOT/index/" \
  40 100 64 64 16 l2 1
```

生成目录包含 `single_index`、两套 PQ pivots/compressed 文件、`medoids.bin` 和 `mem.index`。旧单文件布局把每个节点的两份精确向量和拓扑记录放在同一个 8192B 对齐页中，metadata 的 `nnodes_per_sector` 为 `0`。因此搜索端必须关闭 `USE_TOPO_DISK`，并启用 `NO_MAPPING`，让节点 ID 直接对应单文件中的物理位置：

```bash
export ADDITIONAL_DEFINITIONS="-DNO_MAPPING"
cmake -S . -B build-single-baseline \
  -DBUILD_WITH_PQ=ON \
  -DCMAKE_CXX_COMPILER=g++-11
cmake --build build-single-baseline -j8 --target search_disk_index
```

不要为这个 baseline 定义 `USE_TOPO_DISK` 或 `REORDER_COMPUTE_PQ`。`USE_TOPO_DISK` 面向解耦后的 topology/coordinate 页面布局，会使用 `ntopo_per_sector` 和 `ncoord_per_sector`；将它用于 `nnodes_per_sector=0` 的旧单文件 metadata 会在加载阶段触发除零错误。仅仅移除 `USE_TOPO_DISK` 也不够：若未定义 `NO_MAPPING`，通用 page mapping 初始化会拒绝 `nnodes_per_sector=0`。

对应搜索入口是 `mode=0`：

```bash
./build-single-baseline/tests/search_disk_index \
  float "$BASELINE_ROOT/index/" \
  8 4 \
  "$QUERY_SPACE_1" "$QUERY_SPACE_2" "$QUERY_ALPHA" "$GROUND_TRUTH" \
  10 l2 0 0 0 \
  20 40 80 100 200
```

2026-09-12 在 node3 上使用 10,000 条 OpenImages base 和 100 条匹配查询完成了这条路径的实测。修正后的 single-file writer 在不压缩时继承构建期实际 `max_alpha_range_len`，只有显式选择 merge method 2/3 时才压缩到两个 ranges；它还按模板类型 `sizeof(T)` 计算向量区长度，不再把 `int8/uint8` 当作 `float` 布局。

本次 Full-alpha 单文件 header 为 `node_num=10000, dims=768+768, max_nbr_len=54, max_alpha_range_len=5, nnodes_per_sector=0`。对 10,000 个节点逐项比较 `mem.index` 与 `single_index` 后，degree、neighbor ID 和所有 alpha range 的 mismatch 均为 0，构建期实际最大 range 数为 5。mode 0 搜索及日志管道均返回 0；Recall@10 在 `L=20/40/80/100/200` 时分别为 `77.20%/87.40%/93.20%/94.80%/98.20%`。验证目录为 `/mnt/nvme3/wz/hvs-disk-validation-20260910/single-file-full-alpha-validation-20260912/`，构建和搜索日志分别是 `01_build.log` 和 `02_search.log`。

因此当前产物可以称为“Full-alpha 单文件、无解耦、无 Starling 的 Disk+PQ baseline”。它与解耦 Full-alpha 共享同一图构建语义和完整邻接信息，但磁盘布局及搜索 reader 不同。不要用旧的 `max_alpha_range_len=2` 单文件产物替代新 baseline；编译条件只能决定 reader，不能恢复旧产物在写盘时已经截断的 ranges。

### 4.3 推荐的正式索引流水线

目标流水线应保持以下顺序，尤其不要在不同记录宽度之间复用 reorder map：

```text
两份逐行对齐的 base bin
  -> 构建双空间内存图（变长 alpha ranges）
  -> 写出 Full-alpha 的 meta + coord + topology
  -> 选择且只选择一种拓扑格式
       Full:     原样保留
       Merge:    merge_alpha，重新写 meta + topology
       No-alpha: remove_alpha，重新写 meta + topology
  -> 对所选格式调用外部 Starling partition/reorder
  -> 安装为搜索目录中的统一文件名/符号链接
  -> 格式全量校验
  -> 小查询冒烟测试
  -> 使用匹配的 ground truth 验证 Recall
```

Merge/No-alpha 会改变 node record 大小和 nodes/page，必须先转换再调用 Starling。Full、Merge、No-alpha 应分别产生自己的 partition、reorder map 和 relayout topology；不能先对 Full 做 reorder，再把其映射或页布局直接套给另两种格式。

## 5. 搜索流程与实现变体

### 5.1 PQ 粗排与精确重排

`RERANK_SEARCH` 和 `PQ_PAGE_RERANK_SEARCH` 等 rerank 路径采用两阶段搜索。其他 mode 可能只执行图遍历或采用不同的 pipeline，不要笼统地假设所有搜索入口都会精排。

```text
Phase 1 — PQ coarse search (page-level)
  入口点 (medoids) -> 页面级 beam search
  -> 读取 topo page (SECTOR_LEN=8192)
  -> Full/Merge: 按 alpha range 过滤邻居
  -> No-alpha: 应展开全部邻居
  -> 按双空间 PQ 融合距离更新 retset
  -> 循环直到收敛或达到 l_search

Phase 2 — exact rerank
  候选集 -> 读取 coord 精确双向量
  -> exact = alpha*sqrt(dist_e) + (1-alpha)*sqrt(dist_l)
  -> 排序并返回 top-k_search
```

PQ 阶段和精排阶段的距离定义不同：

```text
PQ:    alpha * PQ_dist_e + (1-alpha) * PQ_dist_l
exact: alpha * sqrt(exact_dist_e) + (1-alpha) * sqrt(exact_dist_l)
```

### 5.2 邻居展开语义

Full/Merge 的每个邻居记录都带有固定数量的 alpha slots。搜索时解析这些 interval，查询 alpha 落入任意有效 interval 时才展开边。参见 `src/search/page_search.cpp` 和 `src/search/starling_page_rerank_search.cpp`。

```cpp
for (unsigned m = 0; m < nnbrs; ++m) {
  parse_neighbor_id_and_alpha_ranges();
  if (query_alpha * 100 落入任意有效区间) {
    expand_neighbor();
  }
}
```

No-alpha 记录只保存 neighbor ID，正确语义是不做 alpha 过滤：

```cpp
for (unsigned m = 0; m < nnbrs; ++m) {
  uint32_t neighbor_id = read_neighbor_id(m);  // stride=4B
  if (!visited(neighbor_id)) expand_neighbor();
}
```

此前 Starling page-rerank 将 `search_flag` 无条件初始化为 `false`；当 `max_alpha_range_len=0` 时 interval 循环不执行，所有边被错误过滤，这是 No-alpha 零召回根因。当前代码已经修复为 `search_flag = (max_alpha_range_len == 0)`：No-alpha 展开全部邻居，Full/Merge 仍按 range 过滤。相同语义也同步到 page、beam 和直接 topology rerank 路径。

### 5.3 搜索实现速查

| 文件 | 主要用途 | 交接状态 |
|---|---|---|
| `src/search/rerank_search.cpp` | 普通双空间 rerank；包含 no-alpha 全展开分支 | OpenImages 原始布局已验证；不兼容当前 Laion Starling relayout |
| `src/search/starling_page_rerank_search.cpp` | PQ page-rerank、异步 I/O、page cache、alpha 过滤 | Full/Merge/No-alpha Starling 均已验证 |
| `src/search/page_search.cpp` | page-level beam search 和 alpha 过滤 | No-alpha 空 interval 语义已修复 |
| `src/search/beam_search.cpp` | 旧的标准 beam/single-index 路径 | 需要按索引布局单独验证 |
| `src/search/coro_search.cpp` | 协程搜索 | 非当前交接基准 |
| `src/search/pipe_search.cpp` | 流水线搜索 | 非当前交接基准 |

`mode=3` 在 Full、Merge、No-alpha 三种 Laion Starling relayout 上都会在 warm-up 崩溃，因此不能用它规避 No-alpha 的 mode 6 问题。

## 6. node3 已验证的 OpenImages 搜索流程

这一节是当前交接代码的首选冒烟测试。2026-09-02 已在 node3 使用 GCC 11.4、Linux AIO 和 AVX-512 完成验证。

### 6.1 服务器上的路径

```text
代码：       /mnt/nvme3/wz/hvs-disk
数据集：     /mnt/nvme3/wz/dataset/Openimage_disk
正确索引：   /mnt/nvme3/wz/Index/Openimage
```

**不要把 `/mnt/nvme3/wz/Index/Openimage-shard` 当作正确性基准。** 它是后续分片构建/合并版本，在当前测试中 Recall@10 明显异常；`/mnt/nvme3/wz/Index/Openimage` 才是已经验证的原始完整索引。两个目录的元数据看起来相同，仅检查节点数和维度不足以判断索引是否正确。

### 6.2 编译

```bash
ssh node3
cd /mnt/nvme3/wz/hvs-disk

export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build \
  -DBUILD_WITH_PQ=ON \
  -DCMAKE_CXX_COMPILER=g++
cmake --build build -j8
```

`BUILD_WITH_PQ=ON` 允许编译 PQ 生成逻辑，已验证不影响加载现有 PQ 文件进行搜索。2026-05 的旧 DGAI 搜索二进制使用 `BUILD_WITH_PQ=OFF`，但搜索所需的关键宏同样是 `REORDER_COMPUTE_PQ` 和 `USE_TOPO_DISK`。

若交接压缩包中出现 `._*` 或 `__MACOSX/`，删除后再配置 CMake；这只是打包清理事项，不属于实验流程。

### 6.3 已验证的搜索命令

```bash
cd /mnt/nvme3/wz/hvs-disk

./build/tests/search_disk_index \
  float \
  /mnt/nvme3/wz/Index/Openimage/ \
  32 4 \
  /mnt/nvme3/wz/dataset/Openimage_disk/query_img_emb.fvecs \
  /mnt/nvme3/wz/dataset/Openimage_disk/query_text_emb.fvecs \
  /mnt/nvme3/wz/dataset/Openimage_disk/base_gt_round_00/query_alpha.fvecs \
  /mnt/nvme3/wz/dataset/Openimage_disk/base_gt_round_00/top10_results.ivecs \
  10 l2 \
  3 0 1 \
  40 100 200 400
```

参数含义：

```text
32          查询线程数
4           SSD I/O/beam width
10          Recall@K 中的 K
l2          两个空间使用的基础距离
3           RERANK_SEARCH
0           mem_L=0，不加载可选内存入口索引
1           strategy=1，仅启用当前索引所需的 rerank 位
40...400    一次运行中依次测试的搜索深度 L
```

当前代码的搜索模式编号是：

| mode | 搜索实现 | 当前交接状态 |
|---:|---|---|
| 0 | `BEAM_SEARCH` | 旧 `single_index` 路径尚未作为交接基准验证 |
| 1 | `PAGE_SEARCH` | 非本节基准 |
| 2 | `PIPE_SEARCH` | 非本节基准 |
| 3 | `RERANK_SEARCH` | **OpenImages 已验证主路径** |
| 4 | `CORO_SEARCH` | 非本节基准 |
| 5 | `HYBRID_PAGE_SEARCH` | 非本节基准 |
| 6 | `PQ_PAGE_RERANK_SEARCH` | 非本节基准 |
| 7 | `DAP_SEARCH` | 非本节基准 |

不要脱离编译宏和索引文件直接复制其他脚本中的 strategy 数字。`23/31/47/63` 等值同时表示 double-PQ、拓扑重排、坐标重排或缓存等组合，需要对应的宏和额外索引文件。对于上述原始 OpenImages 索引，已验证组合是 `mode=3, mem_L=0, strategy=1`。

### 6.4 预期结果

硬件负载会影响 QPS 和延迟，但 Recall 应接近：

| L | Recall@10（当前交接代码） |
|---:|---:|
| 40 | 85.28% |
| 100 | 92.81% |
| 200 | 96.28% |
| 400 | 98.28% |

2026-05 的旧 DGAI 二进制在相同数据、索引和参数下分别得到 84.07%、91.87%、95.85% 和 98.01%，可作为兼容性对照。如果 L=100 仍只有个位数 Recall，优先检查索引路径是否误用了 `Openimage-shard`，不要先修改搜索代码。

### 6.5 当前可接受的警告

现有 OpenImages 索引没有：

```text
_disk.index.tags
_disk.index.alpha
```

tags 缺失时程序使用 ID 与 tag 相等的映射。alpha sidecar 缺失时，当前代码会提示 `alpha filtering disabled` 并展开所有邻居；这会增加比较次数，但上述 Recall 结果已经在该状态下验证。旧 DGAI 版本直接读取拓扑记录内的 alpha range，因此比较次数会更少。若任务涉及 alpha 剪枝性能或 I/O 对比，必须先统一 alpha 存储版本，不能把这条警告忽略为纯日志问题。

### 6.6 关于 `run_search.sh`

`run_search.sh` 固定使用 `mode=3, mem_L=0, strategy=1`，可以作为参数顺序示例，但其默认路径不是 node3 当前路径。脚本中的 `RESULT_PATH`/`-r` 实际上传给程序的是 ground-truth 文件，只用于计算 recall，并不是结果输出文件；运行不会把搜索结果写回该文件。首次验证建议直接复制本节的完整命令。

### 6.7 Full / Merge / No-alpha 与 Starling 的关系

代码历史中所说的三个方案是三种**磁盘拓扑格式**，不是 `search_mode=5/6/7` 三个搜索入口：

| 方案 | `max_alpha_range_len` | 节点记录 | 每个 8192B page 的节点数 |
|---|---:|---:|---:|
| Full alpha range | 7（Laion） | 724B | 11 |
| Merge alpha range | 2 | 324B | 25 |
| No alpha | 0 | 164B | 49 |

Laion 上的三个目录都是 Starling relayout 入口，其中的符号链接已把对应的元数据、拓扑、partition、坐标和 PQ 文件组合在一起：

```text
/mnt/nvme3/wz/Index/Laion-shard/alpha/
/mnt/nvme3/wz/Index/Laion-shard/merge-alpha/
/mnt/nvme3/wz/Index/Laion-shard/no-alpha/
```

历史命令为 `16 threads, width=4, mode=6, mem_L=0, strategy=17`。`strategy=17` 表示启用 rerank 和 topology buffer；Starling 构建必须保留 ID 映射，不能定义 `NO_MAPPING`。Full 和 Merge 在当前交接代码上已验证：

```bash
./build/tests/search_disk_index float \
  /mnt/nvme3/wz/Index/Laion-shard/alpha/ \
  16 4 \
  /mnt/nvme2/wz/disk/dataset/Laion/query_img_emb.fvecs \
  /mnt/nvme2/wz/disk/dataset/Laion/query_text_emb.fvecs \
  /mnt/nvme2/wz/disk/dataset/Laion/base_gt_round_00/query_alpha.fvecs \
  /mnt/nvme2/wz/disk/dataset/Laion/base_gt_round_00/top10_results.ivecs \
  10 l2 6 0 17 40 100 200
```

测 Merge 时只将索引目录替换为 `merge-alpha/`。2026-09-02 实测结果为：

| 方案 | L=40 | L=100 | L=200 |
|---|---:|---:|---:|
| Full | 85.61% | 93.98% | 96.93% |
| Merge | 85.61% | 93.98% | 96.92% |

**No-alpha 已恢复可用。** 修复前 `mode=6, strategy=17` 的 `Mean Cmps=0`、Recall@10=0 是空 range 被当作“不激活”的 reader bug，不是索引损坏或搜索参数错误。修复后在原 Laion 10,642,155-node No-alpha relayout 上，L=40 得到 Mean Cmps=1795.34、Recall@10=88.46%；L=100 得到 Mean Cmps=2777.66、Recall@10=95.40%。命令仍使用 `mode=6, mem_L=0, strategy=17`。不要再切换到不兼容该 relayout 的 mode 3。

加载时出现 `_disk.index.alpha` 缺失警告，只表示旧的独立 alpha sidecar 不存在；`mode=6、strategy=17` 的 Full/Merge 主路径仍从 topology record 内读取 alpha ranges。不要把该警告解释为 Full/Merge 已退化成 No-alpha。最直接的核对方式是比较 Mean Cmps：同一 L 下 No-alpha 应明显高于 Full/Merge。

## 7. 主要可执行程序与索引工具链

成功构建后，可执行程序位于 `build/tests/` 或 `build/tests/utils/`。常用入口包括：

- `build_hybrid_disk_index`：构建双空间图磁盘索引。
- `search_disk_index`：执行 SSD 搜索。
- `build_memory_index`：构建内存入口点索引。
- `merge_alpha`：将每条边的 alpha 区间压缩到至多两段。
- `remove_alpha`：生成不带 alpha 元数据的拓扑布局。
- `reorder_topo`、`reorder_by_map`、`check_reorder_topo`：拓扑重排和验证。
- `split_index`：遗留 4096B DiskANN 拆分工具；不能处理当前双空间 alpha 索引。
- `compute_groundtruth`：计算 ground truth。

具体参数以对应 `tests/*.cpp` 中的 `main` 函数和参数检查为准。现有 `run_search.sh` 可以作为调用示例，但其中的数据和索引路径是原服务器路径，运行前必须全部替换。

### 7.1 `remove_alpha`

`tests/remove_alpha.cpp` 将含 alpha range 的 topology 转换为只保留 neighbor ID 的固定长记录：

```text
1. 读取源 metadata
2. 写入 max_alpha_range_len=0 的新 metadata
3. sector-by-sector 读取源 topology，跳过每页 padding
4. 对每个邻居复制 ID，跳过固定数量的 alpha bytes
5. 按 164B/node、49 nodes/page 重新对齐写入
```

```bash
build/tests/remove_alpha \
  <input_meta> <input_topology> \
  <output_meta> <output_topology> \
  [output_alpha_sidecar]
```

不能把源 topology 当成无 padding 的连续 node array 读取。源和目标记录密度不同，必须完整重写 topology segment。

### 7.2 `merge_alpha`

`tests/merge_alpha.cpp` 将每个邻居的 alpha range 压缩到至多两对：

```bash
build/tests/merge_alpha \
  <input_meta> <input_topology> \
  <output_meta> <output_topology> \
  <merge_method>

merge_method: 2 = greedy gap merge
              3 = max-coverage
```

转换器始终按源 metadata 中的 `old_max_alpha_range_len` 读取每个邻居，因为磁盘上没有单独的 `range_size` 字段。输出 metadata 使用 `max_alpha_range_len=2`，对 degree 40 对应 324B/node 和 25 nodes/page；未使用的 slot 补零。

Method 2 首先按 interval start 排序，每次合并 gap 最小的相邻区间，直到剩下至多两段。它优先填充最小的空洞，会扩大边的激活范围，但不会丢掉原有覆盖点。

Method 3 将 `[0,100]` 离散为 101 点覆盖位图。目标为一段时返回 `[first,last]`；目标为两段时保留最大的连续 uncovered gap，并在其两侧形成 interval。如果中间没有空洞，则退化为一个 `[first,last]`。

2026-04-27 在 test-shard、method 2 上记录的转换结果为：456,699 个节点，6 slots/neighbor 的固定输入约 107.9M pairs，输出约 36.0M pairs，约 18.0M 个 neighbor 需要合并；topology 从约 298MB 降至约 143MB，nodes/page 从 12 提高到 25。这是特定数据集和方法的记录，不要外推成所有数据集的固定比例。

### 7.3 Starling relayout

Starling 是**外部实现**，不包含在本仓库和交接基线中。交接统一使用 [wang29a/starling](https://github.com/wang29a/starling)，不要改用 node3 上其他同名目录，也不要继续引用历史 `/home/gongwei/starling` 路径。

2026-09-02 在 node3 验证的版本：

```text
repository: https://github.com/wang29a/starling.git
commit:     7437c4848a83e7bf62558c5eab1eb999feed8537
path:       /mnt/nvme3/wz/starling-wang29a
build:      /mnt/nvme3/wz/starling-wang29a/release-gcc11-abi0
```

这个 fork 的 `graph_partition/include/partitioner.h` 提供 `load_custom_topology_graph()`，会从 hvs-disk metadata 读取 `max_alpha_range_len`，并按 8192B page 解析 Full/Merge/No-alpha topology：

```text
single_neighbor_size = 4 + 2 * max_alpha_range_len
fixed_topo_size      = 4 + max_nbr_len * single_neighbor_size
nodes_per_page       = 8192 / fixed_topo_size
```

#### 7.3.1 node3 构建注意事项

node3 默认 GCC 4.8 不能编译 oneTBB，必须使用 GCC 11。系统 Boost 1.53 使用旧 libstdc++ ABI，因此还要设置 `_GLIBCXX_USE_CXX11_ABI=0`：

```bash
git clone --recurse-submodules \
  https://github.com/wang29a/starling.git \
  /mnt/nvme3/wz/starling-wang29a

cd /mnt/nvme3/wz/starling-wang29a
git checkout 7437c4848a83e7bf62558c5eab1eb999feed8537
git submodule update --init --recursive
cmake -S . -B release-gcc11-abi0 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=gcc-11 \
  -DCMAKE_CXX_COMPILER=g++-11 \
  -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0 \
  -DTBB_ENABLE_IPO=OFF
cmake --build release-gcc11-abi0 -j8 --target partitioner
```

node3 的汇编器不支持 oneTBB 在 GCC 11 下自动启用的 WAITPKG/`tpause`。当前服务器克隆已做两处兼容修改：从 `graph_partition/oneTBB/cmake/compilers/GNU.cmake` 移除 `-mwaitpkg`，并在 `graph_partition/oneTBB/include/oneapi/tbb/detail/_config.h` 将 `__TBB_WAITPKG_INTRINSICS_PRESENT` 设为 0。它只影响等待指令优化，不改变 partition 算法或索引格式。重新克隆后若再次出现 `tpause` assembler error，需要重做这两处修改。

完整 `index_relayout` CMake target 还会编译与 custom relayout 无关的旧 DiskANN/MKL 代码，在 node3 的 Boost/MKL 环境下不能直接通过。当前验证使用同一仓库中 `tests/utils/index_relayout.cpp` 的 `--custom` 分支独立编译；不要误以为默认 `scripts/run_benchmark.sh gp` 支持 hvs-disk，它调用的是标准 DiskANN 路径。

```bash
cd /mnt/nvme3/wz/starling-wang29a
mkdir -p release-gcc11-abi0/tests/utils
g++-11 -std=c++14 -O2 -mavx2 -include immintrin.h \
  -Iinclude tests/utils/index_relayout.cpp \
  -Wl,--unresolved-symbols=ignore-all \
  -o release-gcc11-abi0/tests/utils/index_relayout_custom

./release-gcc11-abi0/tests/utils/index_relayout_custom --help
```

这里忽略的 unresolved symbols 只属于同一源码中的标准 DiskANN relayout 分支；交接流程只调用 `--custom`。更稳妥的长期修复是让外部 Starling 将 custom relayout 拆成独立 target，并修正 8192B metadata 输出，再去掉这个临时链接选项。

#### 7.3.2 Full/Merge/No-alpha 的实际命令

以下命令中的 `<meta>` 和 `<topology>` 必须来自同一种格式。Full、Merge、No-alpha 分别执行一遍，不能共享 partition：

```bash
cd /mnt/nvme3/wz/starling-wang29a
set -o pipefail
export LD_LIBRARY_PATH=/opt/gcc-11.4/lib64:$PWD/release-gcc11-abi0/gnu_11.4_cxx17_64_release:$LD_LIBRARY_PATH

./release-gcc11-abi0/graph_partition/partitioner \
  --data_type float \
  --gp_file <partition.bin> \
  --custom_graph 1 \
  --graph_file <topology> \
  --meta_file <meta> \
  --ep_size <entrypoint_count> \
  -T 16 -L 8

./release-gcc11-abi0/tests/utils/index_relayout_custom \
  --custom \
  <topology> <meta> <partition.bin> <relayout_topology>

cd /mnt/nvme3/wz/hvs-disk
./build/tests/pad_partition <partition.bin>
# 输出：<partition.bin>.aligned
```

node3 必须把 `/opt/gcc-11.4/lib64` 放在系统运行库之前，否则 partitioner 会误载 GCC 4.8 的 `libstdc++`，并报 `GLIBCXX_3.4.29` 或 `CXXABI_1.3.9 not found`。命令经过 `tee` 保存日志时应先启用 `set -o pipefail`，否则 partitioner 失败后整条流水线仍可能返回 0。

`-L` 是 LDG partition 轮数，不是搜索的 `L_search`。本次快速正确性验证使用 `-L 1`；正式构建建议使用仓库惯例 `-L 8`，并记录线程数和 commit。

当前 fork 的 custom relayout 会生成 `<relayout_topology>_meta.bin`，但它只复制 4096B metadata，而 hvs-disk 要求完整 8192B metadata 页。**不要安装这个 4KB 文件。** 当前记录宽度没有变化，搜索目录应继续使用该方案转换阶段生成的原 8192B `<meta>`；relayout 只替换 topology，并配套使用刚生成的 `.aligned` partition。

外部 Starling 阶段至少要明确交付以下契约：

```text
输入：所选格式对应的 metadata、原 topology、8192B page size
输出：partition/aligned partition、old->new 或 new->old 映射、relayout topology
保持：节点集合、每个节点的 neighbor ID 集合、alpha slots、入口点语义
允许：节点物理顺序和 page 归属变化
```

外部 Starling 不负责生成 PQ、精确双向量或 alpha 压缩。推荐先在 hvs-disk 内完成 Full/Merge/No-alpha 格式选择，再把对应 topology 交给 Starling。外部工具输出后，应将同一方案的 metadata、relayout topology、partition 文件和共享 coord/PQ 文件放入一个独立目录，通过统一名称或符号链接供 loader 使用；不要混用不同方案的文件。

每种方案的 `search/` 目录必须向 loader 提供下面八个名字；其中 metadata、relayout topology 和 partition 属于该方案，coord 和两套 PQ 文件可以共享同一次 Full 构建产物：

```bash
root=/path/to/experiment
kind=full  # 分别替换为 full、merge、no-alpha
mkdir -p "$root/$kind/search"
ln -s ../_disk.index "$root/$kind/search/_disk.index"
ln -s ../disk_index_graph.relayout "$root/$kind/search/disk_index_graph"
ln -s ../partition.bin.aligned "$root/$kind/search/_partition.bin.aligned"
ln -s ../../full/disk_index_data "$root/$kind/search/disk_index_data"
ln -s ../../full/emb_pq_compressed.bin "$root/$kind/search/emb_pq_compressed.bin"
ln -s ../../full/emb_pq_pivots.bin "$root/$kind/search/emb_pq_pivots.bin"
ln -s ../../full/loc_pq_compressed.bin "$root/$kind/search/loc_pq_compressed.bin"
ln -s ../../full/loc_pq_pivots.bin "$root/$kind/search/loc_pq_pivots.bin"
find -L "$root/$kind/search" -maxdepth 1 -type f -printf '%f %s\n' | sort
```

如果接手环境没有上述指定 Starling：可以先完成 Full/Merge/No-alpha 转换和格式校验，但不能运行 `mode=6, strategy=17` 的 Starling page-rerank，也不能据此报告最终性能。仓库内的遗留 reorder 工具不能代替外部 Starling。

#### 7.3.3 node3 端到端验证结果

对 OpenImages Full-alpha 原始 topology（456,699 nodes、40 max degree、6 range pairs、644B/node、12 nodes/page）执行指定 Starling 后：

- partition 包含 38,059 页，456,699 个 ID 恰好各出现一次，无重复、遗漏或越界；
- 每个 partition 至多 12 个节点；
- relayout topology 为 311,779,328B，与输入大小一致；
- 按 partition 的页内位置逐条比较全部 456,699 条 644B record，字节 mismatch 为 0；
- `pad_partition` 输出约 1.9MB 的定长 `_partition.bin.aligned`；
- 使用原 8192B metadata、原 coord/PQ、relayout topology 和 aligned partition，`mode=6, strategy=17` 搜索成功。

2026-09-02、1 轮 LDG 的快速验证结果：

| L_search | Recall@10 |
|---:|---:|
| 40 | 83.52% |
| 100 | 91.85% |

这些数值证明该工具链可加载并返回合理结果，但不是正式的 8 轮 partition 性能结论。正式对比仍需对三个方案分别使用相同 LDG 轮数，并在匹配 Recall 的 operating point 比较 QPS/I/O。

#### 7.3.4 从构建到 Full/Merge 搜索的已跑通记录

完整产物、输入、query、ground truth 和日志保存在 node3：

```text
/mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902/
```

目录含义：

```text
input/       构建使用的 10K 双空间 base
query/       100 条 query、alpha 和针对该 10K 子集计算的 top-100 ground truth
full/        Full-alpha 原始/relayout topology、partition、aligned partition、search/
merge/       Merge-alpha 原始/relayout topology、partition、aligned partition、search/
logs/        每一步完整日志和 JSON 校验摘要
```

执行链为：

```text
build_hybrid_disk_index (... l2 0)
  -> Full: A=5, 760B/node, 10 nodes/page
  -> merge_alpha method=2
  -> Merge: A=2, 436B/node, 18 nodes/page
  -> 两种格式各自运行 Starling -L 8
  -> 两种格式各自 custom relayout
  -> 两种格式各自 pad_partition
  -> mode=6, mem_L=0, strategy=17 搜索
```

reorder 校验结果：两种方案的 partition 都覆盖恰好 10,000 个不同 ID，无遗漏、重复、越界或超页容量；按 partition 页内位置比较 relayout 前后的每条完整 topology record，Full 和 Merge 的 mismatch 都为 0。机器可读结果在 `logs/06_validation_summary.json`。

搜索配置：100 queries、8 threads、beam/I/O width 4、Recall@10、`mode=6`、`mem_L=0`、`strategy=17`，两种方案使用同一 query/alpha/ground truth，L_search sweep 为 20/40/80/100/200。

| L_search | Full Recall | Full Mean I/O | Merge Recall | Merge Mean I/O |
|---:|---:|---:|---:|---:|
| 20 | 75.10% | 25.77 | 75.10% | 23.48 |
| 40 | 86.10% | 36.75 | 86.10% | 33.15 |
| 80 | 92.50% | 61.44 | 92.50% | 54.51 |
| 100 | 93.90% | 73.90 | 93.90% | 64.87 |
| 200 | 97.20% | 132.77 | 97.20% | 112.81 |

本次结果表明链路和格式正确，并展示 Merge 在该小样本上的页面 I/O 降低；100-query/10K 数据仅用于端到端验证，不作为正式论文性能结论。对应日志为 `logs/01_build_full.log`、`02_merge_alpha.log`、`03_starling_{full,merge}.log`、`04_relayout_{full,merge}.log` 和 `05_search_{full,merge}.log`。

#### 7.3.5 No-alpha 构建、重排与搜索

No-alpha 必须从同一份 Full-alpha 原始 topology 转换，再独立运行 Starling：

```bash
root=/mnt/nvme3/wz/hvs-disk-workflow/Openimage/e2e-build10k-20260902

./build/tests/remove_alpha \
  $root/full/_disk.index $root/full/disk_index_graph \
  $root/no-alpha/_disk.index $root/no-alpha/disk_index_graph

# 然后对 no-alpha/_disk.index + no-alpha/disk_index_graph
# 执行 7.3.2 的 partitioner、index_relayout_custom 和 pad_partition。
```

该 10K 索引为 `max_nbr_len=54`、`max_alpha_range_len=0`、220B/node、37 nodes/page。8 轮 Starling 生成 271 个 partition pages；10,000 个 ID 无重复、遗漏、越界或超页容量，relayout 的完整 record mismatch 为 0。

No-alpha 搜索命令与 Full/Merge 相同，只替换索引目录：

```bash
./build/tests/search_disk_index \
  float $root/no-alpha/search/ \
  8 4 \
  $root/query/query_img_100.bin \
  $root/query/query_text_100.bin \
  $root/query/query_alpha_100.bin \
  $root/query/groundtruth_top100.bin \
  10 l2 6 0 17 \
  20 40 80 100 200
```

| L_search | No-alpha Recall | Mean Cmps | Mean I/O |
|---:|---:|---:|---:|
| 20 | 76.20% | 820.51 | 18.43 |
| 40 | 87.10% | 1187.29 | 26.85 |
| 80 | 93.30% | 1860.96 | 43.50 |
| 100 | 94.60% | 2162.18 | 52.34 |
| 200 | 97.50% | 3411.83 | 88.18 |

No-alpha 比 Full/Merge 比较更多邻居是预期行为，因为它不进行 alpha range 剪枝；页面更密，因此本次小样本的 Mean I/O 更低。日志位于 `logs/07_remove_alpha.log`、`08_starling_no_alpha.log`、`09_relayout_no_alpha.log`、`10_search_no_alpha.log` 和 `12_no_alpha_validation.json`。Laion 大规模复测日志为 `logs/11_search_laion_no_alpha_fixed.log`。

对 Laion 现有 Full 和 No-alpha relayout 已完成 10,642,155 节点全量校验：缺失节点、重复节点、越界 neighbor ID、超 degree 记录和逐节点 neighbor-list mismatch 均为 0；修复 reader 后的搜索也已通过。

`tests/check_reorder_topo.cpp` 仍包含 4096B page、132B record 和旧绝对路径的硬编码，不能直接用来验证当前 8192B 双空间索引。

#### 7.3.6 2026-09-10 从干净代码目录重新验证

交接提交 `ea4ac09` 已重新同步到 node3 的独立目录 `/mnt/nvme3/wz/hvs-disk-validation-20260910`。本轮删除了对旧代码、旧编译目录和旧索引输出的依赖，从 CMake 配置、Full 构建、Merge/No-alpha 转换开始，三种格式分别执行指定 Starling 的 8 轮 partition、custom relayout、partition 对齐和统一搜索。只复用了与这批 10K base 严格匹配的 query、alpha 和 ground truth 输入。

严格校验中，三种方案均覆盖恰好 10,000 个唯一 ID，缺失、重复、越界、超页容量、ID-to-partition 映射错误、非法 degree、非法 neighbor、非法 interval 和逐条 topology record mismatch 全部为 0。完整结果和逐步日志保存在 `validation-run/logs/`，机器可读摘要为 `08_validation_summary.json`。

统一搜索参数为 100 queries、8 threads、I/O width 4、Recall@10、`mode=6、mem_L=0、strategy=17`：

| L_search | Full Recall / Mean I/O | Merge Recall / Mean I/O | No-alpha Recall / Mean I/O |
|---:|---:|---:|---:|
| 20 | 74.30% / 26.63 | 74.30% / 23.96 | 76.40% / 18.76 |
| 40 | 84.10% / 38.34 | 84.10% / 33.61 | 85.00% / 27.30 |
| 80 | 92.00% / 62.62 | 92.00% / 54.08 | 92.40% / 44.02 |
| 100 | 94.10% / 74.95 | 94.10% / 64.04 | 94.50% / 52.26 |
| 200 | 97.30% / 133.11 | 97.30% / 112.01 | 97.40% / 87.58 |

本轮 Full 与 Merge 的 Recall 完全一致，Merge 的 Mean I/O 更低；No-alpha 的 Mean Cmps 更高但页面更密，因而 Mean I/O 更低。这是 10K 正确性与流程验收，不是正式性能结论。重新运行时允许 QPS、延迟和 Recall 有小幅变化，但结构校验中的所有错误计数和 record mismatch 必须严格为 0。

### 7.4 重排后的最低校验清单

Starling 完成不等于索引正确。每种方案至少检查：

1. metadata 中 `node_num`、两侧维度、`max_nbr_len`、`max_alpha_range_len` 和入口点数量合理；
2. topology 文件大小等于按 nodes/page 和 page padding 推导的大小；
3. 恰好解析 `node_num` 条记录，无提前 EOF 或尾部非预期数据；
4. 每条 `degree <= max_nbr_len`，每个 neighbor ID 在 `[0,node_num)`；
5. relayout 前后无节点丢失或重复，映射为完整双射；
6. 将 relayout neighbor ID 逆映射后，逐节点 neighbor-list 与输入 topology 一致；
7. Full/Merge 的有效 interval 端点在 `[0,100]`，No-alpha 的 stride 确为 4B；
8. 使用与数据和 alpha 匹配的 ground truth 做 Recall 冒烟测试，而不是只检查 QPS 或进程退出码。

## 8. 磁盘格式：修改前必读

每个 page/sector 固定为 `8192` 字节。索引由元数据页、双空间精确向量段和固定长度拓扑段组成：

```text
[ metadata ][ coordinate segment ][ topology segment ]
```

拓扑中每个节点的记录为：

```text
[neighbor_count: 4B]
[neighbor 0: id 4B + 2 * max_alpha_range_len bytes]
...
```

必须保持以下约束：

1. 每个邻居都读写恰好 `max_alpha_range_len` 对区间，空槽补零；记录不是变长的。
2. 修改拓扑记录大小时，必须重新布局整个拓扑段。
3. 不要随意修改 `SECTOR_LEN` 或页对齐方式。
4. 格式变化要同步修改元数据、尺寸计算、偏移函数、转换工具、reader 和重排工具。
5. `remove_alpha` 和历史 No-alpha 分支常用 `_disk.index.no_alpha`/`disk_index_graph.no_alpha` 后缀；当前统一 loader 读取 `<index_prefix>/_disk.index`，实验目录通过符号链接把具体格式暴露为统一文件名。运行前必须以当前分支的 `src/ssd_index.cpp` 为准。
6. Starling 重排通过元数据中的 `max_alpha_range_len` 自动判断 alpha/no-alpha 布局，这个约定不能破坏。

## 9. 数据集与文件格式（运行前必读）

**本项目不能直接读取任意 NumPy、`.fvecs` 或文本文件。构建和搜索入口实际读取的是带 8 字节文件头的连续二进制矩阵。文件扩展名不是判断格式的依据，内部字节布局才是。**

### 9.1 通用 bin 矩阵格式

base、query 和 alpha 文件都使用以下布局：

```text
offset  size                         content
0       4B                           npts，int32
4       4B                           dim，int32
8       npts * dim * sizeof(T)       按行连续存储的数据
```

即：

```text
[npts:int32][dim:int32][row 0][row 1]...[row npts-1]
```

当前 Linux/x86 实验环境使用小端序。数据区不能有行间 padding，也不能在每行前重复写维度。文件大小应严格满足：

```text
8 + npts * dim * sizeof(T)
```

命令行中的 `data_type` 决定 `T`：

- `float`：每个元素为 4 字节 `float32`，常用扩展名为 `.fbin` 或 `.bin`；
- `int8`：每个元素为 1 字节有符号整数；
- `uint8`：每个元素为 1 字节无符号整数。

扩展名仅是命名习惯，能否传给 `load_bin<T>` 只取决于内部布局。标准 `.fvecs` 不能直接读取；但 node3 的 OpenImages 文件虽然沿用了 `.fvecs/.ivecs` 后缀，内部已经是本节所述的全局 8 字节头 bin 格式，因此可以直接使用，不要再次转换。

### 9.2 双空间 base 数据

hvs-disk 的索引构建需要两份按节点一一对应的数据文件：

```text
base_space_1.bin: N × D1
base_space_2.bin: N × D2
```

必须满足：

- 两个文件的 `npts` 完全相同；第 `i` 行必须描述同一个节点；
- `D1` 和 `D2` 可以不同；
- 当前 `build_hybrid_disk_index` 对两份数据使用同一个 `data_type`；
- 任何独立 shuffle、过滤或重排都必须同时作用于两份文件，并保持相同 ID 顺序。

构建入口的真实参数顺序为：

```bash
build/tests/build_hybrid_disk_index \
  <float|int8|uint8> \
  <base_space_1.bin> <base_space_2.bin> <index_prefix> \
  <R> <L> <B> <M> <threads> <cosine|l2> <single_file_index:0|1>
```

### 9.3 双空间 query 与 alpha

一次搜索需要三份逐查询对齐的文件：

```text
query_space_1.bin: Q × D1
query_space_2.bin: Q × D2
query_alpha.bin:   Q × 1，数据类型必须为 float32
```

必须满足：

- 三个文件的 `npts` 都等于 `Q`；
- query space 1 的维度与 base space 1 相同；
- query space 2 的维度与 base space 2 相同；
- alpha 文件建议并要求按 `Q × 1` 生成，每行一个 `[0,1]` 范围内的 float32；
- 第 `i` 个 alpha 只对应两份 query 文件中的第 `i` 行，不能打乱其中任何一个文件。

搜索使用：

```text
alpha * dist(query_space_1, base_space_1)
+ (1 - alpha) * dist(query_space_2, base_space_2)
```

`tests/search_disk_index.cpp` 当前打印的 Usage 文本仍是旧的单查询版本，**不要照该提示传参**。代码实际读取的参数顺序是：

```bash
build/tests/search_disk_index \
  <float|int8|uint8> <index_prefix> \
  <num_threads> <beam_width> \
  <query_space_1.bin> <query_space_2.bin> <query_alpha.bin> \
  <truthset.bin> <K> <cosine|l2> \
  <search_mode> <mem_L> <strategy> <L1> [L2 ...]
```

如果不计算 recall，可以给 `truthset.bin` 参数传一个不存在的路径；程序通过文件是否存在决定是否加载 ground truth。

### 9.4 Ground truth 格式

Ground truth 使用以下布局：

```text
[num_queries:int32][K:int32]
[num_queries * K 个 neighbor ID:uint32]
[可选：num_queries * K 个 distance:float32]
[可选：num_queries * K 个 tag:uint32]
```

允许三种精确文件大小：

1. header + IDs；
2. header + IDs + distances；
3. header + IDs + distances + tags。

ID、distance 和 tag 都是各自完整的矩阵块，不是按单个邻居交错存储。用于搜索评估时，ground truth 的 `num_queries` 必须等于 query 文件的 `Q`，且文件中的 `K` 不应小于运行参数 `K`。

### 9.5 fvecs、bvecs 和 ivecs 不能直接混用

标准 `fvecs/bvecs/ivecs` 通常在每一行前保存一次维度，而本项目的 bin 格式只在整个文件开头保存一次 `npts` 和 `dim`。两者不是同一格式。注意，文件后缀可能沿用旧名称；node3 的 OpenImages `.fvecs/.ivecs` 已经是项目 bin，不需要转换。对于真正的标准 vecs 文件，仓库提供转换工具：

```bash
build/tests/utils/fvecs_to_bin input.fvecs output.fbin
build/tests/utils/bvecs_to_bin input.bvecs output.bin
build/tests/utils/ivecs_to_bin input.ivecs output.bin
```

转换后仍应检查输出文件头、数据类型、点数、维度和文件大小。不要仅通过后缀判断转换是否正确。

### 9.6 数据交接检查清单

交接包不包含数据集和已生成索引。开始实验前，需要从师兄或实验服务器确认：

- 数据集和查询文件的位置及格式；
- ground truth 的位置和 top-k；
- 原始索引、PQ 文件和重排映射的位置；
- 使用的服务器、SSD、线程数和编译宏；
- 需要复现的具体脚本和参数组合。

拿到数据后，至少检查：两份 base 的 `N` 是否一致、两份 query 和 alpha 的 `Q` 是否一致、两组 base/query 维度是否分别匹配、alpha 是否为 `float32 Q×1`、ground truth 是否对应相同的 query 顺序和 alpha 配置。对于双空间融合查询，不同 alpha 通常对应不同 ground truth，不能拿固定 alpha 的 truthset 评估另一组 alpha。

`scripts/config.sh`、`run_search.sh` 及部分 `shell/`、`_scripts/` 脚本含绝对路径或历史项目名。建议先复制一份本地配置，再替换 `PROJECT_PATH`、`DATA_PATH`、`DISKANN_INDEX_PATH`、`OUTPUT_PATH` 和 `LOG_PATH`，不要直接假设旧路径仍有效。部分脚本还会使用 `sed` 修改源码并重新编译，运行前务必阅读完整脚本并记录 Git diff。

## 10. 工程速查

### 10.1 关键文件与符号

源码会演进，优先按符号搜索，不要长期依赖 README 中的历史行号。

| 关注点 | 文件或符号 |
|---|---|
| metadata 解析和 layout 计算 | `src/ssd_index.cpp`, `SSDIndex::load` |
| page/记录偏移 | `include/ssd_index.h`, `u_loc_offset_*`, `loc_sector_no_*`, `offset_to_loc_*` |
| 构建期邻居 | `include/neighbor.h`, `DEGNeighbor` |
| 序列化邻居 | `include/neighbor.h`, `DEGSimpleNeighbor` |
| query-time node view | `include/query_buf.h`, `DiskNode` |
| alpha range 构建 | `src/hybrid_index.cpp`, `occlude_list`, `intersection`, `get_use_range`, `mergeIntervals`, `prune_neighbors` |
| 普通 rerank/no-alpha 分支 | `src/search/rerank_search.cpp`, `do_rerank_search` |
| page alpha 过滤 | `src/search/page_search.cpp` |
| Starling page-rerank | `src/search/starling_page_rerank_search.cpp`, `pq_page_rerank_search` |
| remove-alpha 转换 | `tests/remove_alpha.cpp` |
| merge-alpha 压缩 | `tests/merge_alpha.cpp`, `merge_ranges_greedy`, `merge_ranges_max_coverage` |
| 异步磁盘 I/O | `src/utils/linux_aligned_file_reader.cpp` |

### 10.2 核心布局公式

```text
coord_len = (data_dim + datal_dim) * sizeof(T)
ncoord_per_sector = 8192 / coord_len

single_neighbor_size = 4 + 2 * max_alpha_range_len
fixed_topo_size = 4 + max_nbr_len * single_neighbor_size
ntopo_per_sector = 8192 / fixed_topo_size
```

degree 40 时：

```text
Full Laion (7 slots): 4 + 40*(4+14) = 724B -> 11 nodes/page
Full test-shard (6):  4 + 40*(4+12) = 644B -> 12 nodes/page
Merge (2):            4 + 40*(4+4)  = 324B -> 25 nodes/page
No-alpha (0):         4 + 40*4      = 164B -> 49 nodes/page
```

### 10.3 技术来源与边界

本实现继承了 DiskANN/PipeANN/OdinANN 系列的 SSD graph search 和工程结构，并结合双空间 alpha-range pruning 与 Starling 式 page-aware relayout。alpha range 概念、两种 merge heuristic 和 Starling partitioner 都不应被描述为 hvs-disk 原创的通用 ANN 算法。本项目的实际范围是 SSD-resident dual-space graph ANN 及其元数据布局/执行取舍。

## 11. 推荐开发流程

拿到压缩包后先执行：

```bash
git status
git log --oneline --decorate -n 3
git switch -c feature/<任务名>
```

每次修改建议遵循：

1. 明确修改涉及构建、搜索、更新还是磁盘格式。
2. 用 `rg` 搜索相关结构和所有 reader/writer。
3. 小步修改并提交，提交信息写清配置和影响范围。
4. C++ 修改后先编译，再运行与改动对应的可执行程序。
5. 实验结果记录数据集、索引版本、alpha、L、rerank depth、beam width、线程数、队列深度、缓存、PQ 配置和硬件。
6. 比较方案时使用 matched recall，不要只比较相同 L 下的 QPS。

如果修改磁盘格式，至少覆盖以下边界：零邻居、最大度数、空 alpha 槽、alpha 端点 0/100 和最后一个不满页的 page。

## 12. 当前已知风险

- 代码源自并扩展了 PipeANN/OdinANN，部分旧命名、注释和脚本仍未统一为 hvs-disk。
- 多个搜索实现可能存在重复逻辑，修改一个实现不会自动影响其他实现。
- 部分实验配置通过编译宏和直接改源码切换，复现实验前要保存完整配置。
- `scripts/`、`shell/` 和 `_scripts/` 的时效性不同；优先使用 `scripts/`，旧脚本需要逐项验证。
- 当前构建系统偏 Linux，依赖和服务器环境需要另行交接。
- `/mnt/nvme3/wz/Index/Openimage-shard` 不是当前正确性基准；使用索引前必须记录完整路径和生成版本。
- `tests/check_reorder_topo.cpp` 仍是 4096B/132B 的旧工具，不能直接校验当前索引。
- No-alpha 不做 alpha range 剪枝，Mean Cmps 高于 Full/Merge 是预期行为；仍需在匹配 recall 下比较 I/O、QPS 和延迟。

## 13. 遇到问题时需要提供的信息

提问或记录 issue 时，至少附上：

- 当前 commit 和 `git status --short`；
- 完整 CMake 命令和 `ADDITIONAL_DEFINITIONS`；
- 编译器、Linux 内核、CPU 指令集和 I/O 后端；
- 数据集、索引前缀及关键元数据；
- 完整运行命令；
- 错误日志或最小复现步骤。

这样可以快速区分代码错误、索引格式不匹配、编译宏不一致和服务器环境问题。
