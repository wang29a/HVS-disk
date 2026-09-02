# hvs-disk 代码交接说明

本仓库是 hvs-disk 的代码交接版本。hvs-disk 面向 **SSD 常驻的双空间图近似最近邻搜索**：查询通过 `alpha` 融合两个距离空间，并可根据边上的 alpha 有效区间决定是否扩展该边。

这份交接包只保留实现、构建文件和运行脚本，不包含论文草稿、写作材料、实验图片、日志、数据集、索引文件和编译产物。Git 历史已压缩为一个交接基线提交；后续开发可以直接在此基础上创建分支和提交。

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

## 5. node3 已验证的 OpenImages 搜索流程

这一节是当前交接代码的首选冒烟测试。2026-09-02 已在 node3 使用 GCC 11.4、Linux AIO 和 AVX-512 完成验证。

### 5.1 服务器上的路径

```text
代码：       /mnt/nvme3/wz/hvs-disk
数据集：     /mnt/nvme3/wz/dataset/Openimage_disk
正确索引：   /mnt/nvme3/wz/Index/Openimage
```

**不要把 `/mnt/nvme3/wz/Index/Openimage-shard` 当作正确性基准。** 它是后续分片构建/合并版本，在当前测试中 Recall@10 明显异常；`/mnt/nvme3/wz/Index/Openimage` 才是已经验证的原始完整索引。两个目录的元数据看起来相同，仅检查节点数和维度不足以判断索引是否正确。

### 5.2 编译

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

如果代码从 macOS 复制到 Linux，先检查是否混入 AppleDouble 文件：

```bash
find . -name '._*' -print
```

这些文件会被 CMake 的 `*.cpp` GLOB 误认为源码。最终压缩或传输时应排除 `._*` 和 `__MACOSX/`；使用 tar 时可以在 macOS 设置 `COPYFILE_DISABLE=1`。

### 5.3 已验证的搜索命令

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

### 5.4 预期结果

硬件负载会影响 QPS 和延迟，但 Recall 应接近：

| L | Recall@10（当前交接代码） |
|---:|---:|
| 40 | 85.28% |
| 100 | 92.81% |
| 200 | 96.28% |
| 400 | 98.28% |

2026-05 的旧 DGAI 二进制在相同数据、索引和参数下分别得到 84.07%、91.87%、95.85% 和 98.01%，可作为兼容性对照。如果 L=100 仍只有个位数 Recall，优先检查索引路径是否误用了 `Openimage-shard`，不要先修改搜索代码。

### 5.5 当前可接受的警告

现有 OpenImages 索引没有：

```text
_disk.index.tags
_disk.index.alpha
```

tags 缺失时程序使用 ID 与 tag 相等的映射。alpha sidecar 缺失时，当前代码会提示 `alpha filtering disabled` 并展开所有邻居；这会增加比较次数，但上述 Recall 结果已经在该状态下验证。旧 DGAI 版本直接读取拓扑记录内的 alpha range，因此比较次数会更少。若任务涉及 alpha 剪枝性能或 I/O 对比，必须先统一 alpha 存储版本，不能把这条警告忽略为纯日志问题。

### 5.6 关于 `run_search.sh`

`run_search.sh` 固定使用 `mode=3, mem_L=0, strategy=1`，可以作为参数顺序示例，但其默认路径不是 node3 当前路径。脚本中的 `RESULT_PATH`/`-r` 实际上传给程序的是 ground-truth 文件，只用于计算 recall，并不是结果输出文件；运行不会把搜索结果写回该文件。首次验证建议直接复制本节的完整命令。

### 5.7 Full / Merge / No-alpha 与 Starling 的关系

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

**No-alpha 目前不能作为可用基线。** 对现有 `no-alpha/` relayout，当前交接代码以 `mode=6, strategy=17` 运行时 `Mean Cmps=0` 且 Recall@10=0；切换为 `mode=3` 会在预热阶段崩溃。单独编译历史提交 `912e599` (`feat/no-alpha-rerank-starling`) 后结果相同。这说明当前问题是 No-alpha 的拓扑解析/映射路径没有与 page-rerank 完整打通，不是调大 `L` 可以解决。修复并在同一 ground truth 上获得非零 Recall 之前，不要报告 No-alpha 的 QPS：零比较次数下的高 QPS 是无效结果。

## 6. 主要可执行程序

成功构建后，可执行程序位于 `build/tests/` 或 `build/tests/utils/`。常用入口包括：

- `build_hybrid_disk_index`：构建双空间图磁盘索引。
- `search_disk_index`：执行 SSD 搜索。
- `build_memory_index`：构建内存入口点索引。
- `merge_alpha`：将每条边的 alpha 区间压缩到至多两段。
- `remove_alpha`：生成不带 alpha 元数据的拓扑布局。
- `reorder_topo`、`reorder_by_map`、`check_reorder_topo`：拓扑重排和验证。
- `split_index`：拆分索引的逻辑组成部分。
- `compute_groundtruth`：计算 ground truth。

具体参数以对应 `tests/*.cpp` 中的 `main` 函数和参数检查为准。现有 `run_search.sh` 可以作为调用示例，但其中的数据和索引路径是原服务器路径，运行前必须全部替换。

## 7. 磁盘格式：修改前必读

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
5. no-alpha 文件后缀为 `_disk.index.no_alpha`。
6. Starling 重排通过元数据中的 `max_alpha_range_len` 自动判断 alpha/no-alpha 布局，这个约定不能破坏。

## 8. 数据集与文件格式（运行前必读）

**本项目不能直接读取任意 NumPy、`.fvecs` 或文本文件。构建和搜索入口实际读取的是带 8 字节文件头的连续二进制矩阵。文件扩展名不是判断格式的依据，内部字节布局才是。**

### 8.1 通用 bin 矩阵格式

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

### 8.2 双空间 base 数据

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

### 8.3 双空间 query 与 alpha

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

### 8.4 Ground truth 格式

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

### 8.5 fvecs、bvecs 和 ivecs 不能直接混用

标准 `fvecs/bvecs/ivecs` 通常在每一行前保存一次维度，而本项目的 bin 格式只在整个文件开头保存一次 `npts` 和 `dim`。两者不是同一格式。注意，文件后缀可能沿用旧名称；node3 的 OpenImages `.fvecs/.ivecs` 已经是项目 bin，不需要转换。对于真正的标准 vecs 文件，仓库提供转换工具：

```bash
build/tests/utils/fvecs_to_bin input.fvecs output.fbin
build/tests/utils/bvecs_to_bin input.bvecs output.bin
build/tests/utils/ivecs_to_bin input.ivecs output.bin
```

转换后仍应检查输出文件头、数据类型、点数、维度和文件大小。不要仅通过后缀判断转换是否正确。

### 8.6 数据交接检查清单

交接包不包含数据集和已生成索引。开始实验前，需要从师兄或实验服务器确认：

- 数据集和查询文件的位置及格式；
- ground truth 的位置和 top-k；
- 原始索引、PQ 文件和重排映射的位置；
- 使用的服务器、SSD、线程数和编译宏；
- 需要复现的具体脚本和参数组合。

拿到数据后，至少检查：两份 base 的 `N` 是否一致、两份 query 和 alpha 的 `Q` 是否一致、两组 base/query 维度是否分别匹配、alpha 是否为 `float32 Q×1`、ground truth 是否对应相同的 query 顺序和 alpha 配置。对于双空间融合查询，不同 alpha 通常对应不同 ground truth，不能拿固定 alpha 的 truthset 评估另一组 alpha。

`scripts/config.sh`、`run_search.sh` 及部分 `shell/`、`_scripts/` 脚本含绝对路径或历史项目名。建议先复制一份本地配置，再替换 `PROJECT_PATH`、`DATA_PATH`、`DISKANN_INDEX_PATH`、`OUTPUT_PATH` 和 `LOG_PATH`，不要直接假设旧路径仍有效。部分脚本还会使用 `sed` 修改源码并重新编译，运行前务必阅读完整脚本并记录 Git diff。

## 9. 推荐开发流程

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

## 10. 当前已知风险

- 代码源自并扩展了 PipeANN/OdinANN，部分旧命名、注释和脚本仍未统一为 hvs-disk。
- 多个搜索实现可能存在重复逻辑，修改一个实现不会自动影响其他实现。
- 部分实验配置通过编译宏和直接改源码切换，复现实验前要保存完整配置。
- `scripts/`、`shell/` 和 `_scripts/` 的时效性不同；优先使用 `scripts/`，旧脚本需要逐项验证。
- 当前构建系统偏 Linux，依赖和服务器环境需要另行交接。
- macOS 打包可能产生 `._*` AppleDouble 文件；它们会被 CMake 的源码 GLOB 误编译，交接包必须排除。
- `/mnt/nvme3/wz/Index/Openimage-shard` 不是当前正确性基准；使用索引前必须记录完整路径和生成版本。

## 11. 遇到问题时需要提供的信息

提问或记录 issue 时，至少附上：

- 当前 commit 和 `git status --short`；
- 完整 CMake 命令和 `ADDITIONAL_DEFINITIONS`；
- 编译器、Linux 内核、CPU 指令集和 I/O 后端；
- 数据集、索引前缀及关键元数据；
- 完整运行命令；
- 错误日志或最小复现步骤。

这样可以快速区分代码错误、索引格式不匹配、编译宏不一致和服务器环境问题。
