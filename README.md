# hvs-disk

hvs-disk 是一个基于 C++17 的 SSD 双空间近似最近邻检索实现。同一个对象有两份向量，例如图像向量和文本向量；查询通过 `alpha` 调整两个空间的权重，在同一张图上搜索。`alpha=1` 只看第一个空间，`alpha=0` 只看第二个空间。

项目提供图索引构建、PQ 编码、拓扑格式转换和磁盘搜索。代码继承了既有图检索实现，适合继续研究双空间检索的元数据、页面布局和 I/O 开销。

## 从哪里开始

第一次使用，先看[数据格式](docs/data-format.md)，再按[构建与搜索流程](docs/workflow.md)运行。完整流程是：

```text
两份对齐的 base 向量
  → 构建 Full-alpha 图与 PQ
  → 派生 Merge-alpha / No-alpha
  → 三种格式分别用外部 Starling 重排
  → 加载对应索引，执行搜索与 Recall 验证
```

如果要接着做这段时间的预实验，入口是 [experiments/README.md](experiments/README.md)。预实验使用独立构建和搜索器，依赖 node3 已有资源；根目录的构建命令不会自动启用这些实验实现。

## 支持哪些方案

Full、Merge 和 No-alpha 共享同一次构建的向量、PQ 和源图，区别在于如何存储及使用边上的 alpha 区间。

| 方案 | 拓扑中的 alpha 信息 | 物理布局与搜索入口 |
|---|---|---|
| Full-alpha | 保留完整区间 | 拓扑与向量分离；拓扑独立做 Starling 重排 |
| Merge-alpha | 每条边合并到至多两段区间 | 与 Full 使用同一类搜索入口 |
| No-alpha | 删除区间，保留邻居 ID，不做区间过滤 | 记录更紧凑，仍须独立进行 Starling 重排 |
| 单文件 baseline | 保留完整区间 | 向量与拓扑放在一起，不用 Starling，使用独立编译配置 |

三种分离式方案使用 `mode=6, mem_L=0, strategy=17`；单文件 baseline 使用 `mode=0`。这些参数必须与编译宏和索引布局配套，不能只换一个 mode 就共用二进制。Baseline 的完整命令见[单文件 baseline](docs/baseline.md)。

## 编译

在 Linux 上构建。依赖 CMake、支持 C++17 的编译器、OpenMP、BLAS/MKL、jemalloc 和 libaio；当前 CMake 使用 MKL 头文件路径，需按服务器环境配置。已有 node3 环境可使用 GCC 11。

```bash
git clone git@github.com:wang29a/HVS-disk.git
cd HVS-disk

export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build \
  -DBUILD_WITH_PQ=ON \
  -DCMAKE_CXX_COMPILER=g++-11
cmake --build build -j8
```

这套配置用于 Full/Merge/No-alpha。成功后工具位于 `build/tests/`。接下来按[构建与搜索流程](docs/workflow.md)设置数据路径、构建索引并运行搜索；仓库不附带数据集、GT 或生成好的索引。

## 数据格式要先确认

Base、query 和 alpha 使用的是**整个文件只有一个 8 字节头的二进制矩阵**：

```text
[行数 uint32][维度 uint32][按行连续的数据]
```

两份 base 必须有相同的行数和对象 ID 顺序，两份 query 与 alpha 也必须逐行对应。Alpha 是 `float32` 的 `Q×1` 矩阵；GT 必须对应同一批 query、同一数据集和同一组 alpha。

**不要按扩展名判断格式。** node3 上部分 `.fvecs/.ivecs` 已经是上述格式；经典 fvecs 的“每行一个维度头”不能直接使用。文件大小、GT 布局和转换方式见[数据格式](docs/data-format.md)。

## Starling 是外部依赖

使用 [wang29a/starling](https://github.com/wang29a/starling)，历史验证版本为 `7437c4848a83e7bf62558c5eab1eb999feed8537`，不包含在本仓库中。它生成 partition 和拓扑重排顺序，hvs-disk 再加载重排后的索引。

Full、Merge、No-alpha 的记录宽度不同，必须各自生成 partition；坐标和 PQ 可以共享。安装方法、node3 兼容处理和输出文件约定见 [Starling 使用说明](docs/starling.md)。

## 预实验与版本边界

`experiments/` 归档了六格过滤对照、alpha 权重扫描、共享缓存、布局和单／双空间对照，以及校验与分析工具。研究代码修正了距离目标、区间合并和写盘量化等问题，但这些修正保留在**隔离实验实现**中，尚未统一合入根目录所有构图和搜索分支。

因此，复现近期结果应从[预实验入口](experiments/README.md)开始；不要把旧图、正式二进制和实验二进制混用。具体实现差异见[代码与索引参考](docs/implementation.md)。

## 代码在哪里

| 目录 | 内容 |
|---|---|
| `include/`, `src/` | 索引、距离、搜索、I/O 和缓存实现 |
| `src/search/` | 各种搜索实现，Starling 路径在 `starling_page_rerank_search.cpp` |
| `tests/` | 构建、搜索、转换和检查工具的命令行入口 |
| `experiments/` | 按实验目录组织的研究驱动、独立实现和运行记录 |
| `scripts/`, `shell/`, `_scripts/` | 继承的实验脚本，使用前核对路径、参数及源码修改行为 |
| `third_party/liburing/` | 随仓库提供的 liburing 源码 |

`run_search.sh` 是旧 mode 3 的参数示例，不是三方案重排搜索的通用入口。第一次运行优先使用文档中的明确命令。

进一步开发时阅读 [AGENTS.md](AGENTS.md) 和[代码与索引参考](docs/implementation.md)。旧的机器路径、逐轮测试数字和故障记录保存在[历史交接记录](docs/archive/handoff-notes-2026-09.md)，不作为当前使用入口。
