# 代码与索引参考

本页供修改构图、搜索或文件格式时查阅。首次运行请先走[构建与搜索流程](workflow.md)，不要从所有搜索变体中猜一个入口。

## 两套实现的距离语义

根目录历史 L2 主路径和近期隔离实验不是同一距离契约：

| 环节 | 根目录历史实现 | `metric-consistent-20261111` 起的隔离实现 |
|---|---|---|
| 双空间构图 | 融合平方 L2 | 融合两空间的欧氏距离 |
| PQ 导航 | 直接融合 PQ 平方距离 | 各空间完整 ADC 求和后开方，再融合 |
| 精排 | 各空间精确平方距离开方后融合 | 与构图、导航统一的融合目标 |

隔离实现采用 `alpha*sqrt(d_x²)+(1-alpha)*sqrt(d_y²)`，以已有 GT 的实际目标为准；低层 L2 kernel 仍返回平方距离。它还修正了 `mergeIntervals` 末区间遗漏、层内度数上限和写盘区间覆盖。修正后的图需要重建，不能只换搜索二进制就将旧实验升级为新语义。

实验的 `sources/` 是生成隔离实现的历史源码快照，其中还保留其他旧路径；不要覆盖到根目录 `src/`。`prepare_variant.py` 生成变体，实际图布局由实验自己的 builder/writer 输出。原生 DEG 复现、当前搜索器的精确 DRAM、PQ DRAM 与 PQ SSD 也属于不同对照，报告时需明确。

## 页面与拓扑记录

页面固定为 `8192` 字节。分离式 Full 的 metadata、coordinate、topology 是三个独立文件；单文件 baseline 使用另一种物理封装，不能把逻辑三段直接当成二者共用的字节布局。

每条拓扑记录为：

```text
[邻居数量 uint32]
[邻居 0：ID uint32 + 固定数量的 int8 区间端点]
[邻居 1：ID uint32 + 同样数量的端点]
...
```

每个邻居总是占用 `max_alpha_range_len` 对区间，没有独立的变长 `range_size`。未用槽补零。记录长度、每页节点数与尾部 padding 都由 metadata 决定：

```text
coord_len        = (data_dim + datal_dim) * sizeof(T)
ncoord_per_page  = 8192 / coord_len
fixed_topo_size  = 4 + max_nbr_len * (4 + 2 * max_alpha_range_len)
ntopo_per_page   = 8192 / fixed_topo_size
```

例如实际 degree 为 40 时，Full 的 6 slots 是 644B／12 节点每页，7 slots 是 724B／11 节点每页，Merge-2 是 324B／25 节点每页，No-alpha 是 164B／49 节点每页。这些只是给定 metadata 的计算结果，不是所有构建的固定密度。

转换器必须按源页读取、跳过 padding，并按目标密度重新打包。只改 metadata、连续遍历无 padding 的 node array，或沿用旧 partition，都会破坏布局。`split_index` 与 `check_reorder_topo` 仍有旧 DiskANN 格式假设，不能直接处理或校验当前双空间格式。

## 搜索实现与入口

| 代码 | 用途 |
|---|---|
| `tests/build_hybrid_disk_index.cpp` | 根目录构图 CLI，`single_file_index=0/1` 选择封装 |
| `tests/search_disk_index.cpp` | 解析双空间 query、alpha、GT 和搜索参数 |
| `src/search/starling_page_rerank_search.cpp` | `mode=6`，重排后的 Full/Merge/No-alpha 主要路径 |
| `src/search/beam_search.cpp` | 包含单文件 baseline 的 `mode=0` 路径 |
| `src/search/rerank_search.cpp` | `mode=3`，历史原顺序索引的兼容路径 |
| `src/search/page_search.cpp` | 页级搜索与 alpha 过滤 |
| `src/search/coro_search.cpp`, `pipe_search.cpp` | 其他历史执行变体，需单独验证 |

Full/Merge 从邻居记录内的区间判断边是否活动；No-alpha 的槽数为零，语义是所有邻居都可以展开。`strategy` 是位组合，不能脱离源码、编译宏及配套文件解释。`strategy=17` 也不等于已经给共享缓存分配了某个预算；查询内页复用、跨查询缓存、逻辑请求和物理读页要分开统计。

旧 mode 3 测试可能提示缺少 alpha sidecar；它和 mode 6 直接从拓扑内读取区间的路径不同。不要凭同一条日志就判断所有搜索分支都关闭了过滤，也不要用 mode 3 读取已验证不兼容的 Starling relayout 来规避问题。

## 修改代码时查哪里

| 关注点 | 位置 |
|---|---|
| 构图与 alpha 区间 | `src/hybrid_index.cpp`：`occlude_list`, `prune_neighbors`, `mergeIntervals`, `get_use_range` |
| 邻居与磁盘节点类型 | `include/neighbor.h`, `include/query_buf.h` |
| Metadata 与偏移 | `src/ssd_index.cpp`, `include/ssd_index.h` |
| 格式转换 | `tests/merge_alpha.cpp`, `tests/remove_alpha.cpp` |
| 异步读取与缓存接入 | `src/utils/linux_aligned_file_reader.cpp`, `src/buffer/` |
| 新实验浮点图／量化检查 | `experiments/metric-consistent-20261111/scripts/validate_graph.py` |
| 新实验 Full writer／检查 | `experiments/laion-six-cell-20261111/scripts/full_layout.cpp` |
| 区间与首发现父边统计 | `experiments/alpha-mechanism-20261112/scripts/edge_audit.cpp` |

实验校验器对应其专用格式和命令契约，不是所有历史索引的通用检查器。修改磁盘记录时，要一起检查 writer、reader、转换器、metadata、偏移与外部 Starling 契约；覆盖空邻居、最大度数、空槽、alpha 0/100 和部分尾页。已有多个 reader 的重复逻辑，只修一个文件不会自动修复其他变体。

性能报告应保留等召回的 QPS、延迟、拓扑与坐标提交页、两阶段计时及实际缓存配置，并记录版本、图身份、GT、线程与 NUMA。过滤了多少边不等于少读了多少页；布局与搜索器一起改变的比较不能说成纯布局消融。
