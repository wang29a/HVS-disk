# 预实验执行手册

先从[预实验入口](README.md)选择任务。本页保留主机配置、历史目录索引和操作细节；

这里保存截至 2026-11-12 的预实验实现与运行记录，目录名沿用 node3 的实验工作目录，保留脚本之间的相对导入。正式 Full/Merge/No-alpha 与单文件 baseline 的用法仍见仓库根目录 README；本目录不替代正式搜索入口。

代码从各次实验的 canonical `scripts/` 整理，未复制 raw 下的重复快照。`source-manifest.json` 记录原归档相对路径、原文件 SHA256，以及整理后的字节数和 SHA256；路径调整单独标注。C++ 源文件保留实验时的内容，便于与已构建二进制的来源核对。数据集、GT、索引、库、二进制、逐查询结果、运行状态和向量快照不提交；相邻 Markdown 是当时的协议/报告，其中提到的 `analysis/`、图和原始结果需到实验机或本地结果归档查找。报告中的“正在运行”是历史记录，不代表当前队列状态。

归档主线的依赖顺序为 `metric-consistent-20261111` → `laion-six-cell-20261111` → `alpha-mechanism-20261112` → `topology-resident-three-arm-20261112`。其余目录主要保留早期对照和依赖实现，不能将它们混为同一个距离语义下的实验。

| 目录 | 内容与入口 |
|---|---|
| `metric-consistent-20261111` | OpenImages 修复后重建图、六格过滤对照、独立单空间与双空间三臂；`scripts/bootstrap.py` |
| `laion-six-cell-20261111` | Laion 全量重建和精确 DRAM / PQ DRAM / PQ SSD × 过滤开关；`scripts/run_queue.py`，缺口补测见 `scripts/coverage_extension.py` |
| `alpha-mechanism-20261112` | 距离尺度、首发现边归因、活动区间统计、局部 L/r 加密和五权重；`scripts/run_phase_one.py` 后运行 `scripts/run_weight_scan.py` |
| `topology-resident-three-arm-20261112` | 单空间 / 双空间 No-alpha / 双空间 Full，纯拓扑 DRAM、坐标 SSD；先 `scripts/preflight.py`，再 `scripts/run_queue.py` |
| `alpha-audit-20261111` | `mergeIntervals` 和浮点图到磁盘区间的离线检查；`scripts/run_audit.py` |
| `layout-preexperiments-20261102`, `budget-reorder-20261103` | 早期布局、区间预算、Starling 顺序与页面工作量；各自 `scripts/run_all.py` |
| `layout-cache-budget-20261109`, `layout-cache-retest-20261110` | 固定布局与共享缓存预算对照；各自 `scripts/run.py` / `scripts/retest.py` |
| `openimage-single-space-20261110`, `second-space-cost-20261110`, `openimage-matched-graphs-20261110` | 早期单空间资源、第二空间成本与独立图对照；各自 `scripts/prepare.py` / `scripts/endpoint.py` / `scripts/compare.py` |
| `graph-filter-followup-20261110`, `alpha-page-factorial-20261110` | 早期过滤、布局四格及原生 DEG 对照；各自 `scripts/followup.py` / `scripts/factorial.py`；原生 DEG 需要外部源码 |
| `shared-cache-t4-20261108`, `laion-shared-cache-t4-20261105` | 四线程共享缓存实现与校验 |
| `io-time-20261104`, `io-time-numa3-20261108`, `cc3m-highrecall-t4-20261104` | 分阶段 I/O 计时与 CC3M 高召回补测 |
| `concurrency-alpha`, `hotdata-alpha`, `laion`, `formal-32` | 前置编译/输入清单、早期四方案和线程扫描；含历史参数，按各自说明使用 |
| `recall-curves-20261102` | 既有结果的绘图代码；需要本地结果归档 |

## 距离语义和版本边界

原始 HVS-disk 基线版本为 `d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e`。本次归档没有将实验性的距离契约替换进根目录全部生产搜索路径。

11 月 11 日起的隔离实现将双空间构图、精确导航、PQ 导航及精排统一为 `alpha*sqrt(squared_l2_x)+(1-alpha)*sqrt(squared_l2_y)`，修正 `mergeIntervals` 最后一个区间遗漏，并在层内执行度数上限；写盘使用保守 floor/ceil 端点和独立校验。低层 L2 内核仍返回平方距离。单空间是独立构图的拼接向量 L2，使用自身 GT，不能拿双空间 GT 代替。

`metric-consistent-20261111/sources/` 是当时用于生成隔离构建的两份源码，包含区间循环修正，但保留其他历史代码。`scripts/prepare_variant.py` 从它们生成 `variant/`，只替换静态库中的对应对象。实际实验使用 `scripts/build_graph.cpp` 和 Laion 的 `scripts/full_layout.cpp` 写盘；不要把快照覆盖到根目录 `src/`，也不要把快照中的历史 writer 当成当前正式 Full writer。新库/图与历史结果必须按各自来源区分。

## node3 的前置资源与构建

这是一份依赖既有实验资源的代码归档，不是附带数据的“一键从零”包。Linux 需要 g++-11、OpenMP、BLAS/OpenBLAS、jemalloc、libaio、numactl、tmux、Python 3 和 NumPy；本地绘图另需 Matplotlib。旧 node3 使用 Python 3.6，新增本地检查器仅用标准库。依赖版本和编译参数应以对应 `analysis/build.json` 为准。

node3 运行根目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027`，其中 `source/` 是原始 HVS-disk 检出，`build-separated/` 和 `build-single-baseline/` 是两个互不混用的构建，`experiments/` 放运行目录。正式构建定义见根 README；实验驱动在此基础上单独链接自己的工具。

`metric-consistent-20261111/scripts/bootstrap.py` 需要前置 `concurrency-alpha/analysis/build.json` 中的编译 flags、原静态库路径及 SHA256，并要求原库哈希匹配；Laion 还需要 `concurrency-alpha/analysis/manifest.json` 中的已有查询、GT 与索引身份。两个 JSON 和它们引用的文件保留在 node3，不在提交中。重新部署到新机器必须先准备两个基础构建和匹配的数据/索引，使用 `concurrency-alpha/scripts/pipeline.py` 的 `build_driver()`、`configurations()` 生成对应清单；该文件的 `main()` 会启动完整历史性能扫描，不要为生成清单而直接运行它。脚本保留版本门禁，不能将源版本断言简单删掉来复用旧结果。

OpenImages 原始矩阵在 `/mnt/nvme3/wz/dataset/Openimage_disk`；其坐标与 PQ 路径还引用 `/mnt/nvme3/wz/Index/Openimage` 和早期 `hvs-disk-paper-20260920/experiments/formal-32/openimage/full/`。Laion 矩阵在 `/mnt/nvme2/wz/disk/dataset/Laion`，坐标在 `/mnt/nvme3/wz/Index/Laion-shard/disk_index_data`，PQ 复用运行根目录的 `experiments/laion/full/`。各脚本中的 `DATA`、`COORD`、`PQ`、`BUILD`、`STORE` 是主机配置，换机器时逐项检查，不能只改一个根路径。

矩阵文件名虽然常以 `.fvecs/.ivecs` 结尾，本实验使用的是文件开头一次 `[uint32 rows, uint32 dims]`，随后连续 float32/uint32 行数据；不是每行有维度头的经典 fvecs。两空间必须共享 ID 顺序。查询向量、alpha 和 GT 行数、顺序及权重必须匹配；复用 GT 前做语义抽检。Laion 五权重使用现有配套 GT，不能把 0.5 的 GT 用于其他权重。

## 部署与串行执行

将需要的目录部署为 `experiments/` 下的同名目录，并同时保留它依赖的兄弟目录。已有 node3 工作目录包含结果、锁和校验清单，更新前先比对代码哈希及冻结身份，不使用 `rsync --delete`，不覆盖已冻结实验；改变会影响搜索语义的代码时使用新实验命名空间并同步修改依赖路径。

以下是新启动命令的形式。先确认 NUMA 3 的使用约束、空闲窗口、磁盘空间和先行阶段已完成，再启动当前需要的一个阶段；不要一次创建全部会话，也不要为验证此提交重跑全量实验。

```bash
cd /mnt/nvme3/wz/hvs-disk-build-20261027/experiments
# OpenImages：构建/校验隔离实现，GT 核验、重建、六格及三臂。
tmux new-session -d -s hvs-metric-consistent-20261111 \
  'cd /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/metric-consistent-20261111 && mkdir -p logs && python3 -u scripts/bootstrap.py >>logs/queue.log 2>&1'
```

OpenImages 完成后，Laion 在自身目录运行 `python3 -u scripts/run_queue.py`。之后在 alpha 机制目录依次运行 `python3 -u scripts/run_phase_one.py`、`python3 -u scripts/run_weight_scan.py`。三臂驻留对照先做 `python3 scripts/preflight.py`，再在 tmux 中运行 `python3 -u scripts/run_queue.py`；它还会等待权重扫描终态。沿用相同 tmux 启动形式并更换会话名、目录和入口即可。

当前主线搜索固定四线程、CPU 72–75、NUMA 3 内存绑定，BLAS 单线程；OpenImages 建图八线程，Laion 建图最多二十四线程。它们共用 `budget-reorder-20261103/queue.lock`。早期线程扫描的 32/64 线程配置仅作历史记录，不是目前默认配额。计时使用运行期 CPU affinity 和 VMA 内存策略验证、外部磁盘/CPU 干扰记录、超时、有限补跑和 pending 清单。查看 `status.json`、`progress.json`、验收点与 pending；日志更新时间不是进展证明。

`alpha-mechanism-20261112/scripts/local_offline_sequence.py` 是本地分析入口，包含经 SSH/scp 读取 node3 资源；其中独立单空间 top-M 并集仅是覆盖率参照，不是任意双索引算法的理论上界。`topology-resident-three-arm-20261112/scripts/relocate_checkpoints.py` 是一次磁盘空间恢复工具，会迁移特定检查点并建立链接；它不属于正常执行顺序，不要自动运行。纯拓扑驻留对照也不是 256 MiB 有限共享缓存对照。

## 提交前的轻量检查

```bash
python3 experiments/check_sources.py
python3 experiments/check_sources.py --unit
```

第一个命令核验归档哈希、Python 语法和 shell 语法，不启动实验。第二个额外运行 NUMA/队列与 Laion 协议的已有单元回归，需要 NumPy；它不读取完整数据集，也不通过 SSH 启动任务。Linux C++ 运行验证仍需在 node3 编译并运行距离、区间、搜索 selftest 和 `test_edge_audit.py`。语法检查通过不能代替性能复现。

本地历史绘图/同步脚本仍读取按日期命名的结果树。运行前设置 `export HVS_RESULT_ARCHIVE=/path/to/experiment_results`，指向已有本地结果归档；其中应有 `2026-10-27-concurrency-alpha/raw/results/` 等原结构。`cc3m` 同步器还依赖归档中的 `2026-11-04-local-results/整理结果.py` 及既有汇总文件，这些属于结果维护环境，不由本代码包生成。`concurrency-alpha/analyze_completed.py` 同样针对原始 raw/results 布局；不要在缺少结果时把空输出解释成实验完成。
