# 预实验入口

2026-11-16 更新：反向边B函数与三处修正版作者实现的10000条输入输出逐位一致，小图统计门槛通过，Laion2M正式A/B与高并发拓扑驻留诊断均已完成。先读[最新有效性说明](VALIDITY-2026-11-16.md)；[11月13日通知](VALIDITY-2026-11-13.md)保留为历史边界，不代表所有旧点已重测。

这里保存截至 2026-11-16 的预实验代码与运行记录，主要回答 alpha 过滤、页面密度、缓存、单双空间布局和统一图／多图架构的性能问题。目录名与 node3 工作目录一致，脚本之间有依赖，应保留兄弟目录结构。

第一次运行普通 Full/Merge/No-alpha，请先看[项目首页](../README.md)和[构建与搜索流程](../docs/workflow.md)。本目录是研究用的隔离实现，依赖已有数据、PQ、GT 和构建清单，不能只 clone 仓库就启动完整实验。

## 近期实验入口

| 想了解什么 | 实现与协议 | 已归档的结果 |
|---|---|---|
| 反向插边是否改变过滤结论 | [A/B冻结协议](backlink-ab-20261115/README.md) | [函数与小图门槛](backlink-ab-20261115/FUNCTION-RESULTS.md)、[正式Laion2M结果](backlink-ab-20261115/RESULTS.md) |
| 高并发回落是否与拓扑供页有关 | [诊断协议](edbt-evidence-20261115/CONCURRENCY-DIAG.md) | [54批驻留对照](edbt-evidence-20261115/CONCURRENCY-RESULTS.md) |
| 原生基线、无旁表、OpenImages多图与松弛剪枝 | [EDBT补测](edbt-evidence-20261115/README.md) | [完成结果与线程边界](edbt-evidence-20261115/RESULTS.md) |
| OpenImages修正图配对与中心度数扫描 | [四项协议](paper-priorities-20261114/README.md) | [已完成记录](paper-priorities-20261114/RESULTS.md) |
| 修正剪枝边界，核验统一图／专用图差别 | [U 形与构图复核](u-shape-20261113/README.md) | [v2 结果](u-shape-20261113/PRUNING-V2-RESULTS.md)、[来源核对](u-shape-20261113/SEMANTICS-PROVENANCE.md) |
| 单空间双索引／权重专用图是否能替代统一图 | [冻结协议](multigraph-20261113/README.md)、[执行手册](multigraph-20261113/RUNBOOK.md) | [三层损失诊断](multigraph-20261113/THREE-LAYER-RESULTS.md)、[Laion 2M 磁盘结果](multigraph-20261113/RESULTS-LAION-2M.md) |
| 过滤在精确 DRAM、PQ DRAM、PQ SSD 下分别有什么作用 | [OpenImages 六格与三臂](metric-consistent-20261111/README.md) | [六格结果](metric-consistent-20261111/STAGE-RESULTS-2026-11-11.md) |
| 同样的过滤结论在 Laion 上是否成立 | [Laion 六格](laion-six-cell-20261111/README.md) | [结果](laion-six-cell-20261111/RESULTS.md) |
| 不同 alpha 的表现为何不同 | [机制分析与局部加密](alpha-mechanism-20261112/README.md) | [权重扫描](alpha-mechanism-20261112/WEIGHT-SCAN-RESULTS.md)、[分析补充](alpha-mechanism-20261112/COMPLETED-RESULTS-SUPPLEMENT.md) |
| 拓扑已在 DRAM 后，三臂的解耦收益还剩多少 | [拓扑驻留三臂](topology-resident-three-arm-20261112/README.md) | [结果](topology-resident-three-arm-20261112/RESULTS.md) |

近期实现将双空间构图、PQ 导航和精排统一到加权欧氏距离，修正区间合并和量化覆盖；这些修正没有统一合入根目录全部搜索路径。新图、新库和旧图、旧二进制要按身份区分，不能混合结果。代码差异见[实现参考](../docs/implementation.md)。

## 要继续运行

Node3 的工作根目录是 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments`。以下是依赖顺序和入口，**不是一次启动四个任务的清单**：

| 顺序 | 工作目录 | 入口 |
|---|---|---|
| OpenImages 构建与对照 | `metric-consistent-20261111/` | `python3 -u scripts/bootstrap.py` |
| Laion 构建与对照 | `laion-six-cell-20261111/` | `python3 -u scripts/run_queue.py` |
| 机制分析、局部加密、再做权重扫描 | `alpha-mechanism-20261112/` | 先 `scripts/run_phase_one.py`，再 `scripts/run_weight_scan.py` |
| 拓扑驻留三臂 | `topology-resident-three-arm-20261112/` | 先 `scripts/preflight.py`，再 `scripts/run_queue.py` |

长任务在 tmux 内串行运行，共用队列锁；近期搜索固定四线程、NUMA 3，构图有单独的线程上限。恢复前确认主机使用约束、磁盘与内存资源、输入哈希和已冻结的配置。历史 Markdown 中的“正在运行”只是当时记录，实时进度看 node3 的 `status.json`、`progress.json`、验收点和 pending 清单。

当前多图实验已完成Laion 2M的1,572个三轮计时测点，pending为0，未自动启动OpenImages。`multigraph-20261113/scripts/enqueue_three_layers.sh` 是已批准协议修订后的恢复入口；它先验证B/C遍历损失，再复用原图、GT和已完成点。原 `enqueue.sh` 保留作首轮追溯，旧计划不覆盖，新计划单独保存。U形首轮专用图已判为无效历史对照；不要从旧脚本重启那批结果，也不要把修复代码直接覆盖到生产源文件。

首次部署、编译清单依赖、具体数据/PQ 路径、tmux 命令和本地结果同步方式见[执行手册](RUNBOOK.md)。旧驱动不一定适合新的资源限制，不要直接重启全部历史队列。

## 只检查代码，不跑实验

在仓库根目录运行：

```bash
python3 experiments/check_sources.py
# 有 NumPy 时，额外执行已有的短回归：
python3 experiments/check_sources.py --unit
```

检查器保留历史218份归档身份与11月14日的86份研究身份，另核对11月16日新增代码清单、冻结协议前缀、Python/shell语法；短回归覆盖NUMA、队列、子集GT、U形、多图、度数判读和函数门槛阈值。它会跳过运行时目录，不会启动搜索队列或SSH计算。Linux编译和功能检查记录见 [VALIDATION.md](VALIDATION.md)；这些检查不等于在新环境重跑了性能实验。

## 早期对照在哪里

| 主题 | 目录 |
|---|---|
| 页面布局、区间预算与 Starling | [布局预实验](layout-preexperiments-20261102/README.md)、[预算与重排](budget-reorder-20261103/README.md) |
| 共享缓存及布局交互 | [共享缓存](shared-cache-t4-20261108/README.md)、[缓存预算](layout-cache-budget-20261109/README.md)、[布局缓存复测](layout-cache-retest-20261110/README.md) |
| 单空间、第二空间与独立图 | [单空间资源](openimage-single-space-20261110/README.md)、[第二空间成本](second-space-cost-20261110/README.md)、[独立图对照](openimage-matched-graphs-20261110/README.md) |
| 过滤与页格式四格 | [四格协议](alpha-page-factorial-20261110/README.md)、[过滤后续](graph-filter-followup-20261110/README.md) |
| I/O 分项及高召回补测 | [I/O 计时](io-time-numa3-20261108/README.md)、[CC3M 补测](cc3m-highrecall-t4-20261104/README.md) |
| 早期四方案及线程扫描 | [线程扫描](concurrency-alpha/README.md)、[热数据](hotdata-alpha/README.md)、[Laion](laion/RUN.md) |

早期结果保留其原图、距离公式和工作负载限制，不与修复后实验拼成一条因果结论。完整历史目录索引放在[执行手册](RUNBOOK.md)。

本仓库不存数据集、索引、GT、二进制、逐查询结果或大体积向量快照。报告中的原始JSON、图和日志需要从node3或本地结果归档读取；`source-manifest.json`保留历史归档身份，`research-manifest-20261114.json`与`research-manifest-20261116.json`记录相应归档源码和协议身份，不能把它们当作运行结果清单。`relocate_checkpoints.py`是一次性空间恢复工具，不属于日常执行流程。

## 本次归档与恢复边界

11月16日归档覆盖 `paper-priorities-20261114`、`edbt-evidence-20261115`、已停止的 `backlink-range-audit-20261115` 和完成的 `backlink-ab-20261115`。不会在clone或检查源码时启动任何实验。已完成队列不需要再次运行；源码历史中的参数、主机绝对路径和恢复状态不能当作新机器的默认值。

最新恢复入口及其依赖见[补测归档执行说明](RUNBOOK-20261116.md)。`backlink-range-audit` 的幂等门槛失败是历史方法验证，不是图错误率；原整图99%一致率门槛也已被函数/统计门槛替代。冻结README只在末尾追加，最新结果应从上表的RESULTS页读取。
