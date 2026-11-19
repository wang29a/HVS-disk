# 静态实验复用清单

以下用 A 表示反向邻居未满时追加，B 表示每次反向插入重剪。A 语义的历史图仍可能在区间边界修正、构建身份或入口上与正式 A1 不同。单空间 Vamana 和固定权重专用图使用各自的构图口径。重跑时间是同条件、已有冻结 GT 和调参文件的墙钟预算，默认复用索引，不含等待独占资源。括号内区分重建代价与根据计时计划估出的下界。

| 结果 | 图身份与口径 | 脚本与冻结参数 | 预计重跑 |
|---|---|---|---|
| Laion 2M 过滤开关、反向边 A/B | A1/A2、B1/B2 都有 DRAM；SSD 为 A1/B1。开关均加载相同完整旁表，拓扑均为紧凑格式。 | `backlink-ab-20261115/scripts/run_function_v2.py` 进入 `stage_checks.py` 的正式 DRAM/SSD 阶段。R40/L200/C750，构图 8 线程，搜索 4 线程，5 个 DRAM 权重、SSD α=0.1/0.5，3 轮，500/500 调参/报告，95% ID Recall@10。 | 复用四图约 30–45 分钟。原正式整段实测 **2.70 小时**，四图构建合计约 124 分钟。 |
| Laion 2M 五权重多图 | 统一图是 `stage2-v2/graphs/unified` 的历史 A 语义图。M 是双 Vamana，Or/F 是固定权重 DEG。统一图开/关均加载旁表。 | `multigraph-20261113/scripts/enqueue_three_layers.sh` 接续 `run_queue.py`；`runtime/frozen.json` 冻结三种工作量规则。R40/L200/C750，无归一化，α=0.1/0.3/0.5/0.7/0.9，SSD 与纯拓扑驻留，4 线程、3 轮，90%/95% ID Recall@10。 | 正式 1,572 个实计时点，每点至少 10 秒，纯计时下界 **4 小时 22 分**，含加载/守护约 5–7 小时。X/Y 与五张专用图的原构建合计约 4 小时 38 分，含复用的中心历史图。 |
| OpenImages 全量多图 | 统一图是 `review-20261113/graphs/openimage` 的历史 A 语义图。等量双索引为 Vamana，近邻路由为 α=0.1/0.5/0.9 专用图。统一图无旁表。本行归档未使用后来构建作连通性诊断的 OpenImages B。 | `edbt-evidence-20261115/scripts/openimages.py`；`runtime/openimages/search/*/frozen.json`。原 mixed-α GT、500/500、4 线程、3 轮、紧凑拓扑和公共坐标。 | 复用图和冻结配置约 15–35 分钟，重建五张图另计，下面脚本参数固定为 8 线程。 |
| 原生 TXY 与 compact SSD | OpenImages 使用上述历史 A，Laion 2M 使用 v2 历史 A；两侧从同一来源构造索引。原生是统一图原生 beam/TXY，已修正导航距离和计数。 | `edbt-evidence-20261115/scripts/native_compare.py`，有效续测入口 `native_guard_recovery.py`。OpenImages mixed、Laion 五权重，1/2/4/8/16/32 线程，3 轮，无共享缓存，补充 4 GiB 预算下的有效记录，每个 L/r 点至少 10 秒。 | 仅已收录内插端点共 882 个独立点，纯计时下界约 2 小时 27 分，复用图/工作点预算约 3–5 小时。 |
| 拓扑驻留三臂 | OpenImages 的历史 metric-consistent 版本，双空间 No-alpha/Full 来自早期 A 反向策略，具体图和后来的 v2 A/B 分开使用。等总维单空间臂没有双空间 A/B 标签。 | `topology-resident-three-arm-20261112/scripts/preflight.py` 与 `run_queue.py`。4 线程、3 轮，纯拓扑复制到 DRAM，坐标仍在 SSD，TXY/T｜XY 同臂同一 L/r，训练冻结、报告匹配 90%/95%。 | 已有六格及冻结参数约 15–35 分钟，耦合文件重新物化另留 10–25 分钟。 |
| 拓扑驻留并发诊断 | OpenImages mixed 和 Laion α=0.5 均沿用上表原生/compact 源图，均为历史 A。compact SSD 与拓扑驻留共用无旁表索引。 | `edbt-evidence-20261115/scripts/concurrency_diag.py`。原生 SSD／compact SSD／compact 纯拓扑驻留，4/16/32 线程，3 轮，500 条报告查询，95% 预冻结，4 GiB 上限。 | 已归档 54 个首轮守护窗口的样本时间合计约 19 分钟，复用图约 20–35 分钟。 |
| Laion 2M 无旁表内存对照 | 同一历史 A 统一图的载入／零旁表对照，另有固定 α=0.5 专用图；B 的零旁表 RSS 仍须独立验证。 | `paper-priorities-20261114/scripts/run_second.py` 起第一轮，`edbt-evidence-20261115/scripts/no_sidecar_repeats.py` 补两轮。固定五权重、4 线程、95% ID Recall@10，沿用 weighted-pages 冻结 L/k，3 轮。 | 每点 10 秒，复用计划约 10–30 分钟。 |

证据优先级为图构建身份和源码策略，其次是执行清单，最后才是目录名。上表中的“原生 TXY”与单空间 OdinANN 插入参照是不同实验。正在排队的新双 OdinANN 也不计入这些静态完成项。

本地汇总入口为 `experiment_results/2026-11-15-backlink-ab/formal-ab-v2/RESULTS.md`、`2026-11-14-multigraph/RESULTS.md`、`2026-11-12-three-arm-topology-resident/RESULTS.md`、`2026-11-15-concurrency-diagnosis/RESULTS.md` 和 `2026-11-15-edbt-evidence/RESULTS-COMPLETE.md`。执行入口路径相对 HVS-disk 的 `experiments/`。部分历史队列保留 node3 路径，启动前需按本表核对资产及冻结配置。
