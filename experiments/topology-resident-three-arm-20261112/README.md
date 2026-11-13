# 三臂：纯拓扑驻留 DRAM，坐标保留 SSD

先完成五权重扫描，再持同一全局锁串行复测。数据集 OpenImages，三臂为等总维单空间（single）、双空间 No-alpha（compact）、双空间 Full（full）。每臂测试 TXY 与 T｜XY 两种物理布局，共六格；四线程，CPU 72–75，NUMA 3，搜索 BLAS 单线程，无 Starling、无跨查询共享缓存，查询内缓冲与原实验相同。数据、修复后的图、入口、PQ、原始 ID 顺序和 GT 全部复用，不重新建图。

现有 r 和坐标页审计见 `../alpha-mechanism-20261112/THREE-ARM-R-COORD.md`。同一臂内的旧 r 相同或解耦更小，TXY 的坐标零新增提交来自遍历页面复用；跨臂则精排量约为 35.5、64.6、76–80，不能忽略。

“纯拓扑驻留”有严格定义：单独复制 canonical topology 的全部字节到 DRAM。遍历直接读取这些记录，不提交拓扑 SSD I/O，也不把物理 TXY 页放进查询缓存。精排继续通过原 aligned direct-AIO 从各自布局的 SSD 坐标页读取。不能把整份 TXY 页面当作拓扑缓存载入内存，否则 XY 也会全部驻留。

这一控制实验有意剥离“遍历顺带读到坐标”的效应，不等同于原 TXY 系统自然增加缓存后的性能。此时 TXY 精排也要单独读取坐标。OpenImages 两种布局均为每页一个坐标记录，固定参数下坐标页数应等于实际精排候选数。每个原始测点都检查 `topology_medium=DRAM`、`coordinate_medium=SSD`、`topology_submitted_pages=0`、内存拓扑节点读取数等于展开数、坐标提交页数等于实际精排数，防止错误启用全 DRAM。

每臂使用同一份训练冻结配置，两布局不各自挑参数。原 500 条训练查询上的 recall、PQ 评估和展开轨迹只依赖图、目标和参数，复用既有完整训练记录。新 I/O 代价由格式确定：拓扑提交为零，坐标页数等于精排节点数。按 88/90/92/94/95/96/97% 训练召回带，分别以坐标页数优先、导航评估数优先选择代表；不使用报告集延迟选择。冻结后对每个代表在新实现中复核原训练 recall/PQ/展开数，以及预测的坐标页数；不一致即停止。

500 条独立报告查询使用原先的划分，单空间使用自己的拼接 L2 GT，双空间使用原 mixed-alpha GT。先核验两布局的轨迹、返回 ID 和距离逐位一致，再做三轮、每点至少 10 秒的正式计时。同一配置两布局交替执行，重复间轮换方法顺序；CPU/磁盘干扰、运行期 NUMA 证据、超时、有限重试和 pending 规则沿用现有运行器。90/95% 只在冻结曲线的实测区间内插值，无覆盖则写明，不外推，也不插值 p95/p99。

输出保留 L/r、实际精排量、精确距离调用、拓扑内存驻留字节、坐标提交与命中、各阶段时间、QPS 及尾延迟。比较三臂剩余坐标/计算成本时仍须承认图、融合目标和 GT 的差异，不能把跨臂绝对延迟直接说成纯双向量开销。

`scripts/layout_store.h` 和 `layout_engine.cpp` 是隔离复制，未改旧实验二进制或生产路径。`LAYOUT_TOPO_DRAM=1` 仅改变供页方式；`LAYOUT_DRAM=1` 只用于非计时诊断，正式计时禁止。预检在 512 节点功能图、真实向量和 PQ 前缀上，验证三臂×两布局全部分支与旧实现的轨迹／ID／距离一致；开驻留后六格均为零拓扑提交、r=40 时 40 个坐标提交。功能图不作为性能或召回证据。

构建和预检入口为 `python3 scripts/preflight.py`，使用既有修复库与 g++-11 构建配置；编译及短回归限 CPU 76–77，不占正式搜索的 72–75 核。结果在 `analysis/preflight.json`。正式队列入口为 `scripts/run_queue.py`，tmux `hvs-three-arm-topology-resident-20261112`；等待 `alpha-mechanism-20261112/weight-scan/status.json` 进入终态，并取得全局 `budget-reorder-20261103/queue.lock`，不会与权重扫描并发。扫描失败也会如实记录后再接续这项独立实验。

远端目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/topology-resident-three-arm-20261112`。一次只生成一个本队列的耦合页镜像，验证所有字段／padding 后使用，每臂完成时保存哈希及生成身份并回收该镜像，保留规范源图和坐标；至少预留 512 MiB，空间不足明确停止，不删除历史索引。`status.json` 记录等待与运行阶段，每臂 `analysis/resident-summary.json` 与最终 `analysis/three-arm-resident-results.json` 记录结果和待补点。

首次接续因 nvme3 无法容纳临时耦合文件及 512 MiB 余量而停止。已将旧统一目标实验的两份离线浮点图检查点和静态构建归档迁到 nvme2，逐文件 SHA256 验证并保留原路径链接；查询用的 canonical 拓扑、坐标、PQ 和可执行文件未移动。迁移记录在远端 `analysis/checkpoint-relocation.json`。队列已恢复，实际状态以 `status.json` 和当前测点验收为准，`queue.log` 保留首次停止的历史记录。
