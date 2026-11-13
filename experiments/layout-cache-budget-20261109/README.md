# TXY / T｜XY 缓存对照与 α 预算扫描

第一阶段在 OpenImages 和 HowTo100M 上做 TXY / T｜XY × 共享缓存关 / 开的四格对照。固定原有 M=2 源图、PQ、入口点、已有混合 α 查询及 GT，以及每个数据集归档的 M=2 Starling 节点顺序。两种布局使用同一搜索 harness、候选与精排逻辑。两组均保留每线程 16 MiB 查询内页缓存，开启时额外使用 256 MiB 的进程级 LRU 页面数据预算。每个条目存放实际的 8 KiB 页，LRU 元数据和进程 RSS 另计。共享缓存由遍历填充，TXY 精排可命中已经缓存的共置页。T｜XY 坐标独立读取，保留查询内去重。每个 L 前清空共享缓存并预热完整的 1,000 条查询，随后回放同一查询集至少 10 秒。

第二阶段固定 T｜XY、同一 M=2 Starling 顺序和共享缓存开启，扫描 M=0/1/2/3/Full。复用已有 M=2，M=0/1/3 由同一 Full 源拓扑转换并完整校验，Full 保持原始区间。每个预算重排物理记录并逐记录验证字段和页尾填充。各预算的坐标文件是同一个经过验证的重排坐标组，通过硬链接复用，并在布局验证中重新核对。保持相同的 ID 排列，拓扑页边界随记录大小变化。M=0/2/Full 也在本批计时。

四格先验证所有入选 L 的返回 ID、距离和逻辑搜索计数一致；预算间允许实际搜索轨迹不同。围绕 90% 和 95% Recall@10 做三次独立重复并轮换顺序。QPS、均值和 reader 计数在每次重复内插值，p95 使用距目标最近的实测 L 并给出实际召回。分别保留拓扑和坐标的后端提交、查询内和共享命中、AIO 提交等待、共享缓存开销。后端提交页数和设备 I/O 分开记录。

node3 的 tmux 会话为 `hvs-layout-cache-budget-20261109`，根目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/layout-cache-budget-20261109`。队列持有现有实验共用锁，内存绑定 NUMA 3，OpenMP 最多四线程。正式曲线固定四个物理核，每条曲线前采样五秒，要求每核两个 SMT 线程各自忙碌率不高于 15%，nvme3 利用率不高于 25%。采样保存在 `logs/entry-load.jsonl`，运行期间保留设备统计。生成布局前保留至少 24 GiB 磁盘余量。四格共 24 条正式曲线，预算共 30 条。查 `status.json`、`analysis/progress.json` 和 `logs/queue.log`；每完成一条曲线更新 `RESULTS.md`、`analysis/summary.json` 和 `analysis/matched.csv`。
