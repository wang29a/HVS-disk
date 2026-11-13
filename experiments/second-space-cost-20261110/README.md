# 第二空间计算与坐标成本对照

OpenImages 固定既有 M=2 双空间图、Starling 节点顺序、入口点、两套 PQ 和同一批 1,000 条查询。所有查询固定 α=1，沿用原图的 α 过滤。三组共享同一份拓扑文件、256 MiB 共享拓扑缓存、查询内复用和精排候选。A 读取 XY 页并计算两空间距离。B 访问与 A 相同的 XY 页，跳过第二组 PQ 表、编码 gather/lookup、坐标复制和精确距离。C 同 B 只计算 X，坐标改为只存 X。三组进程仍加载相同的 PQ 文件并分配相同查询 scratch，以隔离查询期成本。A/B 保持完全相同的坐标页请求及字段请求掩码，B/C 的读页准备随物理坐标布局切换。

每组在 L=40、80、176 检查 1,000 条查询的返回 ID、精确距离和遍历／精排候选轨迹逐字节一致。这个 α=1 同图对照使用输出一致性控制质量，召回字段为 -1，未加载原混合权重和 GT。真正单空间图的建图、独立 GT、图拓扑和搜索深度差异需要另测。

正式阶段四搜索线程、NUMA 3，按 ABC／BCA／CAB 顺序各执行一轮。每个 L 清空共享缓存，预热完整查询集，再计时至少 10 秒。查询期记录 PQ gather/lookup、准备阶段、精排精确计算、拓扑与坐标后端提交及等待，并保留 QPS 和 p95。PQ lookup 计时包含 gather，candidate CPU 已包含这段时间。软件提交页数与物理设备计数分别记录。

使用上批逐秒磁盘监控与计时前负载门槛，保留原始采样和被标记的尝试。远端队列 `hvs-second-space-cost-20261110`，路径 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/second-space-cost-20261110`。检查 `status.json`、`analysis/parity.json`、`analysis/progress.json` 和 `RESULTS.md`。
