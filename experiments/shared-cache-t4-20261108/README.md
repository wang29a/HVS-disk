# 四数据集共享拓扑缓存实验

在 OpenImages、CC3M、HowTo100M 和 Laion 上，固定已重排的 Merge 索引、PQ、查询、alpha、GT、查询内页面复用和精排策略，仅切换跨查询的共享拓扑页缓存。缓存开启时使用 256 MiB 页面数据预算，容纳 32,768 个 8 KiB 页面。前三个数据集使用原有混合 alpha 查询。Laion 有 α=0.1、0.3、0.5、0.7、0.9 的 GT，本批选择 α=0.5，与此前的短测对齐。

使用 T4、beam 4、K10、mode 6、strategy 17、mem_L=0。CPU 和内存绑定 NUMA 3，BLAS 单线程。队列在 node3 的 tmux 会话 `hvs-shared-cache-t4-20261108` 中串行执行，沿用现有实验共用锁。索引及查询参数来自同一代码版本的四数据集归档，启动时检查输入哈希并记录索引文件身份。

初始 L 网格为 10、20、40、60、80、100、120、160、200、300、400、500、600、800、1200、1600。高 L 单独筛选。若某点触及单查询 2048 个拓扑页的 scratch 容量，记录拒绝原因，并在最高安全点与第一个拒绝点之间补一个点。通过筛选的每个 L 都校验缓存开关的返回 ID、距离和逻辑工作量。计时前清空共享缓存并预热 1,000 条查询，随后循环回放同一批查询至少 10 秒。每组做三次成对重复，轮换开关顺序。

远端根目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/shared-cache-t4-20261108`。`status.json` 和 `analysis/progress.json` 记录进度，`analysis/*-parity.json` 记录真实查询校验与实际召回范围，`results/` 保留逐次原始结果。每完成一对曲线，队列更新 `RESULTS.md`、`analysis/curves.csv` 和 `analysis/summary.json`。报告记录 Recall@10、QPS、延迟、共享命中、拓扑／坐标后端提交页数及 reader 时间。设备级数据在 `logs/iostat.log`，按实际时间与测量窗口对齐。等召回结果在各重复的实测区间内插值。
