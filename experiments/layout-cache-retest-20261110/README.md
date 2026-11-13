# 约 95% Recall@10 的安静窗口复测

本批复用 11 月 9 日的布局 harness 二进制、固定 M=2 Starling 节点顺序、查询与 GT。OpenImages 在 L=176、95.12% Recall@10 复测 TXY／T｜XY × 共享缓存开／关，四格重新核对返回 ID 和距离的原始哈希。HowTo100M 固定 T｜XY、共享缓存开启，抽测 M=2 的 L=314（95.11%）和 M=3 的 L=316（95.10%），逐点核对召回和逻辑搜索计数。每格三次重复，四搜索线程固定 CPU 74、75、76、77，内存 NUMA 3。旧批的 CPU 80 被其他进程占满，本批换到同节点的 CPU 76。共享缓存使用原有的 256 MiB 预算，预热和计时沿用原批。

启动前要求连续两个三秒窗口满足 nvme3 利用率≤25%、写入≤1 MiB/s、四核各 SMT 线程忙碌率≤15%。计时期间每秒采样 nvme3 读写以及本进程的 read_bytes。每个测点只使用完全落入计时区间的采样，至少五个有效样本，最高写入≤5 MiB/s，设备和进程读取速率的平均差值≤100 MiB/s。该差值是检测额外读负载的近似量，受计数更新时间影响。标记的测点保留并限次重试，报告的安静列只展示通过该筛选的结果。

node3 队列为 `hvs-layout-cache-retest-20261110`，目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/layout-cache-retest-20261110`。结果见 `RESULTS.md`、`analysis/retest-results.json`，设备与本进程采样见 `analysis/*-telemetry.json`。
