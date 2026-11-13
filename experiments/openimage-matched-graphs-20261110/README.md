# OpenImages 单／双空间图的等召回对照

三组都针对 X 空间的 L2 top-10，使用同一个搜索执行器、同一批 1,000 条查询、同一个 X PQ 码本与 64-byte 编码、同一物理 ID 顺序、beam=4、K=10、256 MiB 共享拓扑页缓存和原查询内页复用。独立单空间 Vamana 只加载一组 PQ 和一组坐标。双空间组固定 α=1、读取两组坐标并计算两组距离。双空间图的 X-only 锚点保留原拓扑和 α 元数据，只算只存 X。单空间与 X-only 锚点直接共享同一份物理坐标文件，拓扑保留各自的边和入口点。物理 ID 顺序固定为双空间 M=2 的归档 Starling 顺序。

单空间图来自已验证的 Vamana 磁盘索引，转换前后逐节点核对坐标、邻居顺序和入口。读取已有 X 空间穷举 GT 的 ID 矩阵。计时前从实际返回 ID 独立重算召回，从原向量核对返回距离，并确认双空间两组的返回 ID、距离完全一致。

各图先筛出自己的 L 范围，按 90%、95%、98% Recall@10 的可达区间计时。QPS、均值和 I/O 在每次重复的实测区间内插值。p95 使用最接近的实测召回，并标出该召回值。NUMA 3 四线程，三次重复，旋转方法顺序，预热后每点计时至少 10 秒，采用原批逐秒设备筛查和负载门槛。报告不外推未覆盖的工作点。

node3 的会话为 `hvs-openimage-matched-graphs-20261110`，实验目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/openimage-matched-graphs-20261110`。查看 `status.json`、`analysis/validation.json`、`analysis/progress.json` 和 `RESULTS.md`。该对照统一检索任务到 X 端点，原有 mixed-alpha 检索是另一种任务。
