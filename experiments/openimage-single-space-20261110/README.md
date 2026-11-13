# OpenImages 单空间对照资源

使用 OpenImages 的 456,699 个图像向量和 1,000 条图像查询，每向量 768 维 float32。逐对象校验 X 的坐标和 ID 顺序与原双空间索引一致。只用 X 建静态 Vamana 图，L2、R=40、Lbuild=100，构建期关闭旧版内联重排。重用原双空间索引的 X 侧 64-byte PQ 编码和码本，以便后续控制 PQ 训练差异。单空间真值用全部 X 数据穷举计算 top-10，保存 ID、平方 L2 和对应标签。验证全部返回距离，并用 float64 独立穷举抽查八条查询。对单空间磁盘索引逐节点验证坐标、邻居、度数和入口可达性。

准备任务在 node3 的 NUMA 2，绑定 CPU 48、49 与内存节点 2，OpenMP 两线程。建图时 BLAS 单线程，穷举 GT 阶段 BLAS 最多两线程，统一 nice 15、ionice best-effort 7。阶段启动前检查 nvme3 利用率≤10% 且写入≤1 MiB/s，保留磁盘遥测。NUMA 3 上的第二空间成本队列在共用锁后排队，自动复用已校验的布局和轨迹。

远端目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/openimage-single-space-20261110`，tmux 会话为 `hvs-openimage-single-space-20261110`。产物位于 `index/`，构建和验证清单位于 `analysis/`。后续端到端性能比较需要为单空间图接入同口径的搜索驱动。现有交接仓库的双空间搜索 CLI 需要两组 PQ，不能直接载入这些单空间文件。若双空间查询固定 α=1，可用同一 X GT；原 mixed-alpha 双空间结果继续用其匹配 GT。
