# 四数据集原生 I/O 计时 · NUMA 3

本批恢复 OpenImages、CC3M、HowTo100M 和 Laion 的原生四方案计时。使用四个搜索线程、三次重复和已有查询与 GT。按原协议测量 216 个计时点，并检查观测开关前后的输出一致性。

CPU 和内存均绑定 NUMA 3。搜索串行执行，BLAS 单线程，优先级为 nice=10、ionice=7。沿用已经校验的计时二进制、索引和搜索参数。

远端实验目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/io-time-numa3-20261108`，tmux 会话为 `hvs-io-time-numa3-20261108`。原批次 `io-time-20261104` 的 59 条正式运行保存在原目录。新批次单独汇总。
