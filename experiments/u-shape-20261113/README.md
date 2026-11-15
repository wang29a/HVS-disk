# Laion U 形诊断与构图复核

这里保存 2M Laion 统一图／专用图诊断，以及由此发现的构图边界问题复核。运行代码使用 node3 上的隔离构图库；不是从裸仓库直接启动整套实验的入口。

2026-11-14 归档状态：OpenImages 已完成重建、逐边比较和两批关键SSD复测；独立核组取得70个有效计时点，但仍有5点pending，不能标成全部完成。中心U1／S未达到完整图“接近”门槛；后续1,000个共同候选池排查没有发现剪枝谓词不一致，排序、度数预算和量化会改变边集。强度扫描仍只冻结协议、尚未启动。新的三层候选诊断和Laion多图磁盘对照已在[相邻实验目录](../multigraph-20261113/RUNBOOK.md)完成，不能把两个队列的完成状态混用。

当前隔离构图实现是 `<=`，与DEG §4.3的严格 `<` 需要分开说明，不能称完全复现论文边界约定。修正仍未合入根生产源码。

## 当前应该读什么

| 内容 | 文档 |
|---|---|
| 本轮优先级、判读规则、SSD 权重分桶、强度扫描协议 | [当前协议](NEXT-CHECKS-PROTOCOL.md) |
| 比较符、零系数、区间合并、距离开方的来源证据 | [语义与来源核对](SEMANTICS-PROVENANCE.md) |
| 正式计时受外部负载影响后的独立核组复测 | [核组补充](AFFINITY-AMENDMENT.md) |
| 修正统一图与中心专用图的上一轮结果 | [v2 结果](PRUNING-V2-RESULTS.md)、[v2 协议](PRUNING-V2-PROTOCOL.md) |
| 旧图结果哪些需要复核 | [跨实验有效性通知](../VALIDITY-2026-11-13.md) |

## 实时状态与目录

Node3 工作目录是 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/u-shape-20261113`。本轮状态在 `review-20261113/status.json`，SSD 状态在其 `ssd/status.json`。同候选池排查状态为 `review-20261113/explanation-status.json`；独立核组复测写入 `review-20261113/cpu-recheck/`，与原核组不混合。

下列会话名保留作运行身份记录，归档时这些会话已结束；不要据此重新启动整批历史任务。

tmux 分别为 `hvs-priority-review-20261113`、`hvs-centre-explanation-20261113`、`hvs-review-recheck-20261113`。它们共用全局队列锁；候选池排查和独立核组复测排队执行，不能因存在多个会话就认定多个性能任务同时运行。构图最多 8 线程，正式搜索 4 线程，内存绑定 NUMA3。外部干扰最多三次，失败保留 pending；不能把有状态文件当作有效计时已经完成。

根 `status.json` 只代表最初的数据准备。`stage2/` 是首轮专用构图出现问题的历史记录；`stage2-v2/` 是修正后的重建与中心门槛结果，现已结束。不要把它们的 complete 状态当成本轮复测结束，也不要同时重启所有旧脚本。

## 数据与 GT

2M Laion 输入在 `/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m`，两个空间各 512 维；`source_ids.bin` 将子集局部 ID 映射回全量 Laion。α=0.1/0.5/0.9 的 GT 是在整个 2M 子集上穷举得到，使用局部 ID，不是从原 GT 截取。

数据文件使用一次全局 `[rows:uint32][dims:uint32]` 头，后接行主序 float32；GT／ID 矩阵使用 uint32。文件虽带 `.fvecs`／`.ivecs` 名字，也不是经典逐行维度头格式。查询 α 是 Q×1 float32，融合目标是加权欧氏和，不额外归一化。

OpenImages 复用原 1,000 条 mixed-α 查询和 GT。本轮偏侧／居中采用原权重分桶，不伪造固定 α 的 GT，不与 Laion 的固定权重结果混称同一工作负载。

## 代码检查与运行边界

在仓库根目录可运行 `python3 experiments/check_sources.py`，以及 `python3 -B -m unittest discover -s experiments/u-shape-20261113/scripts -p 'test_*.py'`（需要 NumPy）。这些检查不启动 SSH 或实验队列。

构建与搜索依赖相邻的 `metric-consistent-20261111` 和 `laion-six-cell-20261111` 已验证库、数据与构建清单；真实 C++ 编译和功能核验在 Linux/node3 完成。新代码在实验目录生成隔离版 `hybrid_index.cpp`，尚未合入主 `src/hybrid_index.cpp`。本地保存代码、协议和小型验证记录，大图、逐查询结果及计时日志只写远端忽略目录，不加入 Git。

历史阶段一的抽样与 GT 代码是 `scripts/prepare_data.py`／`scripts/run.sh`；文献与原始冻结计划见 [文献](LITERATURE.md)、[协议 v1](PROTOCOL.md)。首轮运行记录保留在 [阶段二说明](STAGE2.md) 和 [首轮有效性复核](STATUS-2026-11-13.md)，以当前通知限定其结论范围。
