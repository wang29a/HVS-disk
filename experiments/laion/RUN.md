# Laion 队列运行记录

node3 实验目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/laion/`，tmux session 为 `hvs-laion-20261027`，2026-10-27 13:55（Asia/Shanghai）启动。没有修改仓库C++源码；runner和独立校验器位于实验目录的 `scripts/`。

```bash
ssh node3
tmux attach -t hvs-laion-20261027
```

只读查看当前阶段和日志：

```bash
EXP=/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/laion
cat "$EXP/current-stage.txt"
tail -30 "$EXP/logs/queue.log"
tail -10 "$EXP/logs/01-source-validation.log"
```

顺序为 provenance、source-validation、ground-truth-validation、conversion、三个独立Starling与relayout校验、四方案smoke、32线程12份正式日志、64线程12份正式日志、alpha characterization、summarize。任务串行执行；每个实质校验失败即退出，不继续生成正式搜索结果。

退出时写入 `queue.exit-code` 和 `queue.finished-at`。仅当退出码为0且日志包含 `LAION_QUEUE_COMPLETE` 时，才表示全部阶段完成。smoke、source、query、conversion、relayout的校验报告在 `analysis/`；正式日志在 `logs/threads-32/` 和 `logs/threads-64/`。最终汇总由 `summarize_laion.py` 在检查24份日志、240个L点齐全且搜索正常退出后生成。

`analysis/manifest.json` 记录15个输入文件的路径、realpath、长度、时间、inode、header，较小输入还包含完整SHA-256；大输入使用身份信息加全量内容验收，不伪称已经计算完整SHA-256。工具二进制与runner/validator的hash在 `analysis/tool-sha256.txt`。Starling root和submodule状态、diff单独保存。

`test_validators.py` 已在node3通过合成输入测试：跨页及末尾部分页、零/最大邻居数、alpha endpoints、三种relayout，以及故意损坏坐标、neighbor、partition、range coverage均被正确识别。真实Laion另有257节点启动抽查，baseline完整拓扑与Full原图、坐标与base均一致；仍须以后台全量验收报告为准。

为避免旧队列被误覆盖，runner拒绝在已有 `queue.started-at` 时直接启动第二份。故障后先读取失败阶段和日志，再决定采用新输出目录重跑或显式恢复；不要直接清掉开始标记、覆盖旧输出。使用的源meta为8192B `_disk.index.bp`，源topology为Full `disk_index_graph.bp`，不会修改旧索引入口。
