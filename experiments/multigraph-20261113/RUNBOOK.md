# 运行入口

先读冻结的 README。`FREEZE.json` 保存第一次协议提交的字节数和 SHA256；队列检查 README 的这个前缀未变，允许末尾追加说明。

在 node3 的本实验目录执行 `python3 scripts/compile.py`，然后 `python3 scripts/validate.py tiny`。构图器、PQ 和数据依赖相邻的已验证实验目录；不是从任意 Linux 裸环境直接运行的通用部署脚本。编译日志及二进制指纹写入 `runtime/validation/build.json`。小图不通过不得入性能队列。

`bash scripts/enqueue.sh` 创建后台 tmux `hvs-multigraph-20261113`。它先等已排队的旧图复核、共同候选池排查和独立核组复测结束，然后竞争同一个全局队列锁。它不会修改或重启这些旧任务。状态在 `runtime/status.json`，完整日志在 `runtime/coordinator.log`，每个子进程的命令和日志分别在 `runtime/commands/` 和 `runtime/logs/`。

队列依次准备共用坐标与子集PQ编码，先用现有中心图检查旧引擎兼容性，再补α=0.3／0.7 GT及16条预定查询的穷举双索引覆盖率，构建两张原生单空间图和四张固定权重图，完成真实数据正确性门槛，最后做调参和正式三轮计时。所有原始图只读，新的格式文件在 `runtime/packed/`。每种工作量规则分别生成冻结前沿，不看报告延迟选配置。

正确性失败状态为 `stopped`，没有自动改参数或自动扩大数据集的后备路径。外部资源干扰有有限重试，连续三点拒收则停止；进程退出后保留日志和已经完成的图。若要修复代码后重启，必须先在README末尾记录原因，核对旧 `runtime/plan.json` 与输入身份，不能删除身份门槛强行混合两批结果。

完成状态为 `awaiting_user_review`。结果在 `runtime/report/RESULTS.md`、`summary.json`、`matched.csv`、`graphs.csv`、`storage.csv`、`raw-p95.csv`。`ssd-*-actual.svg/pdf/png` 是实际图像贡献比例主图，名义α图另存。不同工作量选择规则和驻留条件分别画图。任何95%召回未覆盖的点保持空缺，不能补外推值。

源码归档包含冻结协议及追加修订、隔离执行器、格式转换、三层诊断、队列和报告代码，以及小型构建／正确性验证记录。不要提交 `runtime/` 下的数据、GT、大图、二进制、日志或私有绘图依赖。

## 2026-11-14 三层门槛及恢复入口

用户批准用精确穷举A／PQ穷举B／PQ图搜索C替换旧的精确上界门槛。新入口是 `bash scripts/enqueue_three_layers.sh`，tmux为 `hvs-three-layer-20261114`。它先做现有16条查询上的三层检查，通过后复用原GT、坐标、PQ、X／Y图及统一／中心图，继续剩余四张专用图和原性能队列；不重新建已有图，也不修改L、k。所有检查阈值已在README末尾事前追加。

三层状态和结果在 `runtime/three-layer-20261114/status.json`、`summary.json`、`RESULTS.md`。原 `runtime/plan.json` 和旧失败文件保留；新计划在 `runtime/plans/three-layer-20261114.json`，记录父计划指纹。后续构图／计时的当前子任务仍以根 `runtime/status.json` 为准，不能把三层PASS误读为正式性能实验结束。
