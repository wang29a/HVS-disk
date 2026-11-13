# CC3M 原生四方案：四线程 95% Recall 补测

本任务补齐原生 Baseline / Full / Merge / No-alpha 在 CC3M 上的 95% Recall@10，固定四个搜索线程、三次重复、K=10、beam=4。原曲线最高 L=400，对应召回为 93.83% / 93.82% / 93.83% / 94.69%，不能由此外推 95% 的性能。新布局执行器已覆盖 CC3M 的 95%，但执行路径与这四个原生方法不同，其绝对性能不填入本表。

远端根目录为 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/cc3m-highrecall-t4-20261104`，tmux 为 `hvs-cc3m-recall95-t4-20261104`。11 月 6 日恢复时，预算实验已完成，原生 I/O 实验处于用户暂停状态。CC3M 取得两条前序队列的排他锁后开始搜索。I/O 处于运行状态或任一锁被占用时，CC3M 继续等待。

补测复用 d2a39e2 的原生静态库和已经编译的 I/O 分项驱动，不改 C++、算法、索引、重排、PQ、入口点、mixed-alpha、GT 或精排策略。精确命令及输入身份在 analysis/input-cases.json，驱动编译与二进制哈希在 analysis/native-driver-build.json。启动搜索前再次检查库／驱动／旧参照／查询／权重／GT 的 SHA256，以及索引文件大小和 mtime。配置与控制脚本的哈希固定在 analysis/identity.json。

筛选先重测 L=400，与原始 Recall、候选计数、拓扑及精排计数核对，再依次尝试 L=600/800/1200/1600，达到 95% 即停止向上筛选。首次跨过 95% 的区间二分补点，直到相邻 L 相差不超过 100。正式测量采用本批重新测量的 L=400 和 95% 两侧点，不能把十月的低 L 时间与新测高 L 时间拼成一个匹配结果。若 L≤1600 仍不覆盖，保留曲线并明确报告未覆盖。L 上限还低于原生 2048-sector scratch 的容量；本任务不提高原生缓冲配置。

每种方案先做观测开／关的一轮输出检查，IDs 和 distances 必须逐字节相同，Recall 与原生计数必须一致；正式计数也与筛选结果核对。各正式点至少测十秒，沿用原生驱动每个 L 的 256-query pilot，1,000 条独立查询循环回放；循环不增加独立样本。三次重复轮换方法顺序及观测开／关次序，共 12 条观测开启曲线、12 条关闭观测控制运行，以及四组输出一致性检查。正式 L 个数由筛选决定。

所有搜索串行，OpenMP 上限四线程，BLAS 单线程。CPU 和内存均绑定到 NUMA 3，允许 CPU 集为 `72-95,168-191`。运行优先级为 nice=10 和 ionice=7。iostat、mpstat、pidstat 与搜索进程同时采样。

指标同时保留 Recall–QPS、Recall–内部延迟、外部查询服务时间、单文件 reader 时间、拓扑提交／等待、原始向量读取、后端请求数及字节。精确距离计算排除在向量 I/O 外；Baseline 的 reader 后端子调用不再次加入父调用时间。后端提交是软件边界统计，不等于物理 SSD 读取。原始 accuracy_compute 并未完整覆盖 Baseline，不能把其零值当作没有计算。

完成后 `status.json` 才设为 complete，`analysis/completeness.json` 必须验证 24 条正式运行及四组输出检查，保留覆盖缺口和观测开销异常。data/ 导出每次重复、均值／标准差、95% 覆盖及开／关控制；RESULTS.md 给出新批次比较。scripts/summarize.py 使用标准库导出数值，在 matplotlib 可用时额外生成 Recall–Latency / Recall–QPS 图。scripts/test_controller.py 是合成测试，验证前序状态／锁、精确命中／插值／未覆盖及汇总，不代表已完成真实数据集测试。

本地 `scripts/sync_results.py --watch` 只读轮询远端状态。CC3M 完成后独立下载，核验并追加到 `../2026-11-04-local-results/README.md`、supplements/、图索引和 ZIP。前序 I/O 队列完成时才导入其分项结果。若本机退出或重启，可重新运行同一命令。

检查与恢复命令：

```bash
# node3
cat /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/cc3m-highrecall-t4-20261104/status.json
tmux capture-pane -pt hvs-cc3m-recall95-t4-20261104 -S -30

# 本地：可重复执行；同步锁防止重复导入
/Users/wangzheng/.cache/hvs-experiment-analysis-venv/bin/python \
  /Users/wangzheng/Public/disk-hvs/experiment_results/cc3m-highrecall-t4-20261104/scripts/sync_results.py --watch
```
