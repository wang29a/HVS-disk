# 原生四线程 I/O 时间比较（2026-11-04）

本补充把 Baseline 的单文件读取，与 Full / Merge / No-alpha 的拓扑读取、原始向量读取放在同一 reader 边界计时。比较固定四线程，在每次重复内匹配 90% / 95% Recall@10，再汇总三次重复；不按相同 L 直接比较，不重算 GT。

已有总时间见 [LEGACY-TIMES.md](LEGACY-TIMES.md) 和 [四线程旧计时曲线](figures/legacy-recall-io-time-t4.png)。它们从 48 条原生运行、480 个 L 点恢复，未改动旧日志。旧驱动只导出 io_us，没有导出 io_us1 / io_us2。Baseline 的旧 io_us 围绕同步 reader 调用；解耦拓扑只计 poll_wait，漏掉 send_io，坐标部分计阻塞 read_fd。因此旧表可描述原有观测，但不能用于统一计时边界下的阶段归因。旧源码计时位置与哈希保存在 [timer-source-audit.json](analysis/timer-source-audit.json)。独立布局执行器虽已有分项时间，也不能拼接进原生四方案的对照表。

补测已放入 node3 的 tmux 会话 `hvs-io-time-20261104`。本地 [status.json](remote-snapshot/status.json) 是下载时快照，状态为 `waiting_predecessor`，原生分项搜索尚未执行；必须等待 `budget-reorder-20261103` 状态为 complete 并取得其排他队列锁，才会开始读取实验数据。等待期间不运行数据集搜索。远端目录是 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/io-time-20261104`。不能把 reader 合成测试通过说成原生搜索已通过。

计时模型为：

```text
Baseline:
  reader_total = coupled_reader
  coupled_backend 是 coupled_reader 的子项，不再次相加

Full / Merge / No-alpha:
  reader_total = topology_reader + coordinate_reader
  topology_reader = send_io 调用时间 + poll_wait 调用时间
  coordinate_reader = read_fd 阻塞调用时间

external_service = reader_total + other_service
```

这些数值是每查询、非重叠 reader 调用的墙钟时间，单位微秒；图表换算成毫秒。它们衡量请求提交与等待在查询服务时间中占用多久，不是各 SSD 请求设备延迟的简单总和。异步请求批内并发，不能用“页数 × 平均页延迟”代替实际阶段计时。Baseline 的父 reader 时间还包含其内部缓冲／缓存处理，另保留 `reader_coupled_backend_us` 观察真正进入后端读取的子区间。

`reader_coordinate_us` 只含读取原始精确向量的调用，排除精确距离计算与排序。旧 `accuracy_compute_us` 作为 `legacy_exact_distance_us` 保留在逐次原始点中；它并不完整覆盖 Baseline 的精确距离路径，Baseline 的零值不能解释为没有计算。`other_service` 是外部服务时间减去 reader 时间，包含计算、维护数据结构、准备及其他等待，不能直接称为纯 CPU 时间。新表的 I/O 时间占比统一除以 `external_service_us`，不与旧内部 total_us 混用。

同时导出请求数和字节。Baseline 单列 `coupled_requested`（reader 缓冲判定前）及 `coupled_backend_requests`（后端执行前）；解耦方案分别导出 topology / coordinate 后端请求与字节。后端请求是软件提交边界统计，仍不等于物理设备读取或 NAND 操作。iostat 是按时间对齐的设备背景诊断，设备上其他进程的访问不能归到本实验。

代码链接原实验的 d2a39e2 原生静态库，未改搜索算法、索引、PQ、入口点或距离公式。两个测量驱动和 reader 测试已在 node3 编译成功，完整命令及源码／库／二进制 SHA256 见 [build.json](remote-snapshot/build.json)。[test-reader.log](remote-snapshot/test-reader.log) 记录已通过的 24 KiB 合成测试：缓存命中／未命中、父子时间不重复相加、异步提交／等待、坐标读取、四线程观测隔离和关闭观测。该测试只读自建 nvme2 小文件，使用 CPU 48–51，避开现有主队列的 CPU 及 nvme3 数据盘。

正式补测先对四数据集的十六个方法组合分别运行观测开／关检查，比较输出 IDs / distances 的逐字节一致性，并与原归档在相同 L 下的 Recall 和原生计数核对。任一检查不通过即停止，不继续生成性能结论。正式测量每个点至少十秒，1,000 条已有查询循环回放，沿用原驱动 256-query pilot 预热；三次重复轮换方法与观测开／关顺序。关闭新增观测的控制点用于识别插桩开销及环境异常，异常点保留并标记，不自动删去。

配置复用原生 K=10、beam=4、已有 PQ、索引及 GT；OpenImages / CC3M / HowTo100M 为原 mixed-alpha，Laion 为 alpha=0.5。仅选原四线程曲线中包围 90% / 95% 的 L，CC3M 的 95% 原曲线未覆盖，不外推。共计划 96 条计时运行、216 个计时 L 点，另有输出一致性检查。索引文件大小／mtime、查询与 GT 哈希、原库及驱动哈希都会再次核验。

队列串行运行，每个搜索进程最多四个 OpenMP 线程，BLAS 单线程，CPU 允许集为 `0-47,96-111`，内存在 NUMA 0、1 间交错；允许集不是实际使用 64 个搜索线程。进程使用 nice 10 / ionice 7，等待前序队列结束且持有其锁期间不与该队列并行。完整设置见 [protocol.json](remote-snapshot/protocol.json) 和 [run.sh](scripts/run.sh)。

可在 node3 检查状态及日志：

```bash
cat /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/io-time-20261104/status.json
tmux capture-pane -pt hvs-io-time-20261104 -S -40
tail -n 30 /mnt/nvme3/wz/hvs-disk-build-20261027/experiments/io-time-20261104/logs/queue.log
```

完成后应检查全局 status 为 complete、十六组输出一致性检查和 96 条计时运行齐全，再使用 `data/native-phase-matched-recall.csv`、`data/observer-controls.csv` 及 `PHASE-RESULTS.md`。原始 JSON 同时保存新 reader 时间与旧 QueryStats 时间，图表为 Baseline 单文件时间、解耦拓扑时间、向量时间及剩余服务时间的堆叠图。绘图脚本在 matplotlib 可用时输出 PNG / PDF / SVG；远端缺绘图库时可下载完整目录后在本地运行 scripts/analyze_phase.py，不需重跑搜索。

11 月 4 日 17:16，在首条数据集搜索之前修正了 Python 3.6 / LC_ALL=C 环境下中文 Markdown 的导出编码，明确使用 UTF-8。仅重启等待中的控制进程，搜索驱动、库、计时及参数不变。旧／新控制脚本哈希、零搜索结果确认和合成导出测试分别保存在 remote-snapshot/export-encoding-fix.json 与 remote-snapshot/export-encoding-test.log，防止正式测量完成后卡在汇总导出。
