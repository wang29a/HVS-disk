# Laion 补测计划（2026-10-27）

已于2026-10-27 13:55（Asia/Shanghai）启动node3 tmux队列 `hvs-laion-20261027`。当前阶段以实验目录的 `current-stage.txt` 为准，完整退出码与完成时间由队列写入；启动不代表实验已完成。运行入口与查看方式见 `RUN.md`。

目标是补齐此前 OpenImages、CC3M、HowTo100M 的四方案比较，得到 Laion 的 Recall–QPS、Recall–I/O、索引空间与 alpha metadata 特征。本轮优先复用已经存在的图、精确向量和 PQ，在验收后重新生成具有明确版本和参数的 Starling 布局。性能数据统一从 nvme3 读取；nvme2 上的 base 仅供校验，query/alpha/ground truth 也从现有路径加载。

本轮代码是 GitHub main 的 `d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e`，node3 干净 checkout 在 `/mnt/nvme3/wz/hvs-disk-build-20261027/source`。解耦构建在同级 `build-separated/`，单文件 baseline 构建在 `build-single-baseline/`。GCC 11.4、Release、Linux AIO、AVX-512、OpenBLAS 及编译 flags 都已记录在 `PREPARATION.md`。外部 Starling 使用 `/mnt/nvme3/wz/starling-wang29a`，root commit 为 `7437c4848a83e7bf62558c5eab1eb999feed8537`；使用前保存 root/submodule diff 和二进制 hash，保留 node3 的既有兼容修改。

## 已核对的输入与旧产物

数据目录为 `/mnt/nvme2/wz/disk/dataset/Laion/`。两份 base 均为 10,642,155×512，query 均为 1,000×512；alpha 为 `base_gt_round_00/query_alpha.fvecs`（1,000×1），ground truth 为同目录 `top10_results.ivecs`（1,000×10）。文件虽然使用 vecs 后缀，实际都是全局 8 字节头的项目 bin 格式，已核对文件长度。

索引源目录记为 `SRC=/mnt/nvme3/wz/Index/Laion-shard`。以下是实际读取的 header，不能仅凭根目录的通用文件名选择 Full：

| 用途 | metadata / topology 或单文件 | 已核对信息 |
|---|---|---|
| Full 原图 | `_disk.index.bp` + `disk_index_graph.bp` | meta 8192B；R=40，A=7，724B/node，11 nodes/page，22 entrypoints |
| 旧 Full 重排入口 | `alpha/` | topology 为 `disk_index_graph.relayout`；meta 只有4096B |
| 根目录通用入口 | `_disk.index` + `disk_index_graph` | 实际为 A=2 的 Merge，不能用作 Full 转换源 |
| Merge 原格式 | `_disk.index.ma` + `disk_index_graph.ma` | meta 8192B；A=2，324B/node，25 nodes/page |
| No-alpha 原格式 | `_disk.index.no_alpha.bp` + `disk_index_graph.no_alpha.bp` | meta 8192B；A=0，164B/node，49 nodes/page |
| 单文件 baseline | `single_index` | N=10,642,155，512+512，R=40，A=7，nnodes_per_sector=0，22 entrypoints；87,180,541,952B |
| 内存图 | `mem.index` | N=10,642,155，R=40，A=7 |
| 两空间 PQ | `emb_pq_compressed.bin`、`loc_pq_compressed.bin` | 均为 N×64，即每空间64 chunks |

旧 baseline 的 header 没有显示 alpha 被截断，但这不是完整正确性证明。只有全量验收通过后才纳入正式搜索。旧的 Starling 入口可以作为冒烟参考；正式队列重新运行三种布局各自的8轮 partition，以记录本次确切的工具版本和参数。

## 执行顺序与验收门槛

第一步建立独立目录 `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/laion/`，分为 `full/`、`merge/`、`no-alpha/`、`baseline/`、`logs/`、`analysis/`。记录输入文件 realpath、长度、头信息、metadata/PQ/工具 hashes、环境变量、NUMA 和当前磁盘负载。重新检查 alpha 的数值范围与 query 对齐，抽样计算双空间精确距离检查 ground truth；固定距离定义为精排的 `alpha*sqrt(dist1)+(1-alpha)*sqrt(dist2)`。

第二步验收源图和 baseline。完整检查 Full 的节点数、degree 上界、neighbor ID、range bounds 与页尺寸。将单文件 baseline 的全部节点邻居及完整 alpha slots 与同一 Full 源图核对，使用 `mem.index` 作为额外参照；还要验证坐标与两份 base 的行顺序、数值一致。若旧 baseline 不通过，使用同一份已验收图和坐标重新打包单文件，并使用当前修正 writer；不能另起一次随机图构建来替代该方案，否则不再是同图布局比较。重新打包工具若不是当前仓库现有 target，应作为独立实验辅助工具记录源码、编译命令与 hash。

第三步从已验收的 `disk_index_graph.bp`、`_disk.index.bp` 重新生成 Merge（greedy method=2）和 No-alpha，再对三方案分别执行 Starling。partition 参数为 `-T 32 -L 8 --ep_size 22`，输入 meta 与 topology 必须属于同一格式。每种布局生成自己的 partition、relayout topology 与 aligned partition；retain 原始/转换后8192B metadata，不能安装旧 Starling 生成的4096B meta。完整验证 partition 双射、每页容量、节点覆盖和逐节点 relayout record，要求所有错误计数为零；三种格式的 degree/neighbor 顺序必须一致，Merge 不能损失原 Full range 的覆盖。统一搜索目录可共享 nvme3 上的 coord 和 PQ，其他文件均绑定各自布局。

第四步用全部1,000条匹配查询做四方案冒烟，先看 `L=40,100` 的 Recall、比较次数和 I/O，再进入性能队列。检查 No-alpha 在同 L 下确实展开邻居，Full/Merge 的过滤在工作；Recall 应随 L 总体提升，Full/Merge 的明显差异要先解释。当前全量复测尚未运行，不预设四方案之间的性能胜负。

第五步执行正式32线程四方案实验，作为与旧三数据集对应的主结果。每个方案一次运行扫描全部L，完成三次重复；方案顺序轮换：第一轮 baseline→full→merge→no-alpha，第二轮 full→merge→no-alpha→baseline，第三轮 merge→no-alpha→baseline→full。所有长任务在一个tmux队列中串行执行，保留每次命令、起止时间、退出码和 `/usr/bin/time -v` 输出。

第六步在同样输入和布局上补64线程，独立生成同样的12份日志。NUMA仍使用0、1以保持与32线程相同的CPU/内存节点集合。两个NUMA节点合计48物理核、96逻辑CPU，因此64线程是包含SMT的并发实验，不应描述成64个独立物理核。若希望对比32/64物理核，需要另外设计并注明NUMA变化。本轮Laion的64线程结果也不能直接与其他数据集旧32线程结果作横向并发比较。

## 统一搜索协议

| 参数 | 设置 |
|---|---|
| 数据 / query / GT | 同一份 Laion，1,000 queries，Recall@10，L2 |
| 线程 | 主结果32；补充64 |
| NUMA | `numactl --cpunodebind=0,1 --interleave=0,1` |
| OpenMP | `OMP_NUM_THREADS=T OMP_PROC_BIND=close OMP_PLACES=cores` |
| beam / I/O width | 4 |
| L sweep | 20、40、60、80、100、120、160、200、300、400 |
| Full / Merge / No-alpha | `build-separated/tests/search_disk_index`；mode=6，mem_L=0，strategy=17 |
| baseline | `build-single-baseline/tests/search_disk_index`；mode=0，mem_L=0，strategy=0 |
| PQ | 每空间64 chunks，共享源图配套的两套 pivots/compressed 文件 |
| Starling | 三布局独立，8轮，32线程，22 entrypoints |
| repeats / queue | 3次，四方案轮换，串行 |
| 磁盘 | 实际 topology/coord/single_index 均位于nvme3 |
| warm-up / cache | 记录程序现有warm-up、query缓存与共享缓存的实际行为；不执行全局drop_caches |
| 其他需记录项 | BLAS线程环境、rerank截断公式及实际值、cache预算、输入/输出身份和硬件负载 |

当前主配置 `strategy=17` 未启用truncate、double-PQ或triple-PQ位，因此 reader 保留默认 `trunc_len=1e9`，最终精排候选数由L和当前候选集大小限制，相当于对保留候选集完整精排；不能误记为24。只有启用相应位时reader才将该值设为24。本轮保持17以对齐旧实验，后续若研究截断，应作为独立消融。原日志表头写着 `P99 Lat`，但源码实际计算 `0.999` 分位，应按P99.9归档；1,000条查询的该指标接近最大值，不能据此作稳健尾延迟结论。若后续补P95/P99或扩充query，需要单列测量版本与数据，不能把指标名直接改成P99。

先完成32线程12份日志、120个L点，再完成64线程12份日志、120个L点。总计24份正式日志、240个测量点，每个线程配置均为三次重复。不设未经实测的QPS提升目标；从冒烟/首轮耗时再估算剩余队列时间。

## 汇总与论文需要的数据

按32/64线程分别输出三次均值和波动、实测 Recall–QPS/Recall–I/O 曲线，以及Recall=80/85/90/92/95的插值操作点。仅在实测曲线覆盖目标时插值，不外推；同时保留原始L点。baseline端到端QPS可参与比较，其I/O字段与解耦方案的topology/rerank计数需分开解释，不能直接拼接成同一定义的页面节省率。

补充Laion的alpha characterization：固定slots利用率、每边range数量分布、query-alpha加权过滤率、Merge新增激活与覆盖保留、三种topology的绝对空间和转换/重排开销。该步骤复用之前characterization的指标定义；对ranges覆盖关系的验证应与统计加权区分。本轮复用现有图和PQ，因此只报告转换、重排、验证和搜索时间，不把它们称为完整图构建时间。Gated A0/A2尚无正式搜索reader，不混入这次四方案主比较。

## 空间预算

nvme3现有约419GiB可用。Full原图、coord、PQ和通过验收的baseline均可直接链接复用。新增Merge/No-alpha原拓扑合计约4.9GiB，三种新relayout合计约12.3GiB，加partition、日志及统计后预留25GiB即可覆盖主路径。若需要重新打包baseline，再预留约81.2GiB，总预算以120GiB封顶并在运行前复核。所有正式索引保留在nvme3；nvme2保存归档和恢复材料。

完成后归档实验计划、manifest、完整原始日志、格式/重排/基线校验报告、characterization JSON、均值与波动汇总、matched-recall表格和图。新结果独立保存，保留此前三数据集原始记录。
