# 基础更新：阶段 0 评估

本轮完成现有代码的源码评估和环境检查，尚未实现双空间插入、删除或合并。G0 尚未通过：没有当前工作树的成功 Linux 构建、单空间插入测试和删除测试，而且分离式原地更新与查询之间缺少完整的共同读写协议。这里的结论是“尚不能保证并发正确”，不是已经实测出现 SSD 半页读或更新测试失败。

评估对象是 `source-identity.json` 中记录的当前工作树字节，包括用户已有的两处区间合并修复。附件原文保存在同目录 `README.md`，以 SHA256 标识，没有修改原文，也没有执行附件中要求的 Git 提交。附件作为设计依据；自动提交、修改正式实验和自动推进全部测量并不因出现在附件中而获得授权。本轮未修改已有 C++、索引、GT、构图默认配置或远端队列。

## 当前实验与验证边界

只读检查时，node3 的 `hvs-backlink-function-v2` 会话仍在运行，协调器 PID 28396，A2 构图进程 PID 48621，200 万节点、8 线程、CPU 72–79。队列状态为 `formal_AB_builds_and_DRAM`。这些是检查时的快照，不能推断之后仍处于 A2。全局 `budget-reorder-20261103/queue.lock` 非阻塞探测返回 75，表示锁被占用；没有启动远端编译或更新负载。

本地执行 `cmake -S . -B /private/tmp/hvs-update-g0-20261116-build -DBUILD_WITH_PQ=ON`，退出 1，原因是当前 CMake 不再接受项目声明的低于 3.5 版本兼容策略。仅在命令行追加 `-DCMAKE_POLICY_VERSION_MINIMUM=3.5` 后再次配置，退出 1，原因是 AppleClang 环境缺少 OpenMP。未修改 CMakeLists；这些是环境/配置障碍，不能作为 Linux C++ 编译失败的证据。单空间插入和删除测试均未运行，因此没有可报告的通过率、二进制哈希或实际编译选项。命令、退出状态及未验证项在 `validation.json`。

## 上游及编译入口

仓库 `README.md` 将代码描述为 FAST'26 OdinANN 与 PipeANN 的集成；`README-HANDOFF.md` 也明确说明继承 PipeANN/OdinANN，README 进一步注明基于 DiskANN/FreshDiskANN。`LICENSE` 的版权人为 Hao Guo，`NOTICE.txt` 保留 Microsoft 等第三方声明。这些是当前仓库自己的来源记录，并非本轮完成了逐函数上游提交归因。不能据此说分离式拓扑、缓存及删除锁修改全部是未经改动的上游实现。

`src/CMakeLists.txt` 通过 GLOB 将 `update/*.cpp` 纳入 `pipeann` 静态库。`tests/CMakeLists.txt` 定义了 `test_insert_search`、`mixed_workload`、`overall_performance` 三个目标。顶层当前默认开启 `DELTA_PRUNING`，而 `IN_PLACE_RECORD_UPDATE`、`BG_IO_THREAD`、`DIRECT_READ_CC` 是注释状态，也可能由 `ADDITIONAL_DEFINITIONS` 注入。不能将已有静态搜索实验二进制成功运行替代上述更新目标和测试的验收。

## ID、位置和文件扩展

`SSDIndex` 使用原子 `cur_id` 分配新 ID；`insert_in_place` 和 `insert_to_topo_in_place` 均执行 `cur_id++`，没有复用节点 ID。紧凑分离路径中的 `empty_locs_topo`、`empty_locs_coord` 回收的是存储位置，区别于文档要求的空闲 ID 列表。`id2loc_topo_`、`id2loc_coord_` 通过 libcuckoo 映射 ID 到位置，`page_layout_topo` 和坐标布局维护反向关系；位置再通过 `loc_sector_no_topo/coord` 及 `offset_to_loc_topo/coord` 转为页和页内偏移。

组合记录的 `direct_insert.cpp` 在未定义 `IN_PLACE_RECORD_UPDATE` 时，为新点及被修改邻居分配新位置，填好缓存后切换映射并回收旧位置；这条路径确有异地写入设计。分离式 `direct_to_topo_insert.cpp` 在无重排时原地修改旧邻居记录，新点使用空槽或追加位置；启用拓扑重排时还可分裂页面、移动记录并更新映射。这与本次要求的无 Starling 目标不同。

分离文件 reader 以 `O_RDWR | O_DIRECT` 打开 `disk_index_graph`、`disk_index_data`。插入使用位置计数递增和超出旧文件末尾的写入扩展文件，没有看到该入口先完整预分配并初始化新末页的协议。新目标页先读再修改的代码需要在复制的小索引上特别检查短读、部分末页和空文件边界；不能把组合格式测试自动视为分离格式扩展测试。

## 页写入和后台线程

`direct_insert.cpp` 默认先调用 `wbc_write` 将完整页镜像写入用户态 `v2::cache`，随后调用阻塞的 `reader->write` 等待 AIO 完成，再释放写页锁。启用 `BG_IO_THREAD` 时把写请求、缓冲及锁交给 `bg_io_thread`；后台线程写完才释放相关锁和引用。这不是仅提交 AIO 就认为落盘完成。

`direct_to_topo_insert.cpp` 则直接通过 `send_io` 加 `poll_wait` 完成新坐标和新拓扑页的读改写，再用 `write_fd(topo_fd, writes, ctx)` 写旧邻居拓扑页；本函数没有复用上述后台提交分支。I/O 的基本单位是当前 8192 字节 `SECTOR_LEN`，不是变量名或注释里的“4k”。`v2::lockReqs(page_lock_table, ...)` 在读改写前加排他页锁，在最后写完后释放，能串行化使用同一锁表的修改者。坐标与拓扑文件的页号放入同一锁表，没有文件身份，至少会引入跨文件同页号的额外串行化。

## 查询一致性：关键阻碍

不能将“写端有页锁”推导为“查询不会读到混合页面”。分离式插入持有的是 `page_lock_table`；无重排旧邻居原地改写时没有对所有受影响 ID 持有 `idx_lock_table` 写锁。`do_rerank_search` 的读取主要持有 `idx_lock_table` 读锁，并直接 `send_io(..., topo_fd)`，没有取得相同 `page_lock_table` 读锁；`page_search` 使用的 `page_idx_lock_table` 也是另一张表；Starling 页级路径的相关映射读锁调用还被注释掉。因此存在查询读请求与同页原地写请求重叠的合法程序交错。代码未建立这类 8192 字节并发读写的原子性保证。

组合记录路径有异地位置、映射锁和 `v2::cache` 完整镜像发布机制，不能说整个仓库完全没有一致性设计。但是分离路径直接写拓扑 fd，没有沿用组合路径的 `wbc_write` 发布；普通查询内页缓存也不等于可保持更新一致性的共享发布缓存。`trigger_deletion.cpp` 的注释说页锁会阻塞查询，实际查询没有取同一把锁，该注释不足以形成保证。本轮没有人为制造磁盘半页故障来“验证”它。

此外，新点的 `id2loc_topo_`、`id2loc_coord_` 和 tag 在坐标/拓扑写入前已登记，尚无明确 committed 状态。只在写线程持有的静态互斥锁内扩展 PQ `vector`，不能保护未取得该锁的查询读者免受扩容失效影响。这些也需要纳入新路径，而不能只补一个写锁。

## 删除标记及补边

`DynamicSSDIndex::lazy_delete` 在 `delete_lock` 排他锁下向 `tsl::robin_set<TagT>` 写入 tag。这个集合保存 tag，不是文档所说的节点 ID；默认身份 tag 不能掩盖二者差别。搜索把集合指针传入底层导航，部分导航读取发生在末尾结果过滤获得 `delete_lock` 共享锁之前；`active_delete_set` 的选择与集合切换也没有构成覆盖整次查询的稳定快照。因此不能直接认定现有删除集合可安全并发访问。

`DynamicSSDIndex::search` 最后会过滤活动删除集，旧 beam 导航也有根据删除节点扩展候选预算的逻辑；rerank 代码有删除标记检查，但部分精排过滤是注释状态。新实现必须保证“可经过已删除节点，精排和返回跳过”，并覆盖结果不足 K 的返回语义，不能只依靠搜索最后删掉几个结果。

`trigger_deletion_in_place` 先扫描已删除节点，保存其未删除邻居，再用“原存活邻居与被删除邻居的存活邻居的并集”修复未删除点，去自环，并在超度数时用单空间 PQ 剪枝。删除记录清零后回收拓扑/坐标位置。`merge_deletes` 则写新的组合索引、重编号存活 ID、重写 PQ 和 tags，最后 reload。两者均不是已实现的双空间 DEG 合并。动态包装器的插入拿 `_merge_lock` 共享锁，合并拿排他锁，所以现有包装器会阻塞合并期间的插入，不能直接满足文档要求的插入与合并并发。

## 剪枝接口与双空间迁移

旧更新调用 `SSDIndex::prune_neighbors(coord_map, vector<Neighbor>&, vector<uint32_t>&)`；反向边使用 `delta_prune_neighbors_pq` 或 `prune_neighbors_pq`；合并删除使用 `prune_neighbors_pq`。实现位于 `src/utils/prune_neighbors.cpp`，输入是一个标量距离，不能直接表达双空间 DEG 的帕累托层和区间支配。

双空间构图使用 `HybridIndex::iterate_to_fixed_point`、`prune_neighbors_no_opt(vector<NeighborH>&, Parameters&, vector<DEGNeighbor>&)`，剪枝内部直接访问常驻 `_data1/_data2`。不能为复用此接口而把整个更新拓扑或全量坐标搬到内存。适合的迁移是从已验证版本提取候选和距离提供接口，使剪枝在小候选池上运行，内部临时区间只服务选边，磁盘和查询不存取区间。`direct_to_topo_insert` 的 `use_double_pq` 是对同一个 `point` 采用两套码本并逐 chunk 选择编码，不是本次两个不同向量空间各存一份 PQ。

还有明确的源码基线差异：当前本地 `prune_neighbors_no_opt` 仍在帕累托层内部使用没有 `picked.size()<range` 条件的循环，零斜率分支也尚未补齐；距离直接调用平方 L2 内核。工作树两处现有修复仅涉及末区间合并。开方 L2、严格度数上限和零分支修正已在隔离实验变体中验证，见 `experiment_results/2026-11-11-metric-consistent/scripts/prepare_variant.py` 和 `experiment_results/2026-11-15-backlink-ab/function-v2/RESULTS.md`。新实现应固定实际验证过的变体源码/哈希，不能把当前本地默认构图函数等同于该版本。

此前 A/B 的整图 99% 门槛失败属于历史状态；本地后续结果记载，修订后的函数检查与小图搜索工作量门槛已通过，正式 2M A/B 尚在运行。本次不会沿用旧失败阻止基础更新准备，也不会提前宣布 B 是正式质量获胜版本。

## 可供选择的实现方案

建议在独立的紧凑双空间更新执行器中补齐统一页读写锁：以文件身份和物理页号为键，查询共享锁覆盖实际读完成及私有镜像复制，修改者排他锁覆盖读改写直至写完成；页缓存需要带版本或经同一协议失效。新点在两个空间 PQ、坐标、拓扑和反向边都准备好之后发布 committed 状态。PQ 使用不会因扩容使读者引用失效的分段存储或明确读写保护。合并以稳定删除快照修复邻居，旧读者退出后才回收 ID/位置，防止旧查询把复用 ID 当成原节点。此方案保持原 ID 顺序的固定页布局，比较贴近文档目标，但改变了“默认直接沿用旧一致性机制”的假设。

另一方案是文档提到的异地写入再切换映射。必须在新的完整页/受保护目标页中构造版本，完成写入后发布映射，并延迟旧页及 ID 回收。仅将记录搬进一个仍被其他查询读取的既有页，或者切换映射后立即复用旧位置，都不足以保证正确。此方案使物理位置不再始终按原 ID 顺序，需要明确“原 ID 顺序”是仅约束初始图还是也约束更新后快照。

下一步需要确定上述一致性方案和实际修正剪枝源码的归属，再在独立目录实现；既有单空间默认路径维持原行为。Linux 构建及小数据插入/删除验证应在现有队列释放同一全局锁后执行。G1–G4、E1–E3、完整编译选项及二进制身份均尚未产生，本报告不将它们标为通过。
