# Alpha 区间正确性审计

2026-11-11 在 node3 完成离线审计与隔离构建验证。后台会话名为 hvs-alpha-audit-20261111，最多使用 NUMA 3 的两个物理核，nice=15。新实验协议位于 ../alpha-page-factorial-20261110/NEXT-EXPERIMENTS.md。

## 已完成的源图到磁盘图审计

审计输入为 /mnt/nvme3/wz/Index/Openimage/ 下的 mem.index、_disk.index 和 disk_index_graph。完整读取 456,699 个节点、17,986,239 条边和 21,372,966 对区间。邻居 ID 与顺序、区间字节、有效覆盖以及 padding 均通过核验，覆盖缩窄或扩大边数均为零。逐边校验包含 0..100 的闭区间端点，另以百分之一宽度单位计算覆盖长度，检查 reader 的零槽位终止语义与区间排序。

mem.index 的区间端点已经是 int8。上述审计覆盖整数内存图到 Full 磁盘页这一阶段。浮点 available_range 在 save_graph_single 时乘以 100，再对两端都 static_cast<int8_t>。缺少量化前的浮点记录，现有文件无法统计浮点到整数这一阶段实际损失的边数和范围。

## mergeIntervals 缺陷与修复

本地 src/hybrid_index.cpp 和 src/utils/aux_utils.cpp 中两处循环原来使用 i<intervals.size()-1，忽略排序后的最后一个区间。node3 原生 DEG revision 728c4128e74039ce212c8c22a42a5185abd68c23 的 include/component.h 第 347、621 行定义也使用同一循环边界，两处函数体的 SHA-256 一致。

合并函数的输入是剪枝区间，随后 get_use_range 取补集。遗漏最后一个剪枝区间可以扩大活动范围。例如剪枝输入 [0.1,0.2]、[0.7,0.9]，旧合并结果只剩前者，补集错误放行 [0.7,0.9]。后续候选选择可能受影响，不能由这个例子推定整图边数变化的方向或数量。

本地两处循环已改为 i<intervals.size()。Linux 验证按既有 separated 构建参数完整编译这两个翻译单元，将对象替换到隔离静态库后链接回归可执行文件。原库在两个分离区间的用例上失败，修复库通过七组区间合并用例及三组取补集检查。该验证覆盖两处真实实现，未重新构建完整实验图。

已保存的索引仍体现历史构图器行为。后续若评价修正后的构图器，应重新生成同源图，再派生各格式并单独标注。原版 DEG 复现实验也需要写清使用的实现版本。

## 文件

analysis/openimage-mem-disk.json 保存全图审计计数，analysis/input-hashes.json 固定输入文件身份，analysis/original-deg-merge.json 保存原版函数与版本核对，analysis/build-and-regression.json 保存隔离编译与测试记录。日志区分原库的预期失败与修复库通过。审计器曾用合成的部分尾页做校验，修改一个端点后可以检测出一条边的覆盖收缩和一个整数权重点的丢失。
