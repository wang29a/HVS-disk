# 比较符、距离契约与代码来源核对

核查日期：2026-11-13。上游固定到 DEG 作者公开仓库 commit `728c4128e74039ce212c8c22a42a5185abd68c23`，不以浮动 main 代替证据。

## 当前比较符与 DEG 论文

当前 v2 的固定剪枝采用 `d(c,p)<=d(c,q)` 且 `d(p,q)<=d(c,q)`；区间原语求解 `slope*alpha<=rhs`。相同距离时允许已有较早邻居覆盖候选，固定剪枝用 ID 打破排序平局。两个空间距离均为零时，LE 规则会剪掉后续等距候选。

DEG 论文 §4.3 式(2)–(5) 使用的是严格 `<`。严格规则中，两条距离必须都小于待剪边；全部距离为零时，不满足剪枝条件。论文 §3.2 的 RNG 性质表述和 §4.3 的闭区间例子也不能直接拿来消除这个平局区别。因此当前 LE 实验不能称为严格复现论文公式；它是明示的实现约定。[论文 §4.3](https://arxiv.org/html/2502.07343v1#S4.SS3)

本轮 OpenImages 旧／新图保持同一 v2 LE 目标，不在观察结果后静默切换比较符。强度扫描也必须注明比较符；若论文要求严格定义，需要另立严格规则的构图与对照，不能把 LE 和严格图混在一条曲线上。现有几何与 C++ 成员测试验证的是 LE 约定，不是严格约定。

由此还需收紧旧审计数字的解释：`0*alpha<=0` 对 LE 为真、对严格规则为假。以 LE 为反事实得到的边界敏感条目，包含“旧实现遗漏分支”和“平局政策不同”两部分，不能全部归为 DEG 代码缺陷。旧统计仍作为 LE 敏感性审计保留，不冒称作者论文的错误边数。

## 零系数遗漏：公开代码已存在，且不只涉及平局约定

`src/component_prune.cpp` 的 `ComponentDEGPruneHeuristic::PruneInner` 仅覆盖 slope/rhs 的严格正负组合，缺少零值处理。论文文字的部分情况包含非严格的符号边界，而公开代码没有对应覆盖。例如 `0*alpha<2` 和 `0*alpha<=2` 都应全域成立，公开分支却没有给出该剪枝区间；`-alpha<0` 在 (0,1] 成立，公开分支也没有覆盖。故确有不依赖 LE/strict 平局选择的上游实现遗漏。我们的代码继承了同样的条件结构；初版专用化适配和未检出语义边界的责任另行记录。

[上游固定版本：条件分支](https://github.com/Heisenberg-Yin/DEG/blob/728c4128e74039ce212c8c22a42a5185abd68c23/src/component_prune.cpp#L160-L183)，文件 blob SHA 为 `3852d62535b11aaf53c4fc6d3a15f503936176c3`。

## 区间合并末项遗漏：公开代码已存在，此前已经修复

上游 `include/component.h` 的 `ComponentInitDEG` 和 `ComponentDEGPruneHeuristic` 各有一份 mergeIntervals，循环停在 `i < intervals.size()-1`，没有处理最后一个区间。两个不相交区间就能暴露：结果只保留第一个。当前交接源码及 metric-consistent 快照已经使用 `i < intervals.size()`；这是此前已完成的修复，不算本次 v2 新改动。

[上游初始化实现](https://github.com/Heisenberg-Yin/DEG/blob/728c4128e74039ce212c8c22a42a5185abd68c23/include/component.h#L347-L372)，[上游剪枝实现](https://github.com/Heisenberg-Yin/DEG/blob/728c4128e74039ce212c8c22a42a5185abd68c23/include/component.h#L621-L646)，文件 blob SHA 为 `f7afb17a5777950dd101af624393feaa49080993`。

## 距离开方错位：迁移后的接口契约问题

上游 `include/distance.h` 的 E_Distance 返回平方和开方后再除以尺度，S_Distance 同样开方并缩放。原 HVS 调用链改用了返回平方 L2 的 DistanceL2，但构图上层直接使用返回值，精排／现有 GT 却采用开方融合，由此出现目标错位。这支持将其归为迁移后的距离接口契约问题，而不是 DEG 原距离函数忘记开方。

[上游距离函数](https://github.com/Heisenberg-Yin/DEG/blob/728c4128e74039ce212c8c22a42a5185abd68c23/include/distance.h#L58-L67)，blob SHA `b1f73f1f5d9c40501f4ce52368ef7da497f99192`；本地核对位置为 `src/utils/distance.cpp` 的 DistanceL2::compare 与 `src/hybrid_index.cpp` 的构图 compare 调用。交接 Git 历史被压成基线，不能据此精确指出最早是哪一次迁移提交或归责给具体作者。

当前隔离实验采用与现有 GT 一致的加权欧氏和，不额外使用上游 max-distance 归一化；因此即使平方／开方已经统一，也不应写成复现了 DEG 原论文全部尺度与数据处理设置。

以上归因只针对已核验的公开代码和迁移调用链，不据此宣称原论文全部结果失效。论文表述应分别说明实现问题、适配责任、比较符约定及复核范围。
