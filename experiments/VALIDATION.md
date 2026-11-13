# 提交验证（2026-11-12）

目标仓库是 `wang29a/HVS-disk`，本地路径为 `disk-hvs-new`；本次整理没有修改根目录生产 C++ 代码。node3 原始基础检出为 `d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e` 且工作区干净。

本地和 node3 均通过归档身份、95 份 Python 语法、22 份 shell 语法检查，以及 NUMA 5 项、队列 5 项、Laion 协议 8 项已有回归。归档完整性检查另外覆盖新增检查器自身的语法。移动本地分析入口、配置结果归档根路径后再次在本地运行全部轻量检查。

node3 在独立临时目录、CPU 76–77、NUMA 3 重新生成隔离源码，生成记录与正式实验的 `metric-variant.json` 完全一致。重新编译 `hybrid_index.cpp`、`aux_utils.cpp` 并替换静态库中对应对象，链接并运行距离契约回归、区间合并/补集回归、metric 搜索器 selftest、拓扑驻留搜索器 selftest；独立编译 Laion Full writer 与 edge-audit，并通过端点、首发现父边和非法区间回归。

验证日志及 `validation.json` 在 node3 `/tmp/wz-hvs-preexperiment-commit-6d5lm3tm/`，临时目录不是长期结果存档。这次没有启动性能队列，没有重建全量索引，没有重跑论文实验；通过编译和回归不代表已在新机器完成端到端复现。早期历史搜索变体未逐一重新编译。
