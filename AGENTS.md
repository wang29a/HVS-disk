# hvs-disk Agent Guide

hvs-disk 是面向 SSD 的双空间图 ANN 实现。查询使用 `alpha` 融合两个距离空间，磁盘拓扑支持 Full-alpha、Merge-alpha 和 No-alpha 三种布局。开始工作前先完整阅读 `README.md`；其中记录了数据格式、构建流程、外部 Starling 版本、搜索参数和 node3 验证结果。

项目以 Linux 为构建和实验环境。修改 C++ 后应使用以下配置编译，并运行与改动对应的测试：

```bash
export ADDITIONAL_DEFINITIONS="-DREORDER_COMPUTE_PQ -DUSE_TOPO_DISK"
cmake -S . -B build -DBUILD_WITH_PQ=ON -DCMAKE_CXX_COMPILER=g++-11
cmake --build build -j
```

每个磁盘页固定为 8192B。修改 metadata、邻居记录或 topology layout 时，必须同步检查构建器、转换器、所有搜索 reader、偏移计算和 Starling relayout。Full、Merge、No-alpha 的记录宽度不同，必须分别生成 partition 和 reorder，不能共享映射。

No-alpha 的 `max_alpha_range_len=0` 表示所有邻居都可展开，不能按“没有有效区间”过滤。Full/Merge 则从 topology record 内解析固定数量的 alpha slots。PQ 粗排与 exact rerank 的距离公式不同，不要混用。

优先使用 `rg` 查找代码。保留无关的已有改动，不要提交数据集、索引、构建产物、日志或论文材料。实验比较应使用匹配 Recall 的 operating point，并记录完整索引版本、参数、硬件和原始日志。未经明确要求，不要执行 `git add`、`git commit`、`git push` 或破坏性 Git 操作。
