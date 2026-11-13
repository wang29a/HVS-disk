# Recall–Latency / Recall–QPS：1、2、4、8、16、32 线程

这些图完全复用 2026-10-27—28 的四数据集持续回放实验，没有重新运行搜索、构建索引或计算 GT。绘图前从 576 份原始任务 JSON 重新计算三次重复的均值和样本标准差，并逐项核对归档 summary.json；共 5,760 个原始测量点、1,920 个均值点。

每张图包含六个子图，依次为 1/2/4/8/16/32 线程；每个子图显示 Baseline、Full-alpha、Merge-2、No-alpha 四条曲线。横轴是 Recall@10（百分比）；纵轴分别是原实现的平均搜索延迟（毫秒）和 QPS。每个点对应原有 L=20/40/60/80/100/120/160/200/300/400，连线只连接原生实测点，没有平滑、额外拟合或外推。误差棒是三次重复的样本标准差，不是置信区间。

同一张图采用统一 Recall 横轴，各子图纵轴独立缩放，方便看清每个线程档位内的方法差异；跨子图比较时需要读取纵轴刻度。灰色竖虚线标记 90%/95% Recall，曲线没有达到这些目标时不会补造数据。

`recall-latency-all.pdf` 和 `recall-qps-all.pdf` 各八页，前四页依次是 OpenImages、CC3M、HowTo100M、Laion(alpha=0.5)；后四页是 Laion(alpha=0.1/0.3/0.7/0.9)。`main/` 保存四个主工作负载的八张图；`supplement/` 保存 Laion 额外 alpha 的八张图。每张图均提供 PNG、矢量 PDF、可编辑 SVG。`recall-curves-t1-t32.zip` 包含全部图、绘图脚本、数据和校验记录。

前三个数据集使用各自已有的 mixed-alpha 查询；Laion 使用对应固定 alpha 及配套 GT。曲线属于原 SSD 路径实验，NUMA 0、1 绑定、内存交错、beam=4、K=10、两空间各 64-byte PQ、1,000 个独立查询循环回放，每点至少五秒。没有把后续只有 1/4 线程的热数据实验混入这些图，也没有混入 64 线程。

Latency 特别说明：原驱动保存的 latency_us 来自检索实现内部 QueryStats.total_us，未覆盖部分查询准备/缓冲区获取时间。图中明确标注 internal timer，不能称为完整入口到返回延迟。QPS 则覆盖原驱动的整个计时并行循环，不能将两者简单互取倒数。完整实验边界见 ../concurrency-alpha/README.md 和 RESULTS.md。

来源身份、逐原始文件 SHA256、软件版本和输出清单在 manifest.json；图中全部均值、标准差和三次原始数值保存在 plot-data.json。SHA256SUMS 可核验输出。用安装了 Matplotlib 的 Python 3 运行 `plot_recall_curves.py` 可再生成全部图，原始归档需要保留在同级的 concurrency-alpha/。
