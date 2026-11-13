# 四线程 I/O 时间补充

此表从已有原生日志恢复 io_us，并在每次重复内匹配 90% Recall 后聚合。单位为 ms/查询，三次重复。

| 数据集 | Baseline 单文件 I/O | Full 总 I/O | Merge 总 I/O | No-alpha 总 I/O |
|---|---:|---:|---:|---:|
| OpenImages | 2.450 | 2.478 | 2.349 | 1.995 |
| CC3M | 6.401 | 6.713 | 6.491 | 5.096 |
| HowTo100M | 4.787 | 4.494 | 4.144 | 3.398 |
| Laion | 2.335 | 2.125 | 1.997 | 1.640 |

原生旧驱动没有导出 io_us1/io_us2，因此不能从上述总时间倒推出拓扑与坐标时间。Baseline 的旧 io_us 包含同步 reader 调用，解耦拓扑旧 io_us1 只包围 poll_wait，未包含 send_io；坐标旧 io_us2 包围 blocking read_fd。它们可按历史定义报告，统一 submit+wait 的比较使用后续补测。

统一补测将 baseline 单文件读取，与 Full/Merge/No-alpha 的拓扑提交/等待、坐标阻塞读取分别计时，同时导出实际传给后端的请求数和字节。父子时间不重复相加；原始精确距离计算不计入向量 I/O 时间。
