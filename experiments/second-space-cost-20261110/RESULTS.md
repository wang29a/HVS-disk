# Second-space cost at a fixed answer

A computes and stores XY. B computes X and stores XY. C computes and stores X.
All three use the same M=2 graph, alpha=1, identical answers, four threads and 256 MiB topology cache.
Rows show the mean of three accepted curves. Reader timing is submit and wait wall time.

| L | Variant | QPS | Mean ms | p95 ms | PQ gather + lookup ms | Exact ms | Coordinate pages | Coordinate I/O ms |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 40 | A | 4689 | 0.853 | 1.058 | 0.210 | 0.047 | 40.00 | 0.244 |
| 40 | B | 5883 | 0.679 | 0.867 | 0.100 | 0.020 | 40.00 | 0.257 |
| 40 | C | 5952 | 0.672 | 0.852 | 0.100 | 0.022 | 37.03 | 0.248 |
| 80 | A | 2692 | 1.485 | 1.934 | 0.319 | 0.093 | 80.00 | 0.524 |
| 80 | B | 3007 | 1.330 | 1.727 | 0.156 | 0.048 | 80.00 | 0.608 |
| 80 | C | 3278 | 1.220 | 1.621 | 0.158 | 0.050 | 72.64 | 0.495 |
| 176 | A | 1203 | 3.323 | 3.833 | 0.520 | 0.191 | 176.00 | 1.652 |
| 176 | B | 1300 | 3.076 | 3.522 | 0.268 | 0.106 | 176.00 | 1.753 |
| 176 | C | 1388 | 2.881 | 3.345 | 0.268 | 0.112 | 157.10 | 1.550 |
