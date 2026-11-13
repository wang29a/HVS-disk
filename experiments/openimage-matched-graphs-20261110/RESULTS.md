# OpenImages, independent X graph versus dual graph at alpha=1

Three-repeat means at matched Recall@10. p95 uses the nearest measured L.
All methods share the query set, X PQ, physical ID order, search harness and cache budget.

| Recall@10 | Variant | L (interpolated) | QPS | Mean ms | p95 ms | Actual p95 recall | Topology pages | Coord pages | PQ evals |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 90% | single | 23.0 | 8634 | 0.464 | 0.595 | 90.78% | 0.00 | 21.67 | 989 |
| 95% | single | 35.5 | 6161 | 0.666 | 0.858 | 95.12% | 0.00 | 33.10 | 1153 |
| 98% | single | 63.6 | 4020 | 1.015 | 1.418 | 98.19% | 0.00 | 58.37 | 1485 |
| 90% | dual | 33.5 | 5248 | 0.763 | 0.934 | 89.86% | 0.00 | 33.48 | 2673 |
| 95% | dual | 64.3 | 3241 | 1.238 | 1.641 | 95.19% | 0.00 | 64.30 | 3803 |
| 98% | dual | 141.4 | 1381 | 2.915 | 3.408 | 98.09% | 0.00 | 141.41 | 6408 |
| 90% | dual_x | 33.5 | 6503 | 0.616 | 0.765 | 89.86% | 0.00 | 31.17 | 1336 |
| 95% | dual_x | 64.3 | 4055 | 0.988 | 1.328 | 95.19% | 0.00 | 58.73 | 1902 |
| 98% | dual_x | 141.4 | 1624 | 2.479 | 2.984 | 98.09% | 0.00 | 126.74 | 3204 |
