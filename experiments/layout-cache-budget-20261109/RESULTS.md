# Layout-cache and budget, four search threads

QPS and reader metrics interpolate within each repeat. p95 uses the nearest measured L; consult matched.csv for the actual recall.
Page counts are reader submissions. Only conditions with three completed repeats appear.

| Dataset | Stage | Budget | Layout | Cache | Recall | QPS | p95 (ms) | Pages / query |
|---|---|---|---|---|---:|---:|---:|---:|
| howto100m | budget | 0 | T_XY | on | 90% | 951.1 | 5.25 | 153.94 |
| howto100m | budget | 0 | T_XY | on | 95% | 448.2 | 10.75 | 345.43 |
| howto100m | budget | 1 | T_XY | on | 90% | 793.2 | 6.21 | 187.62 |
| howto100m | budget | 1 | T_XY | on | 95% | 392.4 | 12.11 | 403.97 |
| howto100m | budget | 2 | T_XY | on | 90% | 796.9 | 6.21 | 193.09 |
| howto100m | budget | 2 | T_XY | on | 95% | 388.0 | 12.55 | 419.23 |
| howto100m | budget | 3 | T_XY | on | 90% | 663.9 | 9.34 | 210.25 |
| howto100m | budget | 3 | T_XY | on | 95% | 321.0 | 14.99 | 458.08 |
| howto100m | budget | full | T_XY | on | 90% | 688.0 | 6.81 | 232.13 |
| howto100m | budget | full | T_XY | on | 95% | 329.6 | 13.65 | 503.90 |
| howto100m | four-cell | 2 | TXY | off | 90% | 821.9 | 5.67 | 161.66 |
| howto100m | four-cell | 2 | TXY | off | 95% | 407.8 | 10.63 | 331.33 |
| howto100m | four-cell | 2 | TXY | on | 90% | 715.4 | 6.52 | 148.33 |
| howto100m | four-cell | 2 | TXY | on | 95% | 343.9 | 13.07 | 315.68 |
| howto100m | four-cell | 2 | T_XY | off | 90% | 676.9 | 7.44 | 229.10 |
| howto100m | four-cell | 2 | T_XY | off | 95% | 288.9 | 14.28 | 477.29 |
| howto100m | four-cell | 2 | T_XY | on | 90% | 796.7 | 6.67 | 193.09 |
| howto100m | four-cell | 2 | T_XY | on | 95% | 375.5 | 13.13 | 419.22 |
| openimage | budget | 0 | T_XY | on | 90% | 2624.1 | 1.93 | 69.83 |
| openimage | budget | 0 | T_XY | on | 95% | 1169.6 | 4.19 | 149.60 |
| openimage | budget | 1 | T_XY | on | 90% | 2668.4 | 1.97 | 73.22 |
| openimage | budget | 1 | T_XY | on | 95% | 1092.1 | 4.20 | 166.98 |
| openimage | budget | 2 | T_XY | on | 90% | 2495.3 | 2.05 | 77.32 |
| openimage | budget | 2 | T_XY | on | 95% | 1019.7 | 4.90 | 173.22 |
| openimage | budget | 3 | T_XY | on | 90% | 2472.0 | 2.12 | 77.32 |
| openimage | budget | 3 | T_XY | on | 95% | 1043.8 | 4.45 | 173.45 |
| openimage | budget | full | T_XY | on | 90% | 2371.8 | 2.37 | 77.32 |
| openimage | budget | full | T_XY | on | 95% | 969.1 | 4.86 | 173.45 |
| openimage | four-cell | 2 | TXY | off | 90% | 1153.3 | 5.04 | 95.81 |
| openimage | four-cell | 2 | TXY | off | 95% | 642.2 | 7.35 | 189.71 |
| openimage | four-cell | 2 | TXY | on | 90% | 1253.6 | 4.39 | 65.45 |
| openimage | four-cell | 2 | TXY | on | 95% | 622.3 | 8.22 | 140.49 |
| openimage | four-cell | 2 | T_XY | off | 90% | 1251.0 | 4.02 | 132.13 |
| openimage | four-cell | 2 | T_XY | off | 95% | 634.8 | 7.58 | 271.14 |
| openimage | four-cell | 2 | T_XY | on | 90% | 2556.9 | 2.04 | 77.32 |
| openimage | four-cell | 2 | T_XY | on | 95% | 901.1 | 9.21 | 173.22 |
