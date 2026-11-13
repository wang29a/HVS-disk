# Targeted quiet-window retest

| Dataset | Budget | Layout | Cache | Recall | Original QPS range | Retest QPS (quiet) | Retest p95 ms (quiet) |
|---|---|---|---|---:|---:|---|---|
| howto100m | 2 | T_XY | on | 95.11% | 377–384 | 397, 393, 398 | 12.14, 12.41, 12.09 |
| howto100m | 3 | T_XY | on | 95.10% | 261–349 | 356, 360, 357 | 12.93, 12.73, 12.90 |
| openimage | 2 | TXY | off | 95.12% | 503–705 | 725, 722, 719 | 6.10, 6.14, 6.18 |
| openimage | 2 | TXY | on | 95.12% | 567–639 | 702, 699, 700 | 6.51, 6.52, 6.60 |
| openimage | 2 | T_XY | off | 95.12% | 610–640 | 644, 643, 638 | 7.18, 7.18, 7.34 |
| openimage | 2 | T_XY | on | 95.12% | 440–1063 | 1211, 1219, 1215 | 3.85, 3.81, 3.86 |

Flagged timing attempts are preserved in the JSON and are absent from the quiet columns.
