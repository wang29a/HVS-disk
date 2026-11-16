# EDBT evidence: priorities1–4 complete

All216 native comparison timing batches accepted; pending0. Matched-recall coverage216/216 records in the4GiB supplemental sweep. Original2GiB coverage180/216 records, excluding native32-thread cases whose scratch RSS exceeds2GiB. The count of matched records and timing batches happens to coincide; they are different denominators. Original failed NUMA observations were not reused; old accepted<=16-thread points retained with identity-backed provenance,32-thread points measured anew. No graph/binary/GT/parameter selection was changed by the monitor repair.

The earlier reverse-range audit is superseded by the completed [backlink A/B validation](../backlink-ab-20261115/RESULTS.md). No additional graph or timing task was launched for this report. Third dataset and endpoint performance remain planned, not completed; prior weighted-dual anomaly analysis is complete.

## Native single-file comparison, 95% Recall@10

Same corrected Euclidean objective/source graph/PQ/query/GT, separate tune-frozen configurations, original ordering, no persistent shared cache; this is an end-to-end search-path comparison, not a pure layout ablation or a Starling/cache bundle comparison. Means below are calculated by interpolation within each repeat then averaging three repeats. Higher-thread results are retained even when compact loses.

| workload | threads | native ms | compact ms | compact latency reduction |
|---|---:|---:|---:|---:|
| openimages | 1 | 3.4125 | 2.8811 | +15.57% |
| openimages | 2 | 3.6701 | 2.9941 | +18.42% |
| openimages | 4 | 3.8961 | 3.3200 | +14.79% |
| openimages | 8 | 4.2039 | 4.3505 | -3.49% |
| openimages | 16 | 5.2580 | 8.2082 | -56.11% |
| openimages | 32 | 10.1917 | 15.9982 | -56.97% |
| laion-0.1 | 1 | 4.1781 | 3.0227 | +27.65% |
| laion-0.1 | 2 | 4.4965 | 3.1638 | +29.64% |
| laion-0.1 | 4 | 4.7900 | 3.5353 | +26.19% |
| laion-0.1 | 8 | 5.1867 | 4.6492 | +10.36% |
| laion-0.1 | 16 | 6.6539 | 8.8157 | -32.49% |
| laion-0.1 | 32 | 13.1391 | 17.0709 | -29.92% |
| laion-0.3 | 1 | 3.7113 | 2.8604 | +22.93% |
| laion-0.3 | 2 | 3.9849 | 2.9797 | +25.22% |
| laion-0.3 | 4 | 4.2457 | 3.2792 | +22.76% |
| laion-0.3 | 8 | 4.5849 | 4.2108 | +8.16% |
| laion-0.3 | 16 | 5.7969 | 7.9599 | -37.31% |
| laion-0.3 | 32 | 11.4000 | 15.3405 | -34.57% |
| laion-0.5 | 1 | 3.2391 | 2.8549 | +11.86% |
| laion-0.5 | 2 | 3.4651 | 2.9751 | +14.14% |
| laion-0.5 | 4 | 3.6898 | 3.2750 | +11.24% |
| laion-0.5 | 8 | 3.9839 | 4.2663 | -7.09% |
| laion-0.5 | 16 | 4.9795 | 8.0930 | -62.53% |
| laion-0.5 | 32 | 9.7591 | 15.6394 | -60.25% |
| laion-0.7 | 1 | 3.5007 | 2.9137 | +16.77% |
| laion-0.7 | 2 | 3.7674 | 3.0360 | +19.41% |
| laion-0.7 | 4 | 4.0177 | 3.3870 | +15.70% |
| laion-0.7 | 8 | 4.3443 | 4.4899 | -3.35% |
| laion-0.7 | 16 | 5.4625 | 8.4901 | -55.43% |
| laion-0.7 | 32 | 10.6932 | 16.5190 | -54.48% |
| laion-0.9 | 1 | 4.2388 | 3.4227 | +19.25% |
| laion-0.9 | 2 | 4.5686 | 3.6018 | +21.16% |
| laion-0.9 | 4 | 4.9060 | 4.0628 | +17.19% |
| laion-0.9 | 8 | 5.3310 | 5.4614 | -2.45% |
| laion-0.9 | 16 | 6.6890 | 10.3505 | -54.74% |
| laion-0.9 | 32 | 13.1272 | 20.0507 | -52.74% |

At four threads, OpenImages latency reduction is14.79%; Laion2M across the five weights is11.24%–26.19%. All these T4 pairs fit the original2GiB ceiling. At16/32 threads the compact path is slower in every tested workload;8-thread results are mixed. Do not claim superiority across concurrency. The4GiB allowance is a common ceiling, not equal actual RSS, and no memory padding or added cache was used.

## OpenImages full multigraph

Four threads NUMA3, existing mixed-alpha queries/GT, three repeats, weighted-page rule. Methods are implemented using Vamana/DEG, not native HNSW. Nearest routing uses only fixed0.1/0.5/0.9 graphs.

| method | recall | latency ms | QPS | topology pages | coordinate pages |
|---|---:|---:|---:|---:|---:|
| DEG_off | 90% | 2.0257 | 1968.8 | 56.25 | 37.89 |
| DEG_off | 95% | 3.3208 | 1205.6 | 94.38 | 64.21 |
| M_equal | 90% | 6.4020 | 634.7 | 185.71 | 154.15 |
| M_equal | 95% | 11.1179 | 378.2 | 319.37 | 291.16 |
| Or_nearest | 90% | 2.0426 | 1954.1 | 59.90 | 37.66 |
| Or_nearest | 95% | 3.2131 | 1290.2 | 91.19 | 72.93 |

At95% recall, unified no-range3.3208ms vs equal dual11.1179ms and nearest specialized3.2131ms. The unified graph is3.35x faster by latency ratio than equal dual and3.35% slower than the three-specialized route at this operating point, below the declared10% meaningful-gap threshold. This supports a shared-graph storage/performance tradeoff within this workload, not universal necessity or equal performance at every weight.

Priority1: relaxed unified degree33.13→36.68/38.42 with unchanged topology bytes, but no stable10% gain; no design point retained. See relaxed/summary.json. Priority4: no-sidecar three repeats complete; RSS about335–337MiB vs loaded1557–1559MiB, similar to one dedicated graph; mean latency change vs loaded within0.2%. See no-sidecar-three/summary.json.

The subsequent [function gate and corrected-author small-graph control](../backlink-ab-20261115/FUNCTION-RESULTS.md) passed; the [Laion2M A/B comparison](../backlink-ab-20261115/RESULTS.md) retains the filtering disadvantage at the two measured SSD weights. This does not validate every earlier Full-layout or OpenImages filtering point. Native/counting and concurrency measurements above retain their original A graph and scope.

See [topology-resident concurrency diagnosis](CONCURRENCY-RESULTS.md) for the later controlled 4/16/32-thread experiment. Runtime JSON, raw p95 and the accepted monitor records stay in `runtime/` on node3 or the independent result archive; they are not shipped in Git.
