# 32-thread Full/Merge/No-alpha/Baseline experiment archive

This directory archives the September 20–22, 2026 node3 experiments for CC3M, OpenImages, and HowTo100M. It contains the raw search logs, the exact runner and validation scripts, alpha-metadata characterization output, generated CSV summaries, and checksums. Generated indexes and datasets remain on node3 and are not copied here.

## Reproducibility identity

- Repository: `git@github.com:wang29a/HVS-disk.git`
- Effective source: GitHub commit `d2a39e2` (`fix: use location dimension during rerank`)
- Node3 checkout state during execution: commit `0090a32` plus the one-line `datal_dim` change contained in `d2a39e2`
- External Starling: `wang29a/starling`, commit `7437c4848a83e7bf62558c5eab1eb999feed8537`
- Host: Intel Xeon Platinum 8360H, 192 logical CPUs, four NUMA nodes
- Search binding: 32 threads, `numactl --cpunodebind=0,1 --interleave=0,1`
- Search parameters: beam width 4, `K=10`, L2, 1,000 queries, three repetitions, `L={20,40,60,80,100,120,160,200,300,400}`
- Starling path: `mode=6`, `mem_L=0`, `strategy=17`
- Single-file baseline path: `mode=0`, `mem_L=0`, `strategy=0`

Full, Merge, and No-alpha use the same source graph, coordinates, PQ pivots, PQ codes, queries, alpha values, and ground truth. Each topology layout has its own metadata, Starling partition, and relayout. The baseline is the Full-alpha single-file layout without topology/coordinate separation or Starling.

## Directory layout

```text
raw/search/<dataset>/rep<1..3>-<scheme>.log
raw/characterization/<dataset>.json
scripts/                                  exact execution and analysis scripts
search-curve-summary.csv                  three-run mean at every measured L
matched-recall-summary.csv                linearly interpolated matched-recall values
characterization-summary.csv              storage and pruning summary
gated-format-summary.csv                   A0/A2 physical-size and map-overhead summary
GATED-A0-A2-DESIGN.md                     next-stage physical-format contract
SHA256SUMS                                archive integrity manifest
```

## Main observations

At matched Recall@10, Merge consistently reduces topology reads and improves QPS relative to Full. No-alpha reaches the same recall at a lower L and has the fewest topology and coordinate page reads among the separated layouts on all three datasets. The fixed Full-alpha slots are only 14.3%–19.5% utilized, while 97.9%–99.1% of edges need at most two ranges. Merge-2 adds only 0.062%–0.155% query-weighted active edges and introduces no measurable false negatives.

The single-file baseline uses a different I/O organization. Its `topology_ios` counter is not directly additive or comparable to the separated layouts' `topology_ios + rerank_ios`, although end-to-end QPS and latency remain valid system-level measurements.

## Interpretation boundaries

- Matched-recall rows are linear interpolations between measured L values, not additional executions.
- Report absolute measured points alongside interpolated operating points in the paper.
- `p999` over 1,000 queries is effectively the maximum observation; prefer p95/p99 in future instrumentation.
- The runs reuse an existing Full graph and PQ artifacts. They measure format conversion, relayout, and search, not end-to-end graph construction time.
- The range-tuple dictionary counts are exact but do not canonicalize semantically equivalent interval decompositions.
- Benefit-gated A0/A2 values in the JSON files are offline simulations. They do not establish end-to-end performance until a physical layout and reader are implemented.

## Next experiment

Implement a page-classed A0/A2 prototype with two initial thresholds:

- `0.20`: conservative, low extra edge activation;
- `0.30`: aggressive, meaningful topology-size reduction.

Compare Merge-2, No-alpha, Gated-0.20, and Gated-0.30 using independent Starling layouts and matched-recall Recall–QPS/Recall–I/O curves.

The first implementation step is `scripts/build_gated_a0_a2.py`. It creates the split A0/A2 physical format described in `GATED-A0-A2-DESIGN.md`; `scripts/validate_gated_a0_a2.py` independently reads every output record and verifies the map bijection, zero page padding, neighbor identity, and hashes. All six dataset/threshold conversions passed this validation with zero degree or neighbor mismatches.

The first physical prototype exposed a cost hidden by the earlier topology-only estimate: its direct node map used 8 bytes per node. The v2 format uses a 32-bit class/local-ordinal encoding, which is sufficient for the evaluated datasets and halves this overhead. `gated-format-summary.csv` reports both topology-only and topology-plus-map sizes; threshold 0.30 remains the meaningful capacity-saving point, while 0.20 is a conservative control whose small saving must still justify an extra file selection and map lookup in end-to-end search.

These outputs are intentionally not advertised as searchable until the HVS reader and Starling relayout paths gain explicit class-aware support.
