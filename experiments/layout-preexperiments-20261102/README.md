# Four-dataset alpha and five-layout preexperiments

Final status: **complete at 2026-11-03 09:26:41 CST**. All 2,400 query points, 240 trace replays, 480 field-patch tasks and 60 native-result parity checks passed the completion audit. The tmux session has exited. See [RESULTS.md](RESULTS.md) and [AUDIT.json](AUDIT.json); `completed/` is the downloaded final JSON/log snapshot, while `bootstrap/` retains the earlier launch snapshot.

This is the execution artifact for the November 2 alpha/layout plan in the manuscript worktree. It covers OpenImages, CC3M, HowTo100M and Laion. It does not alter the manuscript or the original HVS indexes. Existing four-method performance results remain a separate batch.

Node3 root: `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/layout-preexperiments-20261102/`. Tmux session: `hvs-layout-preexp-20261102`. Inspect `status.json`, `logs/queue.log`, and `analysis/layout-query-summary.json`. The session started at 2026-11-02 19:26:09 CST; all four complete alpha scans passed by 19:26:57. The first OpenImages native-result parity passed and formal query measurement began at 19:53:32. These are launch milestones, not a claim that the whole queue has completed.

## Scope and resources

All experiment processes execute serially. Search and trace replay use T=1,4,32,64, with 64 as the strict OpenMP limit. CPU affinity is confined to logical CPUs `0-47,96-111`, and memory is interleaved across NUMA 0,1. These two NUMA nodes contain 48 physical cores; T=64 includes SMT. BLAS threads=1. The long queue runs at nice=10 with low-priority disk I/O. Compilation and short correctness tests can run normally, per the user's instruction. Telemetry is observational and runs alongside the single experiment process.

The original data and index files are read-only sources. Each physical layout is generated, fully validated, measured and removed before the next layout. The largest Laion layout is about 81.2 GiB; the builder requires at least its full expected size plus 20 GiB free. It does not keep five Laion layouts simultaneously. All performance I/O uses new files on nvme3, not source files on a different device.

## Alpha measurements

`alpha_scan.cpp` scans every original Full topology record and validates degree, neighbor IDs, encoded intervals, empty-slot termination and page padding. It reports stored-pair and normalized connected-component histograms, normalized union-width and gap histograms, endpoint counts, per-node component/filter histograms, unused-neighbor capacity, empty slots inside valid edges, and page padding separately. Node-filter histograms use explicitly binned 0.001-wide bins; interval widths/gaps are exact integer-endpoint histograms.

Actual supplied float32 alpha values are multiplied by float32 100 and compared to closed integer intervals, including the reader's empty-slot and early-stop semantics. This is kept separate from historical round-to-integer analysis. There is also a check of runtime semantics versus normalized-set membership. For this scan their disagreement count is zero on all four sources. Raw query-alpha summaries include the original float32 values, histogram/CDF inputs and file hashes. No new query weights or GT are generated for performance runs.

The completed exact-reader filtering probabilities are approximately 29.2166% (OpenImages), 35.1995% (CC3M), 32.7067% (HowTo100M), and 42.6924% (Laion alpha=0.5). These are all-edge/query-weight statistics, not traversal-access-weighted or device-read reductions. The three mixed-alpha input vectors share the same SHA256; dataset queries and GT remain distinct.

## Physical format and common search

T contains the neighbor count, fixed R=40 ID slots and complete original alpha ranges (M=6/8/8/7). X and Y are the two exact float32 vectors. Five layouts are `TXY`, `TX_Y`, `TY_X`, `T_XY` and `T_X_Y`. A group is physically colocated in a fixed-width page record, not merely placed in the same file. Group bits are 1=T, 2=X, 4=Y; `group-7.bin` contains T/X/Y in one record, for example.

All layouts preserve original global ID order. A group's width is the sum of its field widths; records/page is floor(8192/width). Records do not cross pages. Every evaluated record fits in 8192 bytes; future overflow formats are explicitly unsupported. File length is ceil(N/records_per_page)*8192 and unused bytes are zero. Global ID to record ordinal is the identity, so there is no stored node map. Every output field of every node is compared byte-for-byte with its original source; all padding is validated before performance measurement. No Starling reorder is applied in this first controlled layout axis.

`layout_engine.cpp` is a new controlled harness using the original d2a39e2 PQ table/lookup, distance kernels, Neighbor and InsertIntoPool utilities. It reproduces the native Full PQ traversal and full retained-candidate exact rerank, while routing T/X/Y through one common physical-layout adapter. It is not the original coupled baseline implementation. New absolute timings must not be mixed into the old baseline/Full/Merge/No-alpha plots.

Before each layout enters the timed queue, L=40/100/200 results must have byte-identical per-query IDs and exact distances, matching Recall and PQ counts against the unchanged native d2a39e2 Full search. The complete expansion-batch and coordinate-candidate trace at L=200 must also be byte-identical across layouts. Any mismatch stops the queue. The original code/library remains unchanged; build flags, source hashes and binary identity are recorded.

Each query has an equal 16 MiB bounded LRU page cache, reused across topology and coordinate stages and reset at query boundaries. Co-located fields and duplicate requests in one batch share a page; they are not charged twice. Each batch submits up to 128 mixed-file Linux AIO reads, then waits before processing. This is batch execution, not claimed continuous CPU/I/O overlap. Cache hits, unique requests within batches, actual submitted pages/bytes and logical field bytes are distinct counters. Cache metadata, the shared PQ codes and RSS are recorded/accounted; the cache allowance is equal per query across layouts.

## Query and trace protocol

All four datasets use their existing 1,000 queries and matching GT: original mixed alpha on the first three, Laion alpha=0.5. K=10, beam=4, two 64-byte PQ codes, complete original alpha metadata, and full retained-candidate rerank are fixed. L=20,40,60,80,100,120,160,200,300,400. Each point warms up with one complete query set, then replays the same set continuously for at least ten measured seconds. Replays do not create independent queries.

There are 4 datasets × 5 layouts × 3 repeats × 4 thread settings = 240 query processes / 2,400 measured L points. Layout order and thread order rotate between repeats. Rebuilding layouts between repetitions is intentional: it allows rotation without retaining all five large files. A common L=200 reference trace is also replayed for each layout/repeat/thread combination, adding 240 storage-only probe points. These trace timings exclude PQ/distance/candidate processing and are not ANN QPS or evidence of recall improvement.

Each query point has an epoch start time, QPS, external service latency mean/p50/p95/p99, mutually nested phase fields, topology and coordinate submit/wait, candidate-processing, exact-distance and sort time, a residual time, inspected/rejected edges, PQ comparisons, expanded nodes, candidate insertions, exact candidates, page requests/hits/submissions, bytes and RSS. Broad coarse/rerank phases already include their subphases: do not add parent and child timers twice. External service timing surrounds the function call including reset/destruction, but does not represent open-loop queueing latency.

Each point additionally writes a 1,000-row `.L<L>.queries.json` with per-query alpha, replay count, mean latency and traversal/request counts. This supports access-weighted filtering and alpha-bucket statistics without pretending that mixed-alpha buckets are independent fixed-alpha workloads. These are per-query aggregates across replays, not every invocation's raw timestamp trace. Native parity dumps and logical batch traces are separately retained.

Three-repeat curves are aggregated incrementally. Target 90%/95% Recall is interpolated only inside measured coverage; missing coverage is reported rather than extrapolated. No new high-recall or alpha GT is generated. Timestamped iostat/mpstat and per-PID pidstat are retained. Device/CPU telemetry began during the first curve; per-PID telemetry starts with each timed process. Three repeats and a 1,000-query set do not establish robust tails or eliminate external CPU contention.

## Field-patch storage probes

These are single-writer storage probes, not dynamic ANN updates. Each layout and repeat executes 10,000 deterministic operations for X-only, Y-only, XY and T, separately for uniform IDs and a stated hot-ID distribution. In the hot distribution, 90% of operations target the first 1% of global IDs; the rest are uniform. Seeds are shared across layouts and differ between repeats. There are 480 patch tasks / 4.8 million logical operations in total.

Payloads replace the selected logical field bytes. Vector payloads change low mantissa bits; T payloads alter a valid interval endpoint while retaining IDs and fixed width. They do not run graph/PQ maintenance or evaluate GT after modification. Every operation reads/modifies/writes the affected physical pages with O_DIRECT. Co-located modified fields share their RMW page. Only dirty groups are fdatasync'ed after each batch of 128 operations and at the end. Service p50/p95/p99 exclude batch sync; throughput includes it, and sync time is reported separately. There is no WAL or crash-atomicity claim.

All touched pages are saved from the original generated-layout snapshot before timing. After each probe, entire touched pages are checked against the expected final contents, restored byte-for-byte and synced, then verified again. Thus every probe starts from the same logical snapshot. Source indexes are never opened for writing; their hashes are checked again after a dataset completes. Submitted read/write bytes are software amplification, not NAND write amplification. Concurrent query/update, open-loop update load, recovery and fully maintained dual-space ANN updates are not implemented in this batch.

## Validation and progress

Synthetic tests cover 2,503 nodes, unequal 1024/768 dimensions, zero/max degree, partial final pages, all five layouts, four-thread reads, AIO batches above 128 pages, LRU eviction, and 40 patch/restore cases. They create no dataset GT. Real native-result and cross-layout trace checks are mandatory thereafter.

`run_all.py` is the serial entry point; `layout_runner.py` manages physical layouts and measurements. `analysis/*-completed.json` records dataset/layout completion, while global `status.json` is complete only after all stages return successfully. Failed logs and partial outputs are preserved, and incomplete generated layouts are rebuilt on resume rather than reused after a possibly interrupted patch. Changed search code/library requires a new experiment identity. Old result files alone do not establish that a task completed.

At launch, full mixed-load dynamic ANN and single-space end-to-end recall comparisons remain unimplemented and outside the timed results. Same-byte field-group probes can describe generic grouping effects; they cannot establish a dual-space-only advantage. The manuscript plan and existing results are not rewritten by this queue.
