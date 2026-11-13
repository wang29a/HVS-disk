# OpenImages alpha filter and page format experiment

The node3 queue `hvs-alpha-page-factorial-20261110` uses the archived OpenImages mixed-alpha queries and GT, the Full graph, the existing PQ codes and medoids, and a single T|XY search implementation. It holds the existing shared experiment lock and uses four physical cores on NUMA 3.

| Variant | Topology | Filter | Page source |
|---|---|---|---|
| A | Full | Inline Full intervals | SSD |
| B | Full | Off | SSD |
| C | Compact | Off | SSD |
| D | Compact | Full intervals in a DRAM sidecar | SSD |
| A_dram / B_dram | Full | As in A / B | Resident DRAM copy of both page groups |

All six variants start with the cross-query cache off. The second block gives A/B/C/D 256 MiB of extra memory each. D counts the sidecar against the budget and rounds the remaining page capacity down. Query-local page reuse stays on throughout. A/B use identical physical topology pages, and all four layouts share the coordinate file.

`analysis/parity.json` records byte-identical traces, rerank candidates and outputs for A/D, B/C, the SSD/DRAM pairs, and the cache toggle. `analysis/selected.json` records the L values selected around 90% and 95% Recall@10. Three rotated repetitions use at least ten seconds per timed point, with quiet CPU and disk gates and archived telemetry. `analysis/measurements.json` retains accepted and flagged runs, and `analysis/matched.json` holds the matched-recall comparisons. The JSON separates topology and coordinate requests, distinct pages and service times. In DRAM rows, the inherited `*_submitted_pages` fields count memory-served pages and the corresponding wait fields record memory service time. SSD rows count submissions through the direct I/O path.

The queue `hvs-laion-alpha-ab-20261110` waits for OpenImages, then runs A/B on the same archived Full pages in original node order. It uses the existing paired query weights and GT at 0.1, 0.3, 0.5, 0.7 and 0.9. It reports the 90% and 95% targets only when both variants cover them. This follow-up tests sensitivity on SSD with the shared cache off.

The queue `hvs-native-deg-alpha-20261110` waits for Laion, then runs the DEG active-range ablation on the native 507,444-row OpenImage data. It archives revision `728c412` into an isolated directory, uses the verified native graph at `/mnt/nvme3/wz/Index/Openimage/deg.index`, and searches with one thread using the original exact-distance router. The existing graph is referenced by symlink and pinned by SHA-256 after a read-only search check against the matching GT. Its original construction command is unavailable. If the reuse check fails, the queue builds one graph using eight quiet physical cores on NUMA 3. DEGStatic bypasses the edge interval check and reads the same graph. The workloads use the existing mixed query/GT pair and five original weight buckets. The mixed pair is converted from matrix headers to per-row headers without changing weights or IDs. This run uses node3 hardware and reports its own curves independently of the SSD/DRAM comparison.

All queues are under `/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/alpha-page-factorial-20261110`. OpenImages reports to `status.json` and `logs/queue.log`. Laion reports to `laion-fixed/status.json` and `laion-fixed/queue.log`. Native DEG reports to `native-deg/status.json` and `native-deg/queue.log`.

The native graph archive contains 507,444 consecutive node IDs, 13 entry points and 19,922,190 edges, with a maximum degree of 40. The format check traversed every edge and found valid endpoints in all stored intervals. The read-only search check is recorded in `native-deg/reuse-smoke.json`. The queue records the selected graph path, SHA-256 and whether it reused the archive in `native-deg/analysis/graph.json`.

The November 11 continuation uses `factorial.py --resume` to retry only missing accepted repetitions after checking the archived binary, layout identities and parity hashes. It retains prior attempts and assigns fresh attempt numbers. Laion one-pass screening excludes points with query-local cache eviction and records them in `analysis/*-screen-excluded.json`. At fixed alpha 0.1 the first A sweep had 73 total evictions at L=800, while L<=600 had none and already bracketed 90% and 95% recall. Formal timing still requires zero evictions. OpenImages C/D and DRAM contain visibly variable repetitions despite passing the original disk-load checks. Inspect individual repetitions before attributing their latency differences to layout or medium.
