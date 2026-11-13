# Benefit-gated A0/A2 prototype

This is the next experiment derived from the archived Full/Merge/No-alpha results. It is deliberately a format prototype first: the current HVS loader and the external Starling implementation assume one fixed topology-record width per file, so they cannot safely consume mixed 164B and 324B records without an explicit extension.

For node `u`, let `filter_benefit(u)` be the query-alpha-weighted fraction of its Full-alpha outgoing edges that are inactive. With threshold `t`:

```text
filter_benefit(u) < t  -> A0: neighbor IDs only
filter_benefit(u) >= t -> A2: neighbor IDs plus at most two alpha ranges
```

The classification is computed from Full-alpha ranges and the measured query-alpha distribution. A2 records are copied from the already validated Merge-2 topology, while A0 records retain exactly the same degree and neighbor IDs and remove only range bytes.

## Prototype files

Each output prefix produces:

```text
<prefix>.a0.topo      fixed 164B records, 49 nodes per 8192B page
<prefix>.a2.topo      fixed 324B records, 25 nodes per 8192B page
<prefix>.node-map     global node ID -> class and class-local ordinal
<prefix>.manifest.json format, source identity, counts, sizes, and hashes
```

`node-map` begins with a little-endian header `<8sIIQ>`: magic `HVSGAT02`, version 2, entry width 4, and node count. Each following `uint32` uses bit 31 as the class (`0=A0`, `1=A2`) and the lower 31 bits as the class-local ordinal. Neighbor IDs remain global IDs; the map changes physical placement only. This representation supports fewer than `2^31` nodes in either class and halves the first prototype's unnecessary 8-byte-per-node map cost.

The split files preserve fixed-width records and 8192B page boundaries. This makes the physical-size result real, but these files are not yet accepted by the existing search binary.

## Staged validation

P0 converts Full plus Merge-2 into the split format and checks every node's degree and neighbor sequence. It also checks map bijection, page-aligned file sizes, expected page counts, zero padding, and source/output hashes. This stage establishes semantic preservation and exact storage cost.

P1 extends the HVS topology locator and reader to open both files and resolve a global ID through `node-map`. Search correctness must match Merge-2 when no node is assigned A0, and No-alpha when every node is assigned A0.

P2 adds class-aware Starling support. A0 and A2 require independent physical layouts and maps; a partition/reorder artifact generated for Full, Merge, or No-alpha must never be reused. The retained HVS metadata page remains authoritative; Starling's 4KB metadata output must not replace it.

P3 measures thresholds 0.20 and 0.30 against Merge-2 and No-alpha using the archived protocol: 32 threads, the same NUMA binding, three rotated repetitions, the same L sweep, and matched-recall Recall-QPS/Recall-I/O comparisons. Only after this step can the offline storage-pruning tradeoff be described as an end-to-end improvement.

## Acceptance criteria

The prototype advances to search integration only if conversion reports zero degree/neighbor mismatches, both topology files and all padding validate, the class counts reproduce the characterization JSON, and the exact byte counts match the page-count model. Search integration additionally requires endpoint tests for all-A0 and all-A2, followed by independent Starling relayouts for both threshold configurations.
