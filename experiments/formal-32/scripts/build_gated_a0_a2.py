#!/usr/bin/env python3
"""Build and validate the split-file benefit-gated A0/A2 format."""

import argparse
import hashlib
import json
import os
import struct

import numpy as np


PAGE = 8192
MAP_MAGIC = b"HVSGAT02"
MAP_HEADER = struct.Struct("<8sIIQ")
CLASS_BIT = np.uint32(1 << 31)


def read_meta(path):
    with open(path, "rb") as stream:
        raw = stream.read(PAGE)
    if len(raw) != PAGE:
        raise ValueError("metadata must contain one complete 8192B page: " + path)
    n, d1, d2, degree, slots = struct.unpack_from("<IIIII", raw, 0)
    npp = struct.unpack_from("<Q", raw, 20)[0]
    return {
        "n": n, "d1": d1, "d2": d2, "degree": degree, "slots": slots,
        "npp": npp, "record_bytes": 4 + degree * (4 + 2 * slots),
    }


def read_alpha_weights(path):
    with open(path, "rb") as stream:
        q, dim = struct.unpack("<II", stream.read(8))
        if dim != 1:
            raise ValueError("query alpha must be Qx1")
        alpha = np.fromfile(stream, dtype="<f4", count=q)
    if alpha.size != q:
        raise ValueError("truncated query-alpha file")
    buckets = np.clip(np.rint(alpha * 100.0).astype(np.int64), 0, 100)
    histogram = np.bincount(buckets, minlength=101).astype(np.float64)
    return histogram / histogram.sum(), int(q)


def read_records(mm, meta, start, count):
    ids = np.arange(start, start + count, dtype=np.int64)
    offsets = ((ids // meta["npp"]) * PAGE
               + (ids % meta["npp"]) * meta["record_bytes"])
    indices = offsets[:, None] + np.arange(meta["record_bytes"], dtype=np.int64)
    return mm[indices]


def parse(records, meta):
    count = records.shape[0]
    degree = records[:, :4].copy().view("<u4").reshape(count)
    if np.any(degree > meta["degree"]):
        raise ValueError("node degree exceeds metadata maximum")
    body = records[:, 4:].reshape(count, meta["degree"], 4 + 2 * meta["slots"])
    neighbors = body[:, :, :4].copy().view("<u4").reshape(count, meta["degree"])
    ranges = body[:, :, 4:].view(np.int8).reshape(
        count, meta["degree"], meta["slots"], 2)
    return degree, neighbors, ranges


def node_filter_benefit(degree, ranges, weights):
    buckets = np.arange(101, dtype=np.int16)[None, None, :]
    cover = np.zeros((degree.size, ranges.shape[1], 101), dtype=np.bool_)
    for slot in range(ranges.shape[2]):
        lo = ranges[:, :, slot, 0].astype(np.int16)
        hi = ranges[:, :, slot, 1].astype(np.int16)
        valid = ((lo != 0) | (hi != 0)) & (lo >= 0) & (hi >= lo) & (hi <= 100)
        cover |= (valid[:, :, None] & (buckets >= lo[:, :, None])
                  & (buckets <= hi[:, :, None]))
    active = np.tensordot(cover, weights, axes=([2], [0]))
    edge_valid = np.arange(ranges.shape[1])[None, :] < degree[:, None]
    active_sum = (active * edge_valid).sum(axis=1)
    result = np.zeros(degree.size, dtype=np.float64)
    nonempty = degree > 0
    result[nonempty] = 1.0 - active_sum[nonempty] / degree[nonempty]
    return result


class PagedWriter:
    def __init__(self, path, record_bytes, nodes_per_page):
        self.stream = open(path, "wb")
        self.record_bytes = record_bytes
        self.nodes_per_page = nodes_per_page
        self.pending = np.empty((0, record_bytes), dtype=np.uint8)
        self.count = 0

    def append(self, records):
        if records.size == 0:
            return
        records = np.ascontiguousarray(records, dtype=np.uint8)
        self.count += records.shape[0]
        if self.pending.shape[0]:
            records = np.concatenate((self.pending, records), axis=0)
        complete = (records.shape[0] // self.nodes_per_page) * self.nodes_per_page
        if complete:
            rows = records[:complete].reshape(-1, self.nodes_per_page * self.record_bytes)
            pages = np.zeros((rows.shape[0], PAGE), dtype=np.uint8)
            pages[:, :rows.shape[1]] = rows
            self.stream.write(pages.tobytes())
        self.pending = records[complete:].copy()

    def close(self):
        if self.pending.shape[0]:
            page = np.zeros(PAGE, dtype=np.uint8)
            raw = self.pending.reshape(-1)
            page[:raw.size] = raw
            self.stream.write(page.tobytes())
        self.stream.close()


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-meta", required=True)
    parser.add_argument("--full-topo", required=True)
    parser.add_argument("--merge-meta", required=True)
    parser.add_argument("--merge-topo", required=True)
    parser.add_argument("--query-alpha", required=True)
    parser.add_argument("--threshold", required=True, type=float)
    parser.add_argument("--output-prefix", required=True)
    parser.add_argument("--chunk-nodes", type=int, default=5000)
    args = parser.parse_args()
    if not 0.0 <= args.threshold <= 1.0:
        raise SystemExit("threshold must be in [0,1]")

    full_meta = read_meta(args.full_meta)
    merge_meta = read_meta(args.merge_meta)
    if (full_meta["n"], full_meta["degree"]) != (merge_meta["n"], merge_meta["degree"]):
        raise SystemExit("Full and Merge node/degree metadata mismatch")
    if merge_meta["slots"] != 2:
        raise SystemExit("A2 source must have exactly two alpha-range slots")

    weights, query_count = read_alpha_weights(args.query_alpha)
    full_mm = np.memmap(args.full_topo, dtype=np.uint8, mode="r")
    merge_mm = np.memmap(args.merge_topo, dtype=np.uint8, mode="r")
    os.makedirs(os.path.dirname(os.path.abspath(args.output_prefix)), exist_ok=True)
    paths = {
        "a0": args.output_prefix + ".a0.topo",
        "a2": args.output_prefix + ".a2.topo",
        "map": args.output_prefix + ".node-map",
        "manifest": args.output_prefix + ".manifest.json",
    }
    a0_bytes = 4 + full_meta["degree"] * 4
    a2_bytes = merge_meta["record_bytes"]
    a0_npp, a2_npp = PAGE // a0_bytes, PAGE // a2_bytes
    a0_writer = PagedWriter(paths["a0"], a0_bytes, a0_npp)
    a2_writer = PagedWriter(paths["a2"], a2_bytes, a2_npp)
    map_stream = open(paths["map"], "wb")
    if full_meta["n"] >= (1 << 31):
        raise SystemExit("split-v2 supports fewer than 2^31 nodes per class")
    map_stream.write(MAP_HEADER.pack(MAP_MAGIC, 2, 4, full_meta["n"]))
    a0_count = a2_count = mismatches = 0

    for start in range(0, full_meta["n"], args.chunk_nodes):
        count = min(args.chunk_nodes, full_meta["n"] - start)
        full_records = read_records(full_mm, full_meta, start, count)
        merge_records = read_records(merge_mm, merge_meta, start, count)
        full_degree, full_neighbors, full_ranges = parse(full_records, full_meta)
        merge_degree, merge_neighbors, _ = parse(merge_records, merge_meta)
        valid = np.arange(full_meta["degree"])[None, :] < full_degree[:, None]
        mismatches += int(np.count_nonzero(full_degree != merge_degree))
        mismatches += int(np.count_nonzero((full_neighbors != merge_neighbors) & valid))
        choose_a0 = node_filter_benefit(full_degree, full_ranges, weights) < args.threshold

        encoded = np.empty(count, dtype="<u4")
        n0 = int(choose_a0.sum())
        n2 = count - n0
        if n0:
            local = np.arange(a0_count, a0_count + n0, dtype=np.uint32)
            encoded[choose_a0] = local
            a0_records = np.zeros((n0, a0_bytes), dtype=np.uint8)
            selected_degree = merge_degree[choose_a0].astype("<u4")
            a0_records[:, :4] = selected_degree.view(np.uint8).reshape(n0, 4)
            selected_neighbors = np.ascontiguousarray(merge_neighbors[choose_a0].astype("<u4"))
            a0_records[:, 4:] = selected_neighbors.view(np.uint8).reshape(n0, -1)
            a0_writer.append(a0_records)
        if n2:
            local = np.arange(a2_count, a2_count + n2, dtype=np.uint32)
            encoded[~choose_a0] = CLASS_BIT | local
            a2_writer.append(merge_records[~choose_a0])
        map_stream.write(encoded.tobytes())
        a0_count += n0
        a2_count += n2
        if start and start % 100000 == 0:
            print("converted {}/{} nodes".format(start, full_meta["n"]), flush=True)

    a0_writer.close()
    a2_writer.close()
    map_stream.close()
    if mismatches:
        raise SystemExit("conversion found {} Full/Merge degree or neighbor mismatches".format(mismatches))
    if a0_writer.count != a0_count or a2_writer.count != a2_count:
        raise SystemExit("writer count mismatch")

    expected_a0_size = ((a0_count + a0_npp - 1) // a0_npp) * PAGE
    expected_a2_size = ((a2_count + a2_npp - 1) // a2_npp) * PAGE
    expected_map_size = MAP_HEADER.size + full_meta["n"] * 4
    actual_sizes = {name: os.path.getsize(path) for name, path in paths.items() if name != "manifest"}
    expected_sizes = {"a0": expected_a0_size, "a2": expected_a2_size, "map": expected_map_size}
    if actual_sizes != expected_sizes:
        raise SystemExit("size validation failed: actual={} expected={}".format(actual_sizes, expected_sizes))

    result = {
        "format": "hvs-gated-a0-a2-split-v2",
        "threshold": args.threshold,
        "query_count": query_count,
        "nodes": full_meta["n"],
        "degree": full_meta["degree"],
        "a0": {"nodes": a0_count, "record_bytes": a0_bytes, "nodes_per_page": a0_npp,
               "pages": expected_a0_size // PAGE, "bytes": expected_a0_size},
        "a2": {"nodes": a2_count, "record_bytes": a2_bytes, "nodes_per_page": a2_npp,
               "pages": expected_a2_size // PAGE, "bytes": expected_a2_size},
        "topology_bytes": expected_a0_size + expected_a2_size,
        "map_bytes": expected_map_size,
        "degree_or_neighbor_mismatches": mismatches,
        "sources": {key: os.path.abspath(getattr(args, key.replace("-", "_")))
                    for key in ("full_meta", "full_topo", "merge_meta", "merge_topo", "query_alpha")},
        "sha256": {name: sha256(path) for name, path in paths.items() if name != "manifest"},
    }
    with open(paths["manifest"], "w") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
