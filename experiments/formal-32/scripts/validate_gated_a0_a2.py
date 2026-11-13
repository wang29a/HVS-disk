#!/usr/bin/env python3
"""Independently read back and validate a split A0/A2 topology."""

import argparse
import hashlib
import json
import os
import struct

import numpy as np


PAGE = 8192
HEADER = struct.Struct("<8sIIQ")
CLASS_MASK = np.uint32(1 << 31)
ORDINAL_MASK = np.uint32((1 << 31) - 1)


def read_meta(path):
    with open(path, "rb") as stream:
        raw = stream.read(PAGE)
    n, _, _, degree, slots = struct.unpack_from("<IIIII", raw, 0)
    npp = struct.unpack_from("<Q", raw, 20)[0]
    return {"n": n, "degree": degree, "slots": slots, "npp": npp,
            "record_bytes": 4 + degree * (4 + 2 * slots)}


def read_records(mm, ids, record_bytes, npp):
    ids = np.asarray(ids, dtype=np.int64)
    offsets = (ids // npp) * PAGE + (ids % npp) * record_bytes
    indices = offsets[:, None] + np.arange(record_bytes, dtype=np.int64)
    return mm[indices]


def degree_neighbors(records, degree, slots):
    count = records.shape[0]
    degrees = records[:, :4].copy().view("<u4").reshape(count)
    body = records[:, 4:].reshape(count, degree, 4 + 2 * slots)
    neighbors = body[:, :, :4].copy().view("<u4").reshape(count, degree)
    return degrees, neighbors


def check_padding(mm, record_bytes, npp, node_count, label):
    if mm.size % PAGE:
        raise ValueError(label + " is not page aligned")
    pages = mm.reshape(-1, PAGE)
    used = record_bytes * npp
    if used < PAGE and np.any(pages[:, used:]):
        raise ValueError(label + " has nonzero inter-page padding")
    tail = node_count % npp
    if tail and np.any(pages[-1, tail * record_bytes:used]):
        raise ValueError(label + " has nonzero final-record padding")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-meta", required=True)
    parser.add_argument("--merge-topo", required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--chunk-nodes", type=int, default=10000)
    args = parser.parse_args()

    manifest_path = args.prefix + ".manifest.json"
    with open(manifest_path) as stream:
        manifest = json.load(stream)
    meta = read_meta(args.merge_meta)
    if meta["slots"] != 2 or manifest["nodes"] != meta["n"]:
        raise SystemExit("manifest/source metadata mismatch")

    paths = {"a0": args.prefix + ".a0.topo", "a2": args.prefix + ".a2.topo",
             "map": args.prefix + ".node-map"}
    with open(paths["map"], "rb") as stream:
        magic, version, width, nodes = HEADER.unpack(stream.read(HEADER.size))
    if (magic, version, width, nodes) != (b"HVSGAT02", 2, 4, meta["n"]):
        raise SystemExit("invalid node-map header")
    mapping = np.memmap(paths["map"], dtype="<u4", mode="r", offset=HEADER.size,
                        shape=(meta["n"],))
    is_a2 = (mapping & CLASS_MASK) != 0
    ordinal = mapping & ORDINAL_MASK
    a0_count = int((~is_a2).sum())
    a2_count = int(is_a2.sum())
    if (a0_count, a2_count) != (manifest["a0"]["nodes"], manifest["a2"]["nodes"]):
        raise SystemExit("map class counts differ from manifest")
    if not np.array_equal(ordinal[~is_a2], np.arange(a0_count, dtype=np.uint32)):
        raise SystemExit("A0 ordinals are not a bijective dense sequence")
    if not np.array_equal(ordinal[is_a2], np.arange(a2_count, dtype=np.uint32)):
        raise SystemExit("A2 ordinals are not a bijective dense sequence")

    a0_mm = np.memmap(paths["a0"], dtype=np.uint8, mode="r")
    a2_mm = np.memmap(paths["a2"], dtype=np.uint8, mode="r")
    source_mm = np.memmap(args.merge_topo, dtype=np.uint8, mode="r")
    check_padding(a0_mm, manifest["a0"]["record_bytes"], manifest["a0"]["nodes_per_page"],
                  a0_count, "A0 topology")
    check_padding(a2_mm, manifest["a2"]["record_bytes"], manifest["a2"]["nodes_per_page"],
                  a2_count, "A2 topology")

    mismatches = 0
    for start in range(0, meta["n"], args.chunk_nodes):
        count = min(args.chunk_nodes, meta["n"] - start)
        global_ids = np.arange(start, start + count, dtype=np.int64)
        source = read_records(source_mm, global_ids, meta["record_bytes"], meta["npp"])
        source_degree, source_neighbors = degree_neighbors(source, meta["degree"], 2)
        local_class = is_a2[start:start + count]
        local_ord = ordinal[start:start + count]
        for class_value, mm, record_bytes, npp, slots in (
                (False, a0_mm, manifest["a0"]["record_bytes"],
                 manifest["a0"]["nodes_per_page"], 0),
                (True, a2_mm, manifest["a2"]["record_bytes"],
                 manifest["a2"]["nodes_per_page"], 2)):
            selected = local_class == class_value
            if not selected.any():
                continue
            records = read_records(mm, local_ord[selected], record_bytes, npp)
            degrees, neighbors = degree_neighbors(records, meta["degree"], slots)
            expected_degree = source_degree[selected]
            expected_neighbors = source_neighbors[selected]
            valid = np.arange(meta["degree"])[None, :] < expected_degree[:, None]
            mismatches += int(np.count_nonzero(degrees != expected_degree))
            mismatches += int(np.count_nonzero((neighbors != expected_neighbors) & valid))
        if start and start % 500000 == 0:
            print("validated {}/{} nodes".format(start, meta["n"]), flush=True)
    if mismatches:
        raise SystemExit("read-back degree/neighbor mismatches: {}".format(mismatches))

    hashes = {name: sha256(path) for name, path in paths.items()}
    if hashes != manifest["sha256"]:
        raise SystemExit("output hashes differ from manifest")
    print(json.dumps({"prefix": args.prefix, "nodes": meta["n"], "a0_nodes": a0_count,
                      "a2_nodes": a2_count, "readback_mismatches": mismatches,
                      "map_bijection": True, "page_padding_zero": True,
                      "sha256_match": True}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
