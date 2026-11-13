#!/usr/bin/env python3
import argparse
import json
import os
import struct
import sys

import numpy as np


PAGE = 8192


def read_meta(path):
    with open(path, "rb") as f:
        raw = f.read(PAGE)
    n, d1, d2, degree, slots = struct.unpack_from("<IIIII", raw, 0)
    npp = struct.unpack_from("<Q", raw, 20)[0]
    ep = struct.unpack_from("<I", raw, 28)[0]
    return {"n": n, "d1": d1, "d2": d2, "degree": degree,
            "slots": slots, "npp": npp, "ep": ep,
            "record_bytes": 4 + degree * (4 + 2 * slots)}


def read_alpha_hist(path):
    with open(path, "rb") as f:
        q, dim = struct.unpack("<II", f.read(8))
        if dim != 1:
            raise ValueError("query alpha must be Qx1")
        alpha = np.fromfile(f, dtype="<f4", count=q)
    buckets = np.rint(alpha * 100.0).astype(np.int64)
    buckets = np.clip(buckets, 0, 100)
    return np.bincount(buckets, minlength=101), alpha


def read_records(mm, meta, start, count):
    ids = np.arange(start, start + count, dtype=np.int64)
    offsets = (ids // meta["npp"]) * PAGE + (ids % meta["npp"]) * meta["record_bytes"]
    byte_idx = offsets[:, None] + np.arange(meta["record_bytes"], dtype=np.int64)[None, :]
    return mm[byte_idx]


def parse_records(records, meta):
    count = records.shape[0]
    degree = meta["degree"]
    slots = meta["slots"]
    deg = records[:, :4].copy().view("<u4").reshape(count)
    body = records[:, 4:].reshape(count, degree, 4 + 2 * slots)
    nbr = body[:, :, :4].copy().view("<u4").reshape(count, degree)
    ranges = body[:, :, 4:].view(np.int8).reshape(count, degree, slots, 2)
    edge_valid = np.arange(degree)[None, :] < deg[:, None]
    return deg, nbr, ranges, edge_valid


def interval_stats(ranges, edge_valid, qprefix):
    flat = ranges.reshape(-1, ranges.shape[2], 2)
    valid_edges = edge_valid.reshape(-1)
    flat = flat[valid_edges]
    range_count = np.zeros(flat.shape[0], dtype=np.uint8)
    active_weight = np.zeros(flat.shape[0], dtype=np.float64)
    valid_slots = np.zeros((flat.shape[0], flat.shape[1]), dtype=np.bool_)
    for slot in range(flat.shape[1]):
        lo = flat[:, slot, 0].astype(np.int16)
        hi = flat[:, slot, 1].astype(np.int16)
        valid = ((lo != 0) | (hi != 0)) & (lo >= 0) & (hi >= lo) & (hi <= 100)
        valid_slots[:, slot] = valid
        range_count += valid.astype(np.uint8)
        if valid.any():
            idx = np.nonzero(valid)[0]
            active_weight[idx] += qprefix[hi[idx] + 1] - qprefix[lo[idx]]
    overlap_pairs = 0
    overlap_edges = np.zeros(flat.shape[0], dtype=np.bool_)
    for left in range(flat.shape[1]):
        for right in range(left + 1, flat.shape[1]):
            both = valid_slots[:, left] & valid_slots[:, right]
            if both.any():
                overlapping = both & (flat[:, right, 0] <= flat[:, left, 1])
                overlap_pairs += int(np.count_nonzero(overlapping))
                overlap_edges |= overlapping
    if overlap_edges.any():
        selected = flat[overlap_edges]
        cover = np.zeros((selected.shape[0], 101), dtype=np.bool_)
        buckets = np.arange(101, dtype=np.int16)[None, :]
        for slot in range(selected.shape[1]):
            lo = selected[:, slot, 0].astype(np.int16)
            hi = selected[:, slot, 1].astype(np.int16)
            valid = ((lo != 0) | (hi != 0)) & (lo >= 0) & (hi >= lo) & (hi <= 100)
            cover |= valid[:, None] & (buckets >= lo[:, None]) & (buckets <= hi[:, None])
        qweights = np.diff(qprefix)
        active_weight[overlap_edges] = cover.dot(qweights)
    return flat, valid_slots, active_weight, range_count, overlap_pairs


def sample_signatures(target, flat, limit, seen):
    remaining = limit - seen
    if remaining <= 0:
        return seen
    take = min(remaining, flat.shape[0])
    rows = np.ascontiguousarray(flat[:take].reshape(take, -1))
    row_type = np.dtype((np.void, rows.dtype.itemsize * rows.shape[1]))
    unique = np.unique(rows.view(row_type).reshape(-1))
    target.update(item.tobytes() for item in unique)
    return seen + take


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--full-meta", required=True)
    ap.add_argument("--full-topo", required=True)
    ap.add_argument("--merge-meta", required=True)
    ap.add_argument("--merge-topo", required=True)
    ap.add_argument("--query-alpha", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--chunk-nodes", type=int, default=5000)
    args = ap.parse_args()

    fm = read_meta(args.full_meta)
    mm = read_meta(args.merge_meta)
    if fm["n"] != mm["n"] or fm["degree"] != mm["degree"]:
        raise SystemExit("Full/Merge metadata mismatch")
    qhist, alpha = read_alpha_hist(args.query_alpha)
    qweights = qhist.astype(np.float64) / qhist.sum()
    qprefix = np.concatenate(([0.0], np.cumsum(qweights)))
    full_map = np.memmap(args.full_topo, dtype=np.uint8, mode="r")
    merge_map = np.memmap(args.merge_topo, dtype=np.uint8, mode="r")

    full_hist = np.zeros(fm["slots"] + 1, dtype=np.int64)
    merge_hist = np.zeros(mm["slots"] + 1, dtype=np.int64)
    full_node_max_hist = np.zeros(fm["slots"] + 1, dtype=np.int64)
    merge_node_max_hist = np.zeros(mm["slots"] + 1, dtype=np.int64)
    edges = 0
    neighbor_mismatches = 0
    full_active_sum = 0.0
    merge_active_sum = 0.0
    fp_sum = 0.0
    fn_sum = 0.0
    full_overlap_pairs = 0
    merge_overlap_pairs = 0
    full_signatures = set()
    merge_signatures = set()
    signature_sample_limit = sys.maxsize
    signature_sampled = 0
    benefit_thresholds = (0.05, 0.10, 0.20, 0.30, 0.40, 0.50)
    gated = {threshold: {"a0_nodes": 0, "added_active_edges": 0.0}
             for threshold in benefit_thresholds}

    for start in range(0, fm["n"], args.chunk_nodes):
        count = min(args.chunk_nodes, fm["n"] - start)
        fr = read_records(full_map, fm, start, count)
        mr = read_records(merge_map, mm, start, count)
        fdeg, fnbr, franges, fvalid = parse_records(fr, fm)
        mdeg, mnbr, mranges, mvalid = parse_records(mr, mm)
        if not np.array_equal(fdeg, mdeg):
            raise SystemExit("degree mismatch starting at node {}".format(start))
        neighbor_mismatches += int(np.count_nonzero((fnbr != mnbr) & fvalid))
        ff, fslots, fw, frc, foverlap = interval_stats(franges, fvalid, qprefix)
        mf, mslots, mw, mrc, moverlap = interval_stats(mranges, mvalid, qprefix)
        if ff.shape[0] != mf.shape[0]:
            raise SystemExit("edge-count mismatch starting at node {}".format(start))
        edges += ff.shape[0]
        full_hist += np.bincount(frc, minlength=len(full_hist))[:len(full_hist)]
        merge_hist += np.bincount(mrc, minlength=len(merge_hist))[:len(merge_hist)]
        fstarts = np.concatenate(([0], np.cumsum(fdeg, dtype=np.int64)))
        mstarts = np.concatenate(([0], np.cumsum(mdeg, dtype=np.int64)))
        fnode_max = np.zeros(count, dtype=np.uint8)
        mnode_max = np.zeros(count, dtype=np.uint8)
        fnz = fdeg > 0
        mnz = mdeg > 0
        fnode_max[fnz] = np.maximum.reduceat(frc, fstarts[:-1][fnz])
        mnode_max[mnz] = np.maximum.reduceat(mrc, mstarts[:-1][mnz])
        full_node_max_hist += np.bincount(
            fnode_max, minlength=len(full_node_max_hist))[:len(full_node_max_hist)]
        merge_node_max_hist += np.bincount(
            mnode_max, minlength=len(merge_node_max_hist))[:len(merge_node_max_hist)]
        node_active = np.zeros(count, dtype=np.float64)
        node_active[fnz] = np.add.reduceat(fw, fstarts[:-1][fnz])
        node_filter = np.zeros(count, dtype=np.float64)
        node_filter[fnz] = 1.0 - node_active[fnz] / fdeg[fnz]
        node_filtered_edges = fdeg.astype(np.float64) - node_active
        for threshold in benefit_thresholds:
            choose_a0 = node_filter < threshold
            gated[threshold]["a0_nodes"] += int(np.count_nonzero(choose_a0))
            gated[threshold]["added_active_edges"] += float(
                node_filtered_edges[choose_a0].sum())
        full_active_sum += float(fw.sum())
        merge_active_sum += float(mw.sum())
        fp_sum += float(np.maximum(mw - fw, 0.0).sum())
        fn_sum += float(np.maximum(fw - mw, 0.0).sum())
        full_overlap_pairs += foverlap
        merge_overlap_pairs += moverlap
        before = signature_sampled
        signature_sampled = sample_signatures(full_signatures, ff, signature_sample_limit,
                                              signature_sampled)
        sample_signatures(merge_signatures, mf, signature_sample_limit, before)
        if start and start % 100000 == 0:
            print("{}: {}/{} nodes".format(args.dataset, start, fm["n"]), flush=True)

    full_fixed_alpha = fm["n"] * fm["degree"] * fm["slots"] * 2
    merge_fixed_alpha = mm["n"] * mm["degree"] * mm["slots"] * 2
    full_used_alpha = sum(i * 2 * int(v) for i, v in enumerate(full_hist))
    merge_used_alpha = sum(i * 2 * int(v) for i, v in enumerate(merge_hist))
    gated_results = []
    for threshold in benefit_thresholds:
        a0_nodes = gated[threshold]["a0_nodes"]
        a2_nodes = fm["n"] - a0_nodes
        pages = ((a0_nodes + 48) // 49) + ((a2_nodes + 24) // 25)
        gated_results.append({
            "filter_threshold": threshold,
            "a0_nodes": a0_nodes,
            "a0_node_fraction": a0_nodes / fm["n"],
            "a2_nodes": a2_nodes,
            "topology_bytes": pages * PAGE,
            "effective_nodes_per_page": fm["n"] / pages,
            "added_active_edge_probability":
                gated[threshold]["added_active_edges"] / edges,
        })
    result = {
        "dataset": args.dataset,
        "nodes": fm["n"],
        "degree": fm["degree"],
        "edges": edges,
        "query_count": int(qhist.sum()),
        "query_alpha_min": float(alpha.min()),
        "query_alpha_max": float(alpha.max()),
        "full": {
            "slots": fm["slots"], "nodes_per_page": fm["npp"],
            "record_bytes": fm["record_bytes"],
            "range_count_hist": full_hist.tolist(),
            "node_max_range_count_hist": full_node_max_hist.tolist(),
            "fixed_alpha_bytes": full_fixed_alpha,
            "used_alpha_bytes": full_used_alpha,
            "slot_utilization": full_used_alpha / full_fixed_alpha,
            "query_weighted_active_probability": full_active_sum / edges,
            "query_weighted_filter_probability": 1.0 - full_active_sum / edges,
            "overlapping_interval_pairs": full_overlap_pairs,
        },
        "merge": {
            "slots": mm["slots"], "nodes_per_page": mm["npp"],
            "record_bytes": mm["record_bytes"],
            "range_count_hist": merge_hist.tolist(),
            "node_max_range_count_hist": merge_node_max_hist.tolist(),
            "fixed_alpha_bytes": merge_fixed_alpha,
            "used_alpha_bytes": merge_used_alpha,
            "slot_utilization": merge_used_alpha / merge_fixed_alpha,
            "query_weighted_active_probability": merge_active_sum / edges,
            "query_weighted_filter_probability": 1.0 - merge_active_sum / edges,
            "overlapping_interval_pairs": merge_overlap_pairs,
        },
        "merge_vs_full": {
            "neighbor_mismatches": neighbor_mismatches,
            "false_positive_active_probability": fp_sum / edges,
            "false_negative_active_probability": fn_sum / edges,
        },
        "benefit_gated_a0_a2": gated_results,
        "signature_edges": signature_sampled,
        "full_unique_range_tuple_signatures": len(full_signatures),
        "merge_unique_range_tuple_signatures": len(merge_signatures),
    }
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
