#!/usr/bin/env python3
"""Independent, chunked validators for the Laion experiment artifacts."""
import argparse
import json
import os
import struct
import numpy as np

PAGE = 8192


def meta(path, single=False):
    with open(path, "rb") as f:
        raw = f.read(PAGE)
    if len(raw) != PAGE:
        raise ValueError("metadata must be a full 8192-byte page: " + path)
    n, d1, d2, degree, slots, npp, ep = struct.unpack_from("<IIIIIQI", raw)
    if 32 + 4 * ep > PAGE:
        raise ValueError("invalid entrypoint count")
    entries = list(struct.unpack_from("<{}I".format(ep), raw, 32))
    width = 4 + degree * (4 + slots * 2)
    if not n or width > PAGE or any(i >= n for i in entries):
        raise ValueError("invalid metadata")
    if (single and npp != 0) or (not single and npp != PAGE // width):
        raise ValueError("metadata nodes/page does not match layout")
    return dict(n=n, d1=d1, d2=d2, degree=degree, slots=slots,
                npp=npp, width=width, entries=entries)


def matrix(path, dtype="<f4"):
    with open(path, "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    if os.path.getsize(path) != 8 + n * d * np.dtype(dtype).itemsize:
        raise ValueError("matrix length mismatch: " + path)
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=(n, d))


def mm(path, expected):
    if os.path.getsize(path) != expected:
        raise ValueError("file size mismatch: " + path)
    return np.memmap(path, mode="r", dtype=np.uint8)


def records(data, ids, npp, width, header=0):
    ids = np.asarray(ids, dtype=np.int64)
    offsets = header + ids // npp * PAGE + ids % npp * width
    return data[offsets[:, None] + np.arange(width, dtype=np.int64)]


def parse(rows, m):
    count = len(rows)
    degrees = rows[:, :4].copy().view("<u4").reshape(count)
    body = rows[:, 4:].reshape(count, m["degree"], 4 + 2 * m["slots"])
    neighbors = body[:, :, :4].copy().view("<u4").reshape(count, m["degree"])
    ranges = body[:, :, 4:].view(np.int8).reshape(count, m["degree"], m["slots"], 2)
    valid = np.arange(m["degree"])[None, :] < degrees[:, None]
    if np.any(degrees > m["degree"]) or np.any(neighbors[valid] >= m["n"]):
        raise ValueError("invalid degree or neighbor ID")
    if m["slots"]:
        r = ranges[valid]
        if np.any(r[:, :, 0] < 0) or np.any(r[:, :, 1] > 100) or np.any(r[:, :, 0] > r[:, :, 1]):
            raise ValueError("invalid alpha interval")
    return degrees, neighbors, ranges, valid


def source(args):
    m = meta(args.meta)
    s = meta(args.single, single=True)
    if any(m[k] != s[k] for k in ("n", "d1", "d2", "degree", "slots", "entries")):
        raise ValueError("baseline and Full metadata differ")
    b1, b2 = matrix(args.base1), matrix(args.base2)
    if b1.shape != (m["n"], m["d1"]) or b2.shape != (m["n"], m["d2"]):
        raise ValueError("base dimensions differ from index")
    coord_width = 4 * (m["d1"] + m["d2"])
    cnpp = PAGE // coord_width
    row_width = coord_width + m["width"]
    # The current baseline uses one aligned record per 8192B page.
    if row_width > PAGE or not cnpp:
        raise ValueError("unsupported single-file record width")
    graph = mm(args.topo, ((m["n"] + m["npp"] - 1) // m["npp"]) * PAGE)
    coords = mm(args.coord, ((m["n"] + cnpp - 1) // cnpp) * PAGE)
    single = mm(args.single, PAGE + m["n"] * PAGE)
    for start in range(0, m["n"], args.chunk):
        end = min(start + args.chunk, m["n"])
        ids = np.arange(start, end, dtype=np.int64)
        original = records(graph, ids, m["npp"], m["width"])
        parse(original, m)
        packed = records(single, ids, 1, row_width, PAGE)
        if not np.array_equal(original, packed[:, coord_width:]):
            raise ValueError("baseline Full-alpha record differs near node {}".format(start))
        separate = records(coords, ids, cnpp, coord_width)
        if not np.array_equal(separate, packed[:, :coord_width]):
            raise ValueError("baseline/separated coordinate mismatch near node {}".format(start))
        for block, offset in ((b1[start:end], 0), (b2[start:end], m["d1"] * 4)):
            if not np.isfinite(block).all():
                raise ValueError("non-finite base vector")
            raw = np.ascontiguousarray(block).view(np.uint8).reshape(end - start, -1)
            if not np.array_equal(raw, separate[:, offset:offset + raw.shape[1]]):
                raise ValueError("base coordinate mismatch near node {}".format(start))
        if end == m["n"] or end // 500000 != start // 500000:
            print("source validated {}/{}".format(end, m["n"]), flush=True)
    return dict(status="PASS", nodes=m["n"], metadata=m, baseline_record_mismatches=0,
                coordinate_mismatches=0, invalid_degree_neighbor_interval=0)


def converted(args):
    kinds = ("full", "merge", "no-alpha")
    ms = {k: meta(args.root + "/" + k + "/_disk.index") for k in kinds}
    for k in kinds:
        if any(ms[k][f] != ms["full"][f] for f in ("n", "d1", "d2", "degree", "entries")):
            raise ValueError("conversion metadata mismatch")
    if ms["merge"]["slots"] != 2 or ms["no-alpha"]["slots"] != 0:
        raise ValueError("wrong conversion slots")
    maps = {k: mm(args.root + "/" + k + "/disk_index_graph.original",
                  ((ms[k]["n"] + ms[k]["npp"] - 1) // ms[k]["npp"]) * PAGE) for k in kinds}
    n = ms["full"]["n"]
    for start in range(0, n, args.chunk):
        ids = np.arange(start, min(start + args.chunk, n), dtype=np.int64)
        parsed = {k: parse(records(maps[k], ids, ms[k]["npp"], ms[k]["width"]), ms[k])
                  for k in kinds}
        fd, fn, fr, valid = parsed["full"]
        for k in ("merge", "no-alpha"):
            d, nbr, _, _ = parsed[k]
            if not np.array_equal(fd, d) or np.any((fn != nbr) & valid):
                raise ValueError("conversion changed neighbors near node {}".format(start))
        mr = parsed["merge"][2]
        for slot in range(ms["full"]["slots"]):
            lo, hi = fr[:, :, slot, 0], fr[:, :, slot, 1]
            active = valid & ((lo != 0) | (hi != 0))
            contained = np.zeros_like(active)
            for j in range(2):
                contained |= (mr[:, :, j, 0] <= lo) & (mr[:, :, j, 1] >= hi)
            if np.any(active & ~contained):
                raise ValueError("Merge lost original interval coverage")
        end = int(ids[-1]) + 1
        if end == n or end // 500000 != start // 500000:
            print("conversions validated {}/{}".format(end, n), flush=True)
    return dict(status="PASS", nodes=n, neighbor_mismatches=0, merge_coverage_losses=0)


def relayout(args):
    root = args.root + "/" + args.kind
    m = meta(root + "/_disk.index")
    aligned_path = root + "/partition.bin.aligned"
    with open(aligned_path, "rb") as f:
        c, pages, n = struct.unpack("<QQQ", f.read(24))
    if c != m["npp"] or n != m["n"] or pages != (n + c - 1) // c:
        raise ValueError("partition header mismatch")
    if os.path.getsize(aligned_path) != 24 + pages * (c + 1) * 4:
        raise ValueError("aligned partition length mismatch")
    aligned = np.memmap(aligned_path, dtype="<u4", mode="r", offset=24, shape=(pages, c + 1))
    with open(root + "/partition.bin", "rb") as f:
        raw = f.read()
    if struct.unpack_from("<QQQ", raw) != (c, pages, n):
        raise ValueError("raw/aligned partition header differs")
    offset = 24
    for page in range(pages):
        size = struct.unpack_from("<I", raw, offset)[0]
        offset += 4
        if size != aligned[page, 0] or size > c:
            raise ValueError("partition page size differs")
        group = np.frombuffer(raw, dtype="<u4", count=size, offset=offset)
        if not np.array_equal(group, aligned[page, 1:1 + size]):
            raise ValueError("partition alignment changed IDs")
        offset += size * 4
    if offset + n * 4 != len(raw):
        raise ValueError("raw partition trailing length mismatch")
    id_to_page = np.frombuffer(raw, dtype="<u4", count=n, offset=offset)
    graph = mm(root + "/disk_index_graph.original", pages * PAGE)
    reordered = mm(root + "/disk_index_graph.relayout", pages * PAGE)
    seen = np.zeros(n, dtype=np.uint32)
    for first in range(0, pages, 512):
        last = min(first + 512, pages)
        block = aligned[first:last]
        sizes = block[:, 0]
        if np.any(sizes > c):
            raise ValueError("oversized partition")
        valid = np.arange(c)[None, :] < sizes[:, None]
        ids = block[:, 1:][valid]
        if np.any(ids >= n) or np.any(block[:, 1:][~valid] != 0xffffffff):
            raise ValueError("invalid partition ID or padding")
        page_ids = np.broadcast_to(np.arange(first, last)[:, None], valid.shape)[valid]
        if np.any(id_to_page[ids] != page_ids):
            raise ValueError("ID-to-partition map mismatch")
        np.add.at(seen, ids, 1)
        locations = (np.arange(first, last)[:, None] * c + np.arange(c)[None, :])[valid]
        old = records(graph, ids, c, m["width"])
        new = records(reordered, locations, c, m["width"])
        if not np.array_equal(old, new):
            raise ValueError("relayout changed a full topology record")
        parse(old, m)
        if last == pages or last // 50000 != first // 50000:
            print("{} relayout validated {}/{} pages".format(args.kind, last, pages), flush=True)
    if not np.all(seen == 1):
        raise ValueError("partition is not a complete bijection")
    return dict(status="PASS", kind=args.kind, nodes=n, pages=pages,
                record_mismatches=0, invalid_records=0, bijection=True)


def queries(args):
    b1, b2 = matrix(args.base1), matrix(args.base2)
    q1, q2, alpha = matrix(args.query1), matrix(args.query2), matrix(args.alpha)
    gt = matrix(args.gt, "<u4")
    q = len(q1)
    if len(q2) != q or alpha.shape != (q, 1) or len(gt) != q or gt.shape[1] < 10:
        raise ValueError("query/alpha/GT counts differ")
    if q1.shape[1] != b1.shape[1] or q2.shape[1] != b2.shape[1] or len(b1) != len(b2):
        raise ValueError("base/query dimensions mismatch")
    if not np.isfinite(alpha).all() or np.any(alpha < 0) or np.any(alpha > 1):
        raise ValueError("invalid query alpha")
    if np.any(gt >= len(b1)) or not np.isfinite(q1).all() or not np.isfinite(q2).all():
        raise ValueError("invalid query or ground-truth ID")
    checked = []
    for qi in sorted(set((0, q // 2, q - 1))):
        distances = np.empty(len(b1), dtype=np.float32)
        a = float(alpha[qi, 0])
        for start in range(0, len(b1), 16384):
            end = min(start + 16384, len(b1))
            e = b1[start:end] - q1[qi]
            l = b2[start:end] - q2[qi]
            de = np.einsum("ij,ij->i", e, e)
            dl = np.einsum("ij,ij->i", l, l)
            distances[start:end] = a * np.sqrt(de) + (1 - a) * np.sqrt(dl)
        threshold = float(np.partition(distances, 9)[9])
        truth_ids = np.asarray(gt[qi, :10])
        if len(np.unique(truth_ids)) != 10:
            raise ValueError("duplicate IDs in ground truth")
        tolerance = max(1e-6, abs(threshold) * 1e-5)
        accepted = int(np.count_nonzero(distances[truth_ids] <= threshold + tolerance))
        print("exact GT check query={} accepted={}/10".format(qi, accepted), flush=True)
        if accepted != 10:
            raise ValueError("ground truth disagrees with exact dual-space sqrt distance")
        checked.append(dict(query=qi, accepted=accepted, tenth_distance=threshold))
    return dict(status="PASS", queries=q, exact_gt_queries=checked,
                alpha_min=float(alpha.min()), alpha_max=float(alpha.max()))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("source", "converted", "relayout", "queries"))
    for name in ("meta", "topo", "coord", "single", "base1", "base2", "root", "kind",
                 "query1", "query2", "alpha", "gt"):
        p.add_argument("--" + name)
    p.add_argument("--chunk", type=int, default=4096)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    result = globals()[args.mode](args)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
