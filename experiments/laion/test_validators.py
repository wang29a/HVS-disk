#!/usr/bin/env python3
"""Exercise partial pages, empty/max degree, interval coverage and corruption."""
import argparse
import json
import os
import struct
import subprocess
import sys
import tempfile
import numpy as np

PAGE = 8192
N, D1, D2, R = 257, 4, 4, 3


def write_meta(path, slots, single=False):
    width = 4 + R * (4 + 2 * slots)
    raw = bytearray(PAGE)
    struct.pack_into("<IIIIIQI", raw, 0, N, D1, D2, R, slots,
                     0 if single else PAGE // width, 1)
    struct.pack_into("<I", raw, 32, 0)
    with open(path, "wb") as f:
        f.write(raw)


def make_records(slots):
    rows = []
    width = 4 + R * (4 + 2 * slots)
    for node in range(N):
        row = bytearray(width)
        degree = node % (R + 1)
        struct.pack_into("<I", row, 0, degree)
        for edge in range(degree):
            off = 4 + edge * (4 + slots * 2)
            struct.pack_into("<I", row, off, (node + edge + 1) % N)
            ranges = [(0, 20), (40, 60), (80, 100)] if slots == 3 else [(0, 60), (80, 100)]
            for j, (lo, hi) in enumerate(ranges[:slots]):
                struct.pack_into("<bb", row, off + 4 + j * 2, lo, hi)
        rows.append(bytes(row))
    return rows


def paged(path, rows):
    width = len(rows[0])
    npp = PAGE // width
    with open(path, "wb") as f:
        for first in range(0, len(rows), npp):
            payload = b"".join(rows[first:first + npp])
            f.write(payload + bytes(PAGE - len(payload)))


def run(validator, mode, args, good=True):
    p = subprocess.run([sys.executable, validator, mode] + args,
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if (p.returncode == 0) != good:
        raise AssertionError(p.stdout.decode())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--validator", required=True)
    a = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="hvs-laion-validator-") as root:
        b1 = np.arange(N * D1, dtype="<f4").reshape(N, D1) / 100
        b2 = -np.arange(N * D2, dtype="<f4").reshape(N, D2) / 100
        for name, data in (("base1", b1), ("base2", b2)):
            with open(root + "/" + name, "wb") as f:
                f.write(struct.pack("<II", N, data.shape[1]))
                f.write(data.tobytes())
        all_rows = {}
        for kind, slots in (("full", 3), ("merge", 2), ("no-alpha", 0)):
            directory = root + "/" + kind
            os.mkdir(directory)
            write_meta(directory + "/_disk.index", slots)
            rows = make_records(slots)
            all_rows[kind] = rows
            paged(directory + "/disk_index_graph.original", rows)
            npp = PAGE // len(rows[0])
            order = list(reversed(range(N)))
            groups = [order[i:i + npp] for i in range(0, N, npp)]
            mapping = [0] * N
            with open(directory + "/partition.bin", "wb") as raw, open(directory + "/partition.bin.aligned", "wb") as aligned:
                for f in (raw, aligned):
                    f.write(struct.pack("<QQQ", npp, len(groups), N))
                for page, group in enumerate(groups):
                    for node in group:
                        mapping[node] = page
                    raw.write(struct.pack("<I", len(group)))
                    raw.write(struct.pack("<{}I".format(len(group)), *group))
                    aligned.write(struct.pack("<I", len(group)))
                    padded = group + [0xffffffff] * (npp - len(group))
                    aligned.write(struct.pack("<{}I".format(npp), *padded))
                raw.write(struct.pack("<{}I".format(N), *mapping))
            paged(directory + "/disk_index_graph.relayout", [rows[i] for i in order])
        coord_rows = [b1[i].tobytes() + b2[i].tobytes() for i in range(N)]
        paged(root + "/coords", coord_rows)
        write_meta(root + "/single", 3, single=True)
        with open(root + "/single", "ab") as f:
            for coord, row in zip(coord_rows, all_rows["full"]):
                payload = coord + row
                f.write(payload + bytes(PAGE - len(payload)))
        output = root + "/result.json"
        source_args = ["--meta", root + "/full/_disk.index", "--topo", root + "/full/disk_index_graph.original",
                       "--coord", root + "/coords", "--single", root + "/single", "--base1", root + "/base1",
                       "--base2", root + "/base2", "--chunk", "31", "--output", output]
        run(a.validator, "source", source_args)
        run(a.validator, "converted", ["--root", root, "--output", output])
        for kind in all_rows:
            run(a.validator, "relayout", ["--root", root, "--kind", kind, "--output", output])
        with open(root + "/single", "r+b") as f:
            f.seek(PAGE)
            original = f.read(1)
            f.seek(PAGE)
            f.write(bytes([original[0] ^ 1]))
        run(a.validator, "source", source_args, good=False)
        with open(root + "/merge/disk_index_graph.original", "r+b") as f:
            # Node 1's first valid neighbor; the replacement remains a legal ID.
            f.seek(len(all_rows["merge"][0]) + 4)
            f.write(struct.pack("<I", 99))
        run(a.validator, "converted", ["--root", root, "--output", output], good=False)
        with open(root + "/full/partition.bin.aligned", "r+b") as f:
            f.seek(28)
            f.write(struct.pack("<I", N - 2))
        run(a.validator, "relayout", ["--root", root, "--kind", "full", "--output", output], good=False)
        # A missing original interval must be rejected even with identical neighbors.
        paged(root + "/merge/disk_index_graph.original", all_rows["merge"])
        with open(root + "/merge/disk_index_graph.original", "r+b") as f:
            f.seek(len(all_rows["merge"][0]) + 8)
            f.write(struct.pack("<bb", 0, 10))
        run(a.validator, "converted", ["--root", root, "--output", output], good=False)
    print("PASS: source/conversion/three relayouts; coordinate, neighbor, partition and coverage corruptions rejected")


if __name__ == "__main__":
    main()
