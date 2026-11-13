#!/usr/bin/env python3
import argparse
import os
import struct
import sys


def read_exact(f, size):
    data = f.read(size)
    if len(data) != size:
        raise EOFError(f"wanted {size} bytes, got {len(data)} at offset {f.tell()}")
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mem", required=True)
    ap.add_argument("--single", required=True)
    ap.add_argument("--base1")
    ap.add_argument("--base2")
    ap.add_argument("--limit", type=int, default=0,
                    help="0 validates every node")
    args = ap.parse_args()

    with open(args.mem, "rb") as mem, open(args.single, "rb") as single:
        n_mem, r_mem, a_mem = struct.unpack("<QII", read_exact(mem, 16))
        header = read_exact(single, 32)
        n, d1, d2, degree, alpha_slots = struct.unpack_from("<IIIII", header, 0)
        nodes_per_page = struct.unpack_from("<Q", header, 20)[0]
        ep_size = struct.unpack_from("<I", header, 28)[0]
        if (n, degree, alpha_slots) != (n_mem, r_mem, a_mem):
            raise SystemExit(
                f"header mismatch mem={(n_mem,r_mem,a_mem)} single={(n,degree,alpha_slots)}")
        coord_bytes = (d1 + d2) * 4
        record_bytes = 4 + degree * (4 + 2 * alpha_slots)
        row_bytes = ((coord_bytes + record_bytes + 8191) // 8192) * 8192
        expected_size = 8192 + n * row_bytes
        actual_size = os.path.getsize(args.single)
        if expected_size != actual_size:
            raise SystemExit(f"size mismatch expected={expected_size} actual={actual_size}")
        count = n if args.limit == 0 else min(n, args.limit)
        b1 = open(args.base1, "rb") if args.base1 else None
        b2 = open(args.base2, "rb") if args.base2 else None
        try:
            if b1:
                bn1, bd1 = struct.unpack("<II", read_exact(b1, 8))
                bn2, bd2 = struct.unpack("<II", read_exact(b2, 8))
                if (bn1, bd1, bn2, bd2) != (n, d1, n, d2):
                    raise SystemExit("base headers do not match single-index header")
            for node in range(count):
                single.seek(8192 + node * row_bytes)
                coords = read_exact(single, coord_bytes)
                if b1:
                    expected_coords = read_exact(b1, d1 * 4) + read_exact(b2, d2 * 4)
                    if coords != expected_coords:
                        raise SystemExit(f"coordinate mismatch at node {node}")
                topo = read_exact(single, record_bytes)
                single_degree = struct.unpack_from("<I", topo, 0)[0]
                mem_degree = struct.unpack("<I", read_exact(mem, 4))[0]
                if mem_degree > degree:
                    raise SystemExit(f"mem degree {mem_degree} exceeds max {degree} at node {node}")
                if single_degree != mem_degree:
                    raise SystemExit(
                        f"degree mismatch at node {node}: mem={mem_degree} single={single_degree}")
                stride = 4 + 2 * alpha_slots
                for edge in range(mem_degree):
                    nbr, nranges = struct.unpack("<II", read_exact(mem, 8))
                    ranges = read_exact(mem, nranges * 2)
                    off = 4 + edge * stride
                    single_nbr = struct.unpack_from("<I", topo, off)[0]
                    slots = topo[off + 4:off + stride]
                    if nbr != single_nbr:
                        raise SystemExit(
                            f"neighbor mismatch node={node} edge={edge}: mem={nbr} single={single_nbr}")
                    if nranges > alpha_slots:
                        raise SystemExit(
                            f"range overflow node={node} edge={edge}: {nranges}>{alpha_slots}")
                    if slots[:len(ranges)] != ranges:
                        raise SystemExit(f"range mismatch node={node} edge={edge}")
                    if any(slots[len(ranges):]):
                        raise SystemExit(f"nonzero range padding node={node} edge={edge}")
                if node and node % 100000 == 0:
                    print(f"validated {node}/{count}", flush=True)
            print(
                f"PASS nodes={count} dims={d1}+{d2} R={degree} A={alpha_slots} "
                f"row={row_bytes} nodes_per_page={nodes_per_page} ep={ep_size}")
        finally:
            if b1:
                b1.close()
                b2.close()


if __name__ == "__main__":
    try:
        main()
    except (EOFError, OSError, struct.error) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
