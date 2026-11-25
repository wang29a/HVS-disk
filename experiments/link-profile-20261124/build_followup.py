"""Construct the two K=8 variants from the frozen shard memberships."""

import argparse
import hashlib
from itertools import count
import os
from pathlib import Path
import re
import struct
import subprocess
import time
from types import SimpleNamespace

import numpy as np

from build import read, sha
from build_degree import compile_builder
from run import matrix, save, status
from timing_guard import guarded_call, quiet_window

N = 2000000
CORES = set(range(16, 24))
MERGE_CORES = set(range(48, 56))
MERGE_AFFINITY = ('numactl', '--all', '--physcpubind=48-55', '--membind=2')
BASES = ("base_img_emb.fvecs", "base_text_emb.fvecs")
VARIANTS = {"k8-by-degree": (200, True), "k8-L128": (128, False)}


def degrees(graph, n):
    with graph.open("rb") as source:
        if source.read(16) != struct.pack("<QII", n, 512, 512):
            raise RuntimeError("unexpected degree check fixture")
        for _ in range(n):
            d, = struct.unpack("<I", source.read(4))
            for _ in range(d):
                _, pairs = struct.unpack("<II", source.read(8))
                source.seek(8 * pairs, 1)
            yield d
        if source.read(1):
            raise RuntimeError("fixture has trailing bytes")


def fixture(root, linker, binary):
    out = root / "build" / "degree-fixture"
    out.mkdir(exist_ok=True)
    ref = Path(read(linker / "build/identity.json")["binaries"]["off"]["binary"])
    eq = read(linker / "build/identity.json")["binaries"]["off"]["sha256"]
    if sha(ref) != eq:
        raise RuntimeError("fixed-R control changed")
    files = tuple(linker / "data/legacy" / ("fixture-%d.bin" % j) for j in range(2))
    proof = read(linker / "analysis/inputs.json")["shards"]["legacy"]["material_sha256"]
    for p in files:
        if sha(p) != proof[p.name]:
            raise RuntimeError("frozen fixture vector changed")
    n = 2048
    cases = ((40, "fixed40"), (27, "fixed27"), (40, "mixed"), (40, "L128"))
    results = {}
    with (root / "logs/degree-fixture.log").open("w") as log:
        def run_case(exe, path, r, tail):
            args = (exe, "shard", *files, n, r, path, *tail)
            env = dict(os.environ, OMP_THREAD_LIMIT="1", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            subprocess.run(list(map(str, args)), check=True, env=env, stdout=log, stderr=subprocess.STDOUT)
            return sha(path)
        for r, case in cases:
            cap_file = out / (case + ".u8")
            caps = (np.where(np.arange(n) % 2 == 0, 40, 27) if case == "mixed"
                    else np.full(n, r)).astype("u1")
            caps.tofile(cap_file)
            expected = None
            if case.startswith("fixed"):
                expected = run_case(ref, out / (case + "-reference.bin"), r, ())
            output = out / (case + ".bin")
            subset = [] if case == "L128" else [cap_file]
            digest = run_case(binary, output, 40 if case == "fixed27" else r,
                              [128 if case == "L128" else 200, 750, *subset])
            values = np.fromiter(degrees(output, n), dtype=np.uint32, count=n)
            if (expected and expected != digest) or np.any(values > caps):
                raise RuntimeError("cap fixture failed: " + case)
            results[case] = dict(sha256=digest, parity=(expected == digest) if expected else None,
                                 max_by_cap={str(v): int(values[caps == v].max()) for v in set(caps)})
    save(root / "build/degree-parity.json", results)
    return results


def setup(root, linker, origin):
    proof_path = root / "build/degree-identity.json"
    parity_path = root / "build/degree-parity.json"
    if proof_path.exists() and parity_path.exists():
        proof = read(proof_path)
        binary = Path(proof["binary"])
        if sha(binary) != proof["sha256"] or sha(root / "source/shard_degree.h") != proof["cap_header_sha256"]:
            raise RuntimeError("degree builder identity changed")
        return binary
    binary = compile_builder(root, linker)
    fixture(root, linker, binary)
    return binary


def memberships(origin):
    entries = read(origin / "progress.json")
    counts = np.zeros(N, dtype=np.uint8)
    groups = []
    for i in range(8):
        previous = entries["paired-k8-paired-%d" % i]["result"]
        file = Path(previous["memberships"])
        identity = sha(file)
        ids = np.fromfile(file, "<u4")
        if (identity != previous["memberships_sha256"] or len(ids) != previous["nodes"] or
                np.any(ids[1:] <= ids[:-1]) or ids[-1] >= N):
            raise RuntimeError("frozen K=8 memberships changed")
        counts[ids] += 1
        groups.append((file, identity, ids))
    if not np.all(counts > 0):
        raise RuntimeError("missing K=8 member")
    return counts, groups


def prepare_inputs(run, origin):
    identity = read(origin / "progress.json")["INPUTS"]["result"]
    for name in BASES:
        path = origin / "inputs" / name
        if sha(path) != identity[name]:
            raise RuntimeError("frozen base coordinates changed")
        link = run.root / "inputs" / name
        if not link.is_symlink():
            link.symlink_to(path)
        if link.resolve() != path.resolve():
            raise RuntimeError("unexpected base coordinates symlink")
    return [matrix(run.root / "inputs" / p) for p in BASES]


def stage_shard(run, binary, directory, label, ids, link, link_sha, vectors, caps, L, measured):
    run.guard(extra=(2 * 512 * 4 + 1024) * len(ids), memory=7 * 2**30)
    start = time.monotonic()
    staged = []
    for i, source in enumerate(vectors):
        path = directory / ("space%d.fbin" % i)
        with path.open("wb") as out:
            out.write(struct.pack("<II", len(ids), 512))
            for first in range(0, len(ids), 8192):
                source[ids[first:first+8192]].tofile(out)
        staged.append(path)
    extract = time.monotonic() - start
    graph = directory / "float-graph.bin"
    args = [binary, "shard", *staged, len(ids), 40, graph, L, 750]
    if caps is not None:
        path = directory / "caps.u8"
        caps.tofile(path)
        args.append(path)
    result = measured(run, label, args)
    log = (run.root / "logs" / (label + ".log")).read_text(encoding="utf-8")
    link_time, = re.findall(r"Link time: ([0-9.]+)s", log)
    result.update(nodes=len(ids), max_degree=40, extract_seconds=extract,
                  link_seconds=float(link_time), memberships=str(link), memberships_sha256=link_sha,
                  graph=str(graph), graph_sha256=sha(graph))
    if caps is not None:
        result.update(cap_file_sha256=sha(args[-1]), nodes_R27=int(np.count_nonzero(caps == 27)))
    for path in staged:
        path.unlink()
    return result


def clean_measure(root, name, entry, lock, operation, cores=CORES):
    for attempt in count(1):
        tag = "%s-attempt-%03d" % (name, attempt)
        directory = entry / tag
        record = directory / "measurement.json"
        if record.exists():
            old = read(record)
            if old["environment"]["valid"]:
                return old["result"]
            continue
        if directory.exists():
            continue
        with quiet_window(root, tag, lock, cores) as ready:
            directory.mkdir(parents=True)
            status(root, "TIMED_BUILD_RUNNING", item=tag)
            result, environment = guarded_call(cores, lambda: operation(directory, tag), ready)
        save(record, dict(environment=environment, result=result))
        if environment["valid"]:
            return result
        status(root, "RETRY_SMT_INTERFERENCE", item=tag, peak=environment["max_sibling_busy"])


def group_degree(folder, counts):
    topo = np.memmap(folder / "disk_index_graph", "u1", "r", shape=(N//25, 8192))
    packed = np.ndarray((N//25, 25), dtype="<u4", buffer=topo, strides=(8192, 324))
    values = packed.reshape(-1)
    if values.max() > 40:
        raise RuntimeError("merged degrees exceed R=40")
    stats = {}
    for name, subset in (("single_membership", counts == 1), ("multiple_memberships", counts > 1)):
        hist = np.bincount(values[subset].astype(np.int64), minlength=41)
        stats[name] = dict(nodes=int(hist.sum()), mean=float(values[subset].mean()), histogram=hist.tolist())
    return stats


def build_full(root, linker, origin, host, lock):
    from paired_build_phase1 import measured, merge
    from run_phase0 import Session

    full = root / "full"
    full.mkdir(exist_ok=True)
    while True:
        state = read(root / "status.json")["state"]
        if state == "FAIR_LINK_COMPLETE":
            break
        if state == "FAIR_LINK_STOPPED":
            raise RuntimeError("fair Link stage stopped")
        status(full, "WAITING_FOR_FAIR_LINK")
        time.sleep(20)
    if os.sched_getaffinity(0) != CORES:
        raise RuntimeError("full builds use physical CPUs 16–23")
    class MergeSession(Session):
        def command(self, name, args):
            return super().command(name, (*MERGE_AFFINITY, *args))

    run = Session(SimpleNamespace(root=full, host=host, numa_node=0))
    merge_run = MergeSession(SimpleNamespace(root=full, host=host, numa_node=2))
    binary = setup(root, linker, origin)
    matrixes = prepare_inputs(run, origin)
    membership_counts, groups = memberships(origin)
    inputs = {"count_sha256": hashlib.sha256(membership_counts.tobytes()).hexdigest(),
              "groups": {str(path): digest for path, digest, _ in groups},
              "binary_sha256": sha(binary)}
    save(full / "analysis/inputs.json", inputs)
    old = read(origin / "analysis/phase1-graphs.json")
    record = {"one-shot": old["one-shot"], "paired-k8": old["paired-k8"]}
    record["paired-k8"]["degree_by_membership"] = group_degree(Path(record["paired-k8"]["folder"]),
                                                              membership_counts)
    for name, (L, varied) in VARIANTS.items():
        entry = full / "graphs" / name
        shards = []
        for number, (path, digest, ids) in enumerate(groups):
            cap = np.where(membership_counts[ids] == 1, 40, 27).astype("u1") if varied else None
            label = "%s-shard%d" % (name, number)
            shard = clean_measure(full, label, entry, lock,
                                  lambda d, tag: stage_shard(run, binary, d, tag, ids, path, digest,
                                                            matrixes, cap, L, measured))
            shards.append(shard)
        def merge_shards(directory, tag):
            result = merge(merge_run, binary, shards, directory, tag)
            result["merge_numa_node"] = merge_run.args.numa_node
            result["merge_cpu_list"] = sorted(MERGE_CORES)
            return result
        merged = clean_measure(full, name + "-merge", entry, lock, merge_shards, cores=MERGE_CORES)
        merged.update(build_points=sum(s["nodes"] for s in shards), L=L,
                      cap_policy="single=40, multiple=27" if varied else "constant=40",
                      shard_link_seconds=sum(s["link_seconds"] for s in shards),
                      shard_build_seconds=sum(s["wall_seconds"] for s in shards),
                      shard_extract_seconds=sum(s["extract_seconds"] for s in shards),
                      merge_total_seconds=merged["wall_seconds"],
                      peak_rss_kib_total=max([merged["peak_rss_kib"]]+[s["peak_rss_kib"] for s in shards]),
                      degree_by_membership=group_degree(Path(merged["folder"]), membership_counts))
        record[name] = merged
        save(full / "analysis/phase1-graphs.json", record)
    status(full, "FULL_GRAPHS_READY", graphs=list(record))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "linker", "origin", "host", "lock"):
        parser.add_argument(name, type=Path)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if args.preflight:
        setup(args.root, args.linker, args.origin)
    else:
        try:
            build_full(args.root, args.linker, args.origin, args.host, args.lock)
        except (Exception, KeyboardInterrupt) as error:
            status(args.root / "full", "FULL_BUILD_STOPPED", reason=str(error))
            raise
