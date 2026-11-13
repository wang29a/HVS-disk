#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import struct

p = argparse.ArgumentParser()
p.add_argument("--root", required=True)
p.add_argument("--source", required=True)
p.add_argument("--data", required=True)
a = p.parse_args()
paths = [a.source + "/" + f for f in ("_disk.index.bp", "disk_index_graph.bp", "disk_index_data",
         "single_index", "mem.index", "emb_pq_compressed.bin", "loc_pq_compressed.bin",
         "emb_pq_pivots.bin", "loc_pq_pivots.bin")]
paths += [a.data + "/" + f for f in ("base_img_emb.fvecs", "base_text_emb.fvecs",
          "query_img_emb.fvecs", "query_text_emb.fvecs", "base_gt_round_00/query_alpha.fvecs",
          "base_gt_round_00/top10_results.ivecs")]
files = []
for path in paths:
    s = os.stat(path)
    with open(path, "rb") as f:
        head = f.read(32)
    item = dict(path=path, realpath=os.path.realpath(path), bytes=s.st_size,
                mtime_ns=s.st_mtime_ns, device=s.st_dev, inode=s.st_ino, header_hex=head.hex())
    if s.st_size <= 1024 ** 3:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
        item["sha256"] = h.hexdigest()
    files.append(item)
result = dict(commit="d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e",
              starling_commit="7437c4848a83e7bf62558c5eab1eb999feed8537",
              files=files, threads=[32, 64], beam=4, k=10, query_count=1000, repeats=3,
              numa_nodes=[0, 1], L=[20, 40, 60, 80, 100, 120, 160, 200, 300, 400],
              starling_rounds=8, starling_threads=32, entrypoints=22,
              baseline=dict(mode=0, mem_L=0, strategy=0),
              separated=dict(mode=6, mem_L=0, strategy=17, trunc_len_default=1000000000),
              env={k: os.environ.get(k) for k in ("LD_LIBRARY_PATH", "OPENBLAS_NUM_THREADS",
                   "MKL_NUM_THREADS", "OMP_NUM_THREADS", "OMP_PROC_BIND", "OMP_PLACES")},
              notes=["64 threads on NUMA 0,1 includes SMT: 48 physical cores",
                     "Latency field labelled P99 is actually P99.9 in source",
                     "Large input identities use stat/header plus validation, not full SHA-256",
                     "BLAS internal threads explicitly fixed to 1; old run environment was not fully recorded"])
with open(a.root + "/analysis/manifest.json", "w") as f:
    json.dump(result, f, indent=2, sort_keys=True)
    f.write("\n")
print("manifest recorded {} inputs".format(len(files)))
