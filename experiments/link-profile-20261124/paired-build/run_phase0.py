"""Phase 0 of paired-cluster construction. All artifacts are under one work directory."""

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

import numpy as np

GIB = 2 ** 30
N = 2000000
SEED = 20261123
WEIGHTS = (.1, .3, .5, .7, .9)
FILES = ("base_img_emb.fvecs", "base_text_emb.fvecs")


def save(path, value):
    temp = path.with_suffix(".partial")
    temp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def read(path):
    return json.loads(path.read_bytes())


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def open_matrix(path):
    n, d = np.fromfile(path, "<u4", count=2)
    if path.stat().st_size != 8 + 4 * int(n) * int(d):
        raise ValueError("Matrix size " + str(path))
    return np.memmap(path, "<f4", mode="r", offset=8, shape=(int(n), int(d)))


class Session:
    def __init__(self, args):
        self.args = args
        self.root = args.root
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("inputs", "analysis", "logs", "build"):
            (self.root / name).mkdir(exist_ok=True)
        self.path = self.root / "progress.json"
        self.results = read(self.path) if self.path.exists() else {}
        self.started = time.time()

    def status(self, state, **extra):
        peer = os.environ.get("HVS_PEER_STATUS")
        if peer:
            other = read(Path(peer))
            extra["topology_peer"] = {k: other.get(k) for k in ("state", "task", "item")}
        save(self.root / "status.json", dict(state=state, time=time.time(), numanode=self.args.numa_node,
             cpus=sorted(os.sched_getaffinity(0)), completed=list(self.results), **extra))
        print(state, json.dumps(extra, ensure_ascii=True), flush=True)

    def guard(self, extra=0, memory=0):
        free = shutil.disk_usage(self.root).free
        mem = next(int(row.split()[1]) for row in Path("/proc/meminfo").read_text().splitlines()
                   if row.startswith("MemAvailable:")) * 1024
        path = Path("/sys/devices/system/node/node%d/meminfo" % self.args.numa_node)
        local = next(int(row.split()[3]) for row in path.read_text().splitlines()
                     if row.split()[2] == "MemFree:") * 1024
        if free < 32 * GIB + extra or min(mem, local) < 24 * GIB + memory:
            raise RuntimeError("Workspace or selected NUMA node reached the shared-host resource floor")

    def command(self, name, args):
        with (self.root / "logs" / (name + ".log")).open("w") as f:
            print("argv:", [str(a) for a in args], file=f, flush=True)
            p = subprocess.Popen(list(map(str, args)), stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                while p.poll() is None:
                    self.guard()
                    time.sleep(2)
            finally:
                if p.poll() is None:
                    os.killpg(p.pid, signal.SIGTERM)
                    try:
                        p.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(p.pid, signal.SIGKILL)
                        p.wait()
        if p.returncode:
            raise RuntimeError(name + " failed. See " + str(self.root / "logs" / (name + ".log")))

    def step(self, name, fn):
        if name in self.results:
            return self.results[name]
        self.status("RUNNING", item=name)
        start = time.monotonic()
        value = fn()
        self.results[name] = dict(seconds=time.monotonic() - start, result=value)
        save(self.path, self.results)
        self.status("ITEM_RECORDED", item=name)
        return self.results[name]


def inputs(run):
    dirs = run.root / "inputs"
    manifest = read(run.args.subset_manifest)
    contract = read(run.args.graph_identity)
    if run.args.base.resolve() == dirs.resolve():
        raise ValueError("Input source and isolated copy must be distinct")
    observed = {}
    for name in FILES:
        target = dirs / name
        src = run.args.base / name
        expected = manifest["files"][name]["sha256"]
        if not target.is_file():
            run.guard(extra=src.stat().st_size)
            part = target.with_suffix(".partial")
            shutil.copyfile(src, part)
            if digest(part) != expected:
                raise ValueError("Source data hash mismatch for " + name)
            part.replace(target)
        if digest(target) != expected or open_matrix(target).shape != (N, 512):
            raise ValueError("Frozen matrix mismatch: " + name)
        observed[name] = expected
    ids = np.random.RandomState(SEED + 40).choice(N, 10000, replace=False).astype("<u4")
    train = np.random.RandomState(SEED).choice(N, 200000, replace=False).astype("<u4")
    for name, values in (("coverage", ids), ("training", train)):
        out = dirs / (name + ".u32")
        if out.exists() and not np.array_equal(np.fromfile(out, "<u4"), values):
            raise ValueError("Frozen sample mismatch: " + name)
        if not out.exists():
            temp = out.with_suffix(".partial")
            values.tofile(temp)
            temp.replace(out)
        observed[name + "_ids_sha256"] = digest(out)
    observed.update(semantic_graph_sha256=contract["graph_sha256"], merge2_sha256=contract["merge2_sha256"],
                    report_and_tune_sha256=contract["queries"], nodes=N, dim=[512,512],
                    training_rate=.1, sample_queries=len(ids), seed=SEED)
    save(dirs / "identity.json", observed)
    return observed


def validate_inputs(run, known):
    for name in FILES:
        if digest(run.root / "inputs" / name) != known[name]:
            raise ValueError("Staged vector hash changed: " + name)
    for name in ("coverage", "training"):
        if digest(run.root / "inputs" / (name + ".u32")) != known[name + "_ids_sha256"]:
            raise ValueError("Staged sample hash changed: " + name)


def exact(run):
    root = run.root
    src = Path(__file__).with_name("exact_top40.cpp")
    binary = root / "build/exact_top40"
    build_id = root / "build/exact_top40.json"
    if build_id.is_file():
        pair = read(build_id)
        if pair != dict(source_sha256=digest(src), binary_sha256=digest(binary)):
            raise ValueError("Exact scanner identity changed")
    else:
        blas = run.args.blas
        run.command("build-exact", ["g++-11", "-std=c++17", "-O3", "-march=native", "-fopenmp",
                    "-fno-math-errno", "-ffp-contract=off", "-I"+str(blas/"include"), src,
                    "-L"+str(blas/"lib"), "-Wl,-rpath,"+str(blas/"lib"), "-lopenblas", "-o", binary])
        save(build_id, dict(source_sha256=digest(src), binary_sha256=digest(binary)))
    run.guard(memory=18 * GIB)
    part = root / "analysis/exact-top40.partial.bin"
    dest = root / "analysis/exact-top40.bin"
    time_file = root / "logs/exact-top40.time"
    d = root / "inputs"
    if not dest.is_file():
        run.command("exact-top40", ["/usr/bin/time", "-v", "-o", time_file, binary, d / FILES[0],
                    d / FILES[1], d/"coverage.u32", part])
        part.replace(dest)
    raw = np.fromfile(dest, "<u4")
    if raw[:3].tolist() != [10000, 5, 40] or len(raw) != 3 + 10000*5*40:
        raise ValueError("Exact ground truth shape")
    nbr = raw[3:].reshape(10000, 5, 40)
    ids = np.fromfile(d/"coverage.u32", "<u4")
    if np.any(nbr >= N) or np.any(nbr == ids[:,None,None]) or np.any(np.diff(np.sort(nbr,axis=-1),axis=-1) == 0):
        raise ValueError("Invalid exact top-40 labels")
    peak = next(int(s.rsplit(":",1)[-1]) for s in time_file.read_text().splitlines()
                if "Maximum resident set size" in s)
    return dict(sha256=digest(dest), peak_rss_kib=peak, records=len(nbr), threads=8, weights=WEIGHTS)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("root", type=Path)
    p.add_argument("--base", type=Path, required=True)
    p.add_argument("--subset-manifest", type=Path, required=True)
    p.add_argument("--graph-identity", type=Path, required=True)
    p.add_argument("--blas", type=Path, required=True)
    p.add_argument("--numa-node", type=int, default=0)
    p.add_argument("--queue-lock", type=Path, help="Pass the global queue lock for serial execution.")
    args = p.parse_args()
    run = Session(args)
    lock_path = args.queue_lock or (args.root / "queue.lock")
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            ready = "INPUTS" in run.results
            identity = run.step("INPUTS", lambda: inputs(run))["result"]
            if ready:
                validate_inputs(run, identity)
            run.step("A3_EXACT_TOP40", lambda: exact(run))
            from clustering import analyse
            analyse(run)
        except (KeyboardInterrupt, Exception) as exc:
            run.status("PAUSED" if isinstance(exc, KeyboardInterrupt) else "STOPPED", reason=str(exc))
            raise


if __name__ == "__main__":
    main()
