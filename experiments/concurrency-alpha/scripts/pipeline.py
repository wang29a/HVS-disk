#!/usr/bin/env python3
"""Serial, resumable concurrency/alpha sweep using supplied query/GT files only."""
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import statistics
import struct
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path("/mnt/nvme3/wz/hvs-disk-build-20261027")
OLD = Path("/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32")
LS = [20, 40, 60, 80, 100, 120, 160, 200, 300, 400]
THREADS = [1, 2, 4, 8, 16, 32, 64]
KINDS = ["baseline", "full", "merge", "no-alpha"]
SECONDS = 5.0
ENV = os.environ.copy()
ENV.update(OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_PROC_BIND="close",
           OMP_PLACES="cores", HVS_POINT_SECONDS=str(SECONDS), LC_ALL="C")
ENV["LD_LIBRARY_PATH"] = "/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:" + ENV.get("LD_LIBRARY_PATH", "")
ENV.pop("HVS_ONE_PASS", None)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def write_json(path, value):
    path = Path(path)
    temporary = str(path) + ".tmp"
    with open(temporary, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(temporary, str(path))


def sha(path):
    h = hashlib.sha256()
    with open(str(path), "rb") as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def log(message):
    print(now(), message, flush=True)


def header(path, dtype_bytes=4):
    with open(str(path), "rb") as f:
        n, d = struct.unpack("<II", f.read(8))
    if os.path.getsize(str(path)) != 8 + n * d * dtype_bytes:
        raise ValueError("not an exact headered matrix: " + str(path))
    return n, d


def configurations():
    import numpy as np
    configs = []
    for dataset, directory in [("openimage", "Openimage_disk"), ("cc3m", "CC3M_disk"),
                               ("howto100m", "Howto100M_disk"), ("laion", "Laion")]:
        data = Path("/mnt/nvme2/wz/disk/dataset/Laion") if dataset == "laion" else Path("/mnt/nvme3/wz/dataset") / directory
        indexes = BUILD / "experiments/laion" if dataset == "laion" else OLD / dataset
        pairs = [("fixed-0.5" if dataset == "laion" else "mixed", data / "base_gt_round_00/query_alpha.fvecs",
                  data / "base_gt_round_00/top10_results.ivecs", .5 if dataset == "laion" else None)]
        if dataset == "laion":
            pairs += [("fixed-0.{}".format(i), data / ("alpha{}.bin".format(i)),
                       data / ("gt/base_gt_round_00/top10_results_alpha_0.{}.ivecs".format(i)), i / 10)
                      for i in (1, 3, 7, 9)]
        q1, q2 = data / "query_img_emb.fvecs", data / "query_text_emb.fvecs"
        q, d1 = header(q1)
        q_other, d2 = header(q2)
        if q != q_other:
            raise ValueError("query counts mismatch")
        for condition, alpha, gt, fixed in pairs:
            if header(alpha) != (q, 1) or header(gt)[0] != q or header(gt)[1] < 10:
                raise ValueError("unpaired alpha/GT")
            values = np.fromfile(str(alpha), dtype="<f4", offset=8)
            truth = np.fromfile(str(gt), dtype="<u4", offset=8).reshape(q, -1)
            if not np.isfinite(values).all() or np.any(values < 0) or np.any(values > 1):
                raise ValueError("invalid alpha values")
            if fixed is not None and not np.allclose(values, fixed, atol=1e-7, rtol=0):
                raise ValueError("alpha file differs from condition label")
            for kind in KINDS:
                index = indexes / kind
                meta_file = index / ("single_index" if kind == "baseline" else "_disk.index")
                with open(str(meta_file), "rb") as f:
                    n, a, b = struct.unpack("<III", f.read(12))
                if (a, b) != (d1, d2) or np.any(truth >= n):
                    raise ValueError("index/query/GT ID mismatch")
                perf_files = [index / "single_index"] if kind == "baseline" else [index / "disk_index_graph", index / "disk_index_data"]
                if any(os.stat(str(p)).st_dev != os.stat("/mnt/nvme3").st_dev for p in perf_files):
                    raise ValueError("performance input is not on nvme3")
                for file in ("emb_pq_compressed.bin", "loc_pq_compressed.bin", "emb_pq_pivots.bin", "loc_pq_pivots.bin"):
                    if not (index / file).is_file():
                        raise ValueError("missing PQ file")
            configs.append(dict(dataset=dataset, condition=condition, fixed_alpha=fixed,
                query1=str(q1), query2=str(q2), alpha=str(alpha), gt=str(gt), indexes=str(indexes),
                queries=q, dims=[d1, d2], alpha_min=float(values.min()), alpha_max=float(values.max()),
                bucket_queries=np.bincount(np.minimum(4, (values * 5).astype(int)), minlength=5).tolist(),
                input_sha256={str(p): sha(p) for p in (q1, q2, alpha, gt)}))
    # Cover all four original workloads first; fixed-alpha extras follow.
    return [c for c in configs if c["dataset"] != "laion" or c["condition"] == "fixed-0.5"] + [
        c for c in configs if c["dataset"] == "laion" and c["condition"] != "fixed-0.5"]


def build_driver():
    source = BUILD / "source"
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(source)).decode().strip()
    if commit != "d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e":
        raise ValueError("unexpected source commit")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=str(source)).strip():
        raise ValueError("original checkout is dirty")
    cpp = ROOT / "scripts/steady_search.cpp"
    subprocess.run(["cp", str(source / ".clang-format"), str(ROOT / ".clang-format")], check=True)
    subprocess.run(["clang-format", "-i", "-style=file", str(cpp)], check=True)
    identities = {}
    for kind, directory in (("separated", "build-separated"), ("baseline", "build-single-baseline")):
        build = BUILD / directory
        flags = (build / "tests/CMakeFiles/search_disk_index.dir/flags.make").read_text()
        args = []
        for key in ("CXX_DEFINES", "CXX_INCLUDES", "CXX_FLAGS"):
            args += shlex.split(re.search(r"^" + key + r" = (.*)$", flags, re.M).group(1))
        binary = ROOT / "bin" / ("steady-" + kind)
        command = ["g++-11"] + args + [str(cpp), str(build / "src/libpipeann.a"),
            "-L/mnt/nvme2/wz/disk/local/openblas/lib", "-laio", "-ljemalloc", "-lopenblas", "-o", str(binary)]
        with open(str(ROOT / "logs" / ("build-" + kind + ".log")), "w") as f:
            f.write(shlex.join(command) + "\n" if hasattr(shlex, "join") else " ".join(command) + "\n")
            f.flush()
            subprocess.run(command, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
        identities[kind] = dict(binary=str(binary), binary_sha256=sha(binary),
            library=str(build / "src/libpipeann.a"), library_sha256=sha(build / "src/libpipeann.a"),
            flags=args, build_command=command)
    identities["driver_sha256"] = sha(cpp)
    write_json(ROOT / "analysis/build.json", identities)


def command(c, kind, threads, ls, original=False):
    if original:
        binary = BUILD / ("build-single-baseline" if kind == "baseline" else "build-separated") / "tests/search_disk_index"
    else:
        binary = ROOT / "bin" / ("steady-baseline" if kind == "baseline" else "steady-separated")
    return ["numactl", "--cpunodebind=0,1", "--interleave=0,1", str(binary), "float",
        c["indexes"] + "/" + kind + "/", str(threads), "4", c["query1"], c["query2"],
        c["alpha"], c["gt"], "10", "l2", "0" if kind == "baseline" else "6", "0",
        "0" if kind == "baseline" else "17"] + [str(v) for v in ls]


def points(path):
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = [json.loads(line[len("HVS_POINT "):]) for line in text.splitlines() if line.startswith("HVS_POINT ")]
    if "HVS_STEADY_COMPLETE" not in text:
        raise ValueError("steady run incomplete: " + str(path))
    return rows


def parity(configs):
    cases = [(configs[0], k) for k in KINDS] + [(configs[1], "full"), (configs[3], "full")]
    reports = []
    for c, kind in cases:
        label = c["dataset"] + "-" + kind
        env = dict(ENV, OMP_NUM_THREADS="2", HVS_ONE_PASS="1")
        logs = []
        for original in (True, False):
            path = ROOT / "logs" / ("parity-" + label + ("-original.log" if original else "-steady.log"))
            with open(str(path), "w") as f:
                subprocess.run(command(c, kind, 2, [40, 100], original), env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            logs.append(path)
        old = {}
        for line in logs[0].read_text(encoding="utf-8", errors="replace").splitlines():
            f = line.split()
            if len(f) >= 17 and f[0].isdigit() and int(f[0]) in (40, 100):
                old[int(f[0])] = dict(recall=float(f[-1]), hops=float(f[5]), comparisons=float(f[6]))
        new = points(logs[1])
        if sorted(old) != [40, 100] or [p["L"] for p in new] != [40, 100]:
            raise ValueError("missing parity points")
        for p in new:
            for metric in ("recall", "hops", "comparisons"):
                if abs(p[metric] - old[p["L"]][metric]) > .011:
                    raise ValueError("parity mismatch {} {}".format(label, metric))
        reports.append(dict(dataset=c["dataset"], kind=kind, passed=True))
        log("PARITY_PASS " + label)
    # Timed replay must honor its duration and preserve the original input set.
    path = ROOT / "logs/timed-driver-test.log"
    env = dict(ENV, OMP_NUM_THREADS="4", HVS_POINT_SECONDS="1")
    with open(str(path), "w") as f:
        subprocess.run(command(configs[0], "full", 4, [40]), env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
    p = points(path)[0]
    if p["seconds"] < 1 or p["samples"] % configs[0]["queries"] or p["unique_queries"] != configs[0]["queries"]:
        raise ValueError("invalid timed replay")
    write_json(ROOT / "analysis/driver-validation.json", dict(parity=reports, timed_replay=p, status="PASS"))


def refresh_summary(total):
    grouped = collections.defaultdict(list)
    completed = 0
    for path in sorted((ROOT / "results").glob("*.json")):
        r = json.loads(path.read_text())
        completed += 1
        for p in r["points"]:
            key = (r["dataset"], r["condition"], r["threads"], r["scheme"], p["L"])
            grouped[key].append(p)
    means = []
    for key, samples in sorted(grouped.items()):
        row = dict(zip(("dataset", "condition", "threads", "scheme", "L"), key))
        row["repeats"] = len(samples)
        for metric in ("qps", "recall", "latency_us", "p95_us", "p99_us", "p999_us", "hops",
                       "comparisons", "topology_ios", "rerank_ios", "cache_hits", "cpu_us", "io_us"):
            values = [p[metric] for p in samples]
            row[metric] = statistics.mean(values)
            row[metric + "_stddev"] = statistics.stdev(values) if len(values) > 1 else None
        means.append(row)
    curves = collections.defaultdict(list)
    for row in means:
        if row["repeats"] == 3:
            curves[row["dataset"], row["condition"], row["threads"], row["scheme"]].append(row)
    matched = []
    for key, curve in sorted(curves.items()):
        if len(curve) != len(LS):
            continue
        curve.sort(key=lambda p: p["recall"])
        for target in (80, 85, 90, 92, 95):
            for left, right in zip(curve, curve[1:]):
                if left["recall"] <= target <= right["recall"]:
                    span = right["recall"] - left["recall"]
                    ratio = (target - left["recall"]) / span if span else 0
                    row = dict(zip(("dataset", "condition", "threads", "scheme"), key))
                    row["target_recall"] = target
                    for metric in ("L", "qps", "latency_us", "topology_ios", "rerank_ios"):
                        row[metric] = left[metric] + ratio * (right[metric] - left[metric])
                    matched.append(row)
                    break
    write_json(ROOT / "analysis/summary.json", dict(completed_runs=completed, total_runs=total,
               measured_points=sum(len(v) for v in grouped.values()), mean_curves=means,
               matched_recall=matched, note="Partial summaries only interpolate curves with all 3 repeats; alpha buckets are descriptive within mixed workload."))
    return completed


def execute(c, kind, threads, rep, ordinal, total):
    label = "{}__{}__T{}__{}__rep{}".format(c["dataset"], c["condition"], threads, kind, rep)
    result = ROOT / "results" / (label + ".json")
    if result.exists():
        return
    write_json(ROOT / "status.json", dict(status="running", task=label, ordinal=ordinal, total=total,
                                         updated_at=now(), stage="search"))
    path = ROOT / "logs" / (label + ".log")
    temporary = str(path) + ".running"
    started = time.time()
    args = command(c, kind, threads, LS)
    log("RUN {}/{} {}".format(ordinal, total, label))
    env = dict(ENV, OMP_NUM_THREADS=str(threads))
    with open(temporary, "w") as out, open(str(ROOT / "logs" / (label + ".pidstat")), "w") as cpu:
        proc = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env=env)
        monitor = subprocess.Popen(["pidstat", "-u", "-w", "-p", str(proc.pid), "1"], stdout=cpu, stderr=subprocess.STDOUT, env=ENV)
        try:
            code = proc.wait()
        finally:
            monitor.terminate()
            monitor.wait()
    if code:
        os.replace(temporary, str(path) + ".failed-" + str(int(time.time())))
        raise RuntimeError("search failed: " + label)
    rows = points(temporary)
    if [p["L"] for p in rows] != LS:
        raise ValueError("missing search L points")
    for p in rows:
        if p["seconds"] < SECONDS or p["samples"] % c["queries"] or p["unique_queries"] != c["queries"]:
            raise ValueError("short or unpaired measurement")
        if any(not math.isfinite(p[k]) for k in ("qps", "recall", "latency_us")) or not 0 <= p["recall"] <= 100:
            raise ValueError("invalid point metrics")
    os.replace(temporary, str(path))
    write_json(result, dict(dataset=c["dataset"], condition=c["condition"], scheme=kind,
        threads=threads, rep=rep, points=rows, command=args, started_unix=started,
        finished_unix=time.time(), raw_log=str(path), input_sha256=c["input_sha256"]))
    completed = refresh_summary(total)
    log("DONE {}/{} {}".format(completed, total, label))


def main():
    for name in ("bin", "logs", "analysis", "results"):
        (ROOT / name).mkdir(exist_ok=True)
    lock = open(str(ROOT / "queue.lock"), "w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    write_json(ROOT / "status.json", dict(status="running", stage="preparation", updated_at=now()))
    configs = configurations()
    manifest = dict(configurations=configs, threads=THREADS, schemes=KINDS, repeats=3, L=LS,
                    point_seconds=SECONDS, numa_nodes=[0, 1], supplied_GT_only=True,
                    skipped="No fixed-alpha GT for other three datasets; Laion endpoints 0/1 lack paired supplied GT.",
                    notes=["Mixed-alpha buckets do not isolate alpha from query difficulty",
                           "Repeated queries extend measurement duration, not statistical query diversity",
                           "64 threads on nodes 0,1 includes SMT; native IO counters differ by implementation"])
    manifest_path = ROOT / "analysis/manifest.json"
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError("resume inputs or experiment settings changed")
    write_json(manifest_path, manifest)
    if not (ROOT / "analysis/driver-validation.json").exists():
        build_driver()
        parity(configs)
    else:
        build = json.loads((ROOT / "analysis/build.json").read_text())
        if sha(ROOT / "scripts/steady_search.cpp") != build["driver_sha256"]:
            raise ValueError("driver source changed; refuse resume")
        for key in ("baseline", "separated"):
            if sha(build[key]["binary"]) != build[key]["binary_sha256"] or sha(build[key]["library"]) != build[key]["library_sha256"]:
                raise ValueError("binary/library changed; refuse resume")
    total = len(configs) * len(THREADS) * len(KINDS) * 3
    io_log = open(str(ROOT / "logs/iostat-live.log"), "a")
    monitor = subprocess.Popen(["iostat", "-dxm", "1", "nvme3n1"], stdout=io_log, stderr=subprocess.STDOUT, env=ENV)
    try:
        ordinal = 0
        for c in configs:
            for rep in (1, 2, 3):
                ts = THREADS if rep == 1 else list(reversed(THREADS)) if rep == 2 else THREADS[3:] + THREADS[:3]
                kinds = KINDS[rep - 1:] + KINDS[:rep - 1]
                for threads in ts:
                    for kind in kinds:
                        ordinal += 1
                        execute(c, kind, threads, rep, ordinal, total)
        completed = refresh_summary(total)
        if completed != total:
            raise RuntimeError("incomplete task count")
        write_json(ROOT / "status.json", dict(status="complete", completed_runs=completed,
                    total_runs=total, measured_points=completed * len(LS), updated_at=now()))
        log("CONCURRENCY_ALPHA_QUEUE_COMPLETE")
    finally:
        monitor.terminate()
        monitor.wait()
        io_log.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(ROOT / "status.json", dict(status="failed", error=str(error), updated_at=now()))
        raise
