"""Retry the two historical time checks, then complete the frozen K=8 grid."""

import argparse
import fcntl
import os
from pathlib import Path
import threading
import time

from build import read, sha
from run import obtain, render, save, status


CORES = set(range(16, 24))


def siblings(cpus):
    result = set()
    for cpu in cpus:
        path = Path("/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list" % cpu)
        for item in path.read_text().strip().split(","):
            limits = [int(i) for i in item.split("-")]
            result.update(range(limits[0], limits[-1] + 1))
    return result - cpus


def ticks(cpus):
    result = {}
    for row in Path("/proc/stat").read_text().splitlines():
        split = row.split()
        if not split:
            continue
        label, *fields = split
        if label.startswith("cpu") and label[3:].isdigit() and int(label[3:]) in cpus:
            counts = [int(i) for i in fields[:8]]
            result[int(label[3:])] = (sum(counts), counts[3] + counts[4])
    if result.keys() != cpus:
        raise RuntimeError("CPU topology changed while sampling")
    return result


def occupancy(before, after, cpus):
    total = sum(after[i][0] - before[i][0] for i in cpus)
    idle = sum(after[i][1] - before[i][1] for i in cpus)
    return 1 - idle / total


def quiet(root, tag, cores, smt):
    last = None
    while True:
        cpus = cores | smt
        before = ticks(cpus)
        time.sleep(4)
        after = ticks(cpus)
        a, b = occupancy(before, after, cores), occupancy(before, after, smt)
        if max(a, b) <= .10:
            return {"physical_busy": a, "smt_busy": b, "sample_seconds": 4}
        now = time.monotonic()
        if last is None or now - last >= 60:
            status(root, "WAITING_FOR_ORIGINAL_CORES", item=tag,
                   physical_busy_pct=round(a * 100, 1), smt_busy_pct=round(b * 100, 1))
            last = now
        time.sleep(15)


def monitor_smt(smt, quit_event, samples):
    current = ticks(smt)
    while not quit_event.wait(5):
        new = ticks(smt)
        samples.append(occupancy(current, new, smt))
        current = new


def run(root, parent, lock, scan_validated_k8=False):
    if os.sched_getaffinity(0) != CORES:
        raise RuntimeError("expected original physical cores 16-23")
    smt = siblings(CORES)
    original = read(parent / "status.json")
    snapshot = root / "analysis/original-status.json"
    if not snapshot.exists():
        if original["state"] != "STOPPED_OVERHEAD_GATE":
            raise RuntimeError("expected completed first attempts at the historical gate")
        save(snapshot, original)
    status(parent, "RESUMING_OVERHEAD_GATE", retry_status=str(root / "status.json"))
    choices = read(parent / "analysis/inputs.json")["shards"]
    proof = read(parent / "build/identity.json")["binaries"]["on"]
    binary = Path(proof["binary"])
    if sha(binary) != proof["sha256"]:
        raise RuntimeError("profile binary changed")
    old_rows = {r["shard"]: r for r in read(parent / "analysis/results.json")}
    failed = set(read(snapshot)["shards"])
    for tag, item in choices.items():
        for name, digest in item["material_sha256"].items():
            if sha(parent / "data" / tag / name) != digest:
                raise RuntimeError("profile input changed: " + tag)
    rows, retries = [], []

    def measure(tag, L, C, baseline=False):
        item = choices[tag]
        key = "%s-L%d-C%d" % (tag, L, C)
        folder = parent / "data" / tag
        slot = root / "runs" / key
        runtime = slot / "environment.json"
        cached = (slot / "result.json").exists()
        if cached:
            row = obtain(root, key, binary, folder / (tag + "-0.bin"),
                         folder / (tag + "-1.bin"), item["nodes"], L, item["R"], C)
        else:
            quit_event, smt_samples = threading.Event(), []
            # Wait outside the experiment lock so an idle queue does not reserve SSD.
            while True:
                admission = quiet(root, key, CORES, smt)
                with lock.open("rb") as handle:
                    try:
                        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        status(root, "WAITING_FOR_EXCLUSIVE_WINDOW", item=key)
                    else:
                        watcher = threading.Thread(target=monitor_smt,
                                                   args=(smt, quit_event, smt_samples), daemon=True)
                        watcher.start()
                        try:
                            row = obtain(root, key, binary, folder / (tag + "-0.bin"),
                                         folder / (tag + "-1.bin"), item["nodes"], L, item["R"], C)
                        finally:
                            quit_event.set()
                            watcher.join()
                        save(runtime, dict(admission=admission, sibling_cpus=sorted(smt),
                                           sibling_busy_fraction=smt_samples))
                        break
                time.sleep(10)
        row["shard"] = item["name"]
        if runtime.exists():
            row["environment"] = read(runtime)
        if baseline:
            row["original_link_s"] = item["old_link_s"]
            row["first_profile_link_s"] = old_rows[item["name"]]["link_s"]
            row["delta_pct"] = (row["link_s"] / row["original_link_s"] - 1) * 100
            retries.append(row)
        rows.append(row)
        render(root, rows)

    for tag in ("legacy", "k8", "k4"):
        name = choices[tag]["name"]
        if name in failed:
            measure(tag, 200, 750, baseline=True)
        else:
            rows.append(old_rows[name])
            render(root, rows)
    missed = [row["shard"] for row in retries if abs(row["delta_pct"]) > 5]
    k8_gate_met = choices["k8"]["name"] not in missed
    if missed and (not scan_validated_k8 or not k8_gate_met):
        status(root, "STOPPED_OVERHEAD_GATE", shards=missed)
        status(parent, "STOPPED_OVERHEAD_GATE", shards=missed, retry_report=str(root / "analysis/REPORT.md"))
        return
    outcome = dict(historical_gate_met=not missed, failed_shards=missed,
                   k8_historical_gate_met=k8_gate_met)
    save(root / "analysis/scan-eligibility.json", outcome)
    for L, C in ((200, 400), (128, 750), (128, 400)):
        measure("k8", L, C)
    if missed:
        with (root / "analysis/REPORT.md").open("a", encoding="utf-8") as report:
            report.write("\nK=8 扫描分片的原参数复测通过历史 Link 墙钟的 ±5% 校验。"
                         "旧双路的复测仍超界，因此涉及旧双路的墙钟对照单独标注。\n")
    status(root, "COMPLETE", measurements=len(rows), **outcome)
    status(parent, "COMPLETE", measurements=len(rows),
           retry_report=str(root / "analysis/REPORT.md"), **outcome)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "parent", "lock"):
        parser.add_argument(key, type=Path)
    parser.add_argument("--scan-validated-k8", action="store_true")
    args = parser.parse_args()
    (args.root / "analysis").mkdir(parents=True, exist_ok=True)
    try:
        run(args.root, args.parent, args.lock, args.scan_validated_k8)
    except (Exception, KeyboardInterrupt) as error:
        status(args.root, "STOPPED", reason=str(error))
        if (args.root / "analysis/original-status.json").is_file():
            status(args.parent, "STOPPED", reason=str(error), retry_status=str(args.root / "status.json"))
        raise
