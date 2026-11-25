"""Compare one fixed-seed random shard with the profiled K=8 shard."""

import argparse
import fcntl
from pathlib import Path

import numpy as np

from build import read, sha
from run import obtain, save, stage, status

SEED = 20261124
LABELS = {
    "candidate_search": "候选搜索",
    "new_point_prune": "新点剪枝",
    "reverse_maintenance": "反向边维护",
    "cleanup": "末尾清理",
    "bookkeeping": "其他构图工作",
    "search_visited": "访问节点",
    "search_expanded": "展开节点",
    "prune_candidates": "剪枝候选",
    "skyline_layers": "skyline 层数",
    "retained_edges": "新点保留边",
    "overflow_reprunes": "溢出重剪",
}


def k8_profile(parent):
    selection = read(parent / "analysis/inputs.json")["shards"]["k8"]
    source = parent / "runs/k8-L200-C750/result.json"
    row = read(source)
    if (row["nodes"], row["R"], row["L"], row["C"], row["threads"]) != (
            selection["nodes"], selection["R"], 200, 750, 8):
        raise RuntimeError("K=8 comparator parameters differ from the frozen run")
    return selection, row, source


def stage_sample(root, old, n):
    manifest = root / "analysis/random-inputs.json"
    ids_path = root / "members.u32"
    population = read(old / "inputs/identity.json")["nodes"]
    selected = np.random.RandomState(SEED).choice(population, n, replace=False)
    ids = np.sort(selected).astype("<u4")
    if manifest.exists():
        witness = read(manifest)
        choice = witness["shards"]["random"]
        if (len(ids) != choice["nodes"] or str(ids_path) != choice["memberships"]
                or sha(ids_path) != choice["memberships_sha256"]
                or not np.array_equal(np.fromfile(ids_path, "<u4"), ids)):
            raise RuntimeError("staged random membership changed")
        for name, digest in choice["material_sha256"].items():
            if sha(root / "data/random" / name) != digest:
                raise RuntimeError("staged random vector matrix changed")
    else:
        ids.tofile(ids_path)
        choices = {"random": {"nodes": n, "memberships": str(ids_path),
                              "memberships_sha256": sha(ids_path)}}
        stage(root, old, choices, manifest)
    return population, sha(ids_path)


def report(root, baseline, random, identity):
    def dist(row, key):
        s = row["count_summary"][key]
        return "%.2f / %d / %d / %d / %d–%d" % (s["mean"], s["p50"], s["p90"], s["p99"],
                                                 s["min"], s["max"])

    lines = [
        "# 随机分片与 K=8 配对分片的 Link 对照", "",
        "两片各 {:,} 点，R=40、L=200、C=750，8 个物理核，绑定 NUMA0。"
        "随机分片从同一 Laion 基库均匀无放回抽取，种子为 {}，按原始 ID 顺序写入。"
        "本文比较两个单次分片运行。原 K=8 剖析与历史日志的墙钟偏差为 {:+.1f}%，"
        "故耗时保留各自实测值，工作量优先看候选、展开及重剪计数。"
        .format(identity["nodes"], SEED, identity["k8_vs_history_pct"]),
        "",
        "| 指标 | K=8 双侧扩边 | 随机分片 |",
        "|---|---:|---:|",
        "| Link 墙钟 秒 | %.2f | %.2f |" % (baseline["link_s"], random["link_s"]),
    ]
    for k in ("candidate_search", "new_point_prune", "reverse_maintenance",
              "cleanup", "bookkeeping"):
        lines.append("| %s，累计线程时间占比 | %.2f%% | %.2f%% |" %
                     (LABELS[k], baseline["thread_time_percent"][k], random["thread_time_percent"][k]))
    for k in ("search_visited", "search_expanded", "prune_candidates", "skyline_layers",
              "retained_edges", "overflow_reprunes"):
        lines.append("| %s，均值 / P50 / P90 / P99 / 范围 | %s | %s |" %
                     (LABELS[k], dist(baseline, k), dist(random, k)))
    lines.extend([
        "| 溢出重剪总次数 | {:,} | {:,} |"
        .format(baseline["overflow_reprunes_total"], random["overflow_reprunes_total"]),
        "",
        "分阶段比例以各线程累计时间之和归一。访问统计不同的距离计算节点，展开统计实际读取邻接表的次数。"
        "保留边取新点剪枝后、写入当前行前的数量。完整直方图和身份见 comparison.json。",
        "",
    ])
    path = root / "analysis/REPORT.md"
    temp = path.with_suffix(".partial")
    temp.write_text("\n".join(lines), encoding="utf-8")
    temp.replace(path)
    save(root / "analysis/comparison.json", dict(identity=identity, k8=baseline, random=random))


def run(root, parent, old):
    selection, baseline, source = k8_profile(parent)
    proof = read(parent / "build/identity.json")["binaries"]["on"]
    binary = Path(proof["binary"])
    if sha(binary) != proof["sha256"] or baseline["binary_sha256"] != proof["sha256"]:
        raise RuntimeError("comparison requires the same profiled executable")
    n = selection["nodes"]
    status(root, "STAGING_RANDOM_MEMBERSHIP", nodes=n, seed=SEED)
    population, members = stage_sample(root, old, n)
    folder = root / "data/random"
    result = obtain(root, "random-L200-C750", binary, folder / "random-0.bin",
                    folder / "random-1.bin", n, baseline["L"], baseline["R"],
                    baseline["C"], threads=baseline["threads"])
    identity = dict(seed=SEED, population=population, nodes=n, sampler="RandomState choice without replacement",
                    order="ascending original ID", membership_sha256=members,
                    comparator=selection["name"], comparator_result_sha256=sha(source),
                    k8_vs_history_pct=(baseline["link_s"]/selection["old_link_s"]-1)*100)
    report(root, baseline, result, identity)
    status(root, "COMPLETE", nodes=n, seed=SEED)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("root", "parent", "old", "queue_lock"):
        p.add_argument(key, type=Path)
    a = p.parse_args()
    for subdir in ("analysis", "logs"):
        (a.root / subdir).mkdir(parents=True, exist_ok=True)
    try:
        status(a.root, "WAITING_FOR_LINK_WINDOW")
        with a.queue_lock.open("rb") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            run(a.root, a.parent, a.old)
    except (Exception, KeyboardInterrupt) as error:
        status(a.root, "STOPPED", reason=str(error))
        raise
