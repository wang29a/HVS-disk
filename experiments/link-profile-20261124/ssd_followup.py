"""Retest the two new full graphs against freshly staged reference graphs."""

import argparse
import fcntl
from itertools import count
import os
from pathlib import Path
from statistics import median, mean
import sys
import time

from timing_guard import LIMIT, guarded_call, siblings, snapshot


CORES = set(range(72, 76))


def imports(topology, original):
    # The phase-1 SSD scripts and Link profiler have different instrument.py files.
    sys.modules.pop("instrument", None)
    sys.path[0:0] = [str(topology), str(original / "source")]
    import paired_build_ssd as ssd
    from phase0b_common import Batch, Fatal, Pending
    return ssd, Batch, Fatal, Pending


def tuner(ssd, run, trace, folder, label):
    rows = [(alpha, depth, run.holdout["tune"], "%s-L%d" % (alpha, depth))
            for alpha in ssd.WEIGHTS for depth in ssd.DEPTHS]
    raw = ssd.trace_job(run, trace, folder, "tune-" + label, "dram", rows)
    gt = {alpha: ssd.matrix(run.truth[alpha], "<u4") for alpha in ssd.WEIGHTS}
    curves = {str(alpha): [] for alpha in ssd.WEIGHTS}
    for (alpha, depth, _, _), path in zip(rows, raw):
        hits = [len(set(q["result"]) & set(gt[alpha][q["id"], :10]))
                for q in ssd.read_trace(path, ssd.NODES)]
        curves[str(alpha)].append((depth, mean(hits) / 10))
    plans = {}
    for alpha, curve in curves.items():
        first = next((i for i, point in enumerate(curve) if point[1] >= .95), None)
        if first is None:
            plans[alpha] = list(ssd.DEPTHS[-5:])
        else:
            plans[alpha] = list(ssd.DEPTHS[max(0, first-2):first+3])
    return dict(plan=plans, curves=curves)


def deliver(ssd, run, graphs, rows):
    labels = list(graphs)
    md = ["# K=8 配对扩边构建 · 度数与候选搜索深度", "",
          "Laion 200 万，原 K=8 分片成员顺序。子图使用 NUMA0 的 8 个物理核，"
          "两张新图的合并使用 NUMA2 的 8 个物理核。原 K=8 构建时间为 NUMA0 归档。"
          "SSD 统一使用 Merge-2、冻结 Starling 顺序、开方模式 6 和 4 物理核，"
          "同一批次上盘后静置 10 分钟，权重倒序，轮内交替，同召回内插。",
          "", "| 图 | 提取分片秒 | 子图构建秒 | 合并秒 | 总分钟 | 平均度数 | 单分片点平均度 | 多分片点平均度 |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for label, g in graphs.items():
        group = g.get("degree_by_membership")
        single = "%.2f" % group["single_membership"]["mean"] if group else "—"
        multiple = "%.2f" % group["multiple_memberships"]["mean"] if group else "—"
        if "archived_build_seconds" in g:
            times = "— | — | — | %.2f（归档）" % (g["archived_build_seconds"]/60)
        else:
            values = tuple(g[k] for k in ("shard_extract_seconds","shard_build_seconds","merge_total_seconds"))
            times = "%.2f | %.2f | %.2f | %.2f" % (*values, sum(values)/60)
        md.append("| %s | %s | %.2f | %s | %s |" %
                  (label, times, g["verification"]["mean_degree"], single, multiple))
    md+=["", "| α | 图 | 达到 95% 的轮次 | 平均查询延迟中位数 ms | 拓扑页 | 坐标页 | 相对原 K=8 | 相对 one-shot |",
         "|---|---|---:|---:|---:|---:|---:|---:|"]
    data = {}
    for alpha in reversed(ssd.WEIGHTS):
        for label in labels:
            points = {}
            for r in range(1,4):
                prefix = "r%d-%s-%s-" % (r, alpha, label)
                rate = ssd.match([v for k,v in rows.items() if k.startswith(prefix)], .95)
                if rate:
                    points[str(r)] = rate
            data["%s-%s" % (alpha,label)] = points
        for label in labels:
            points = data["%s-%s" % (alpha,label)]
            if points:
                values = [median(v[k] for v in points.values()) for k in
                          ("latency_us","topology_pages","coordinate_pages")]
                cells = ["%.3f" % (values[0]/1000), "%.2f" % values[1], "%.2f" % values[2]]
            else:
                cells = ["—"]*3
            for baseline in ("paired-k8", "one-shot"):
                ref = data["%s-%s" % (alpha, baseline)]
                pairs = set(points) & set(ref)
                cells.append("%+.2f%%" % (100*median(points[i]["latency_us"]/ref[i]["latency_us"]-1
                                                     for i in pairs)) if len(pairs)==3 else "—")
            md.append("| %s | %s | %d/3 | %s |" % (alpha,label,len(points)," | ".join(cells)))
    ssd.save(run.analysis / "matched95.json", data)
    (run.analysis / "RESULTS.md").write_text("\n".join(md)+"\n", encoding="utf-8")
    return [k for k,v in data.items() if len(v)!=3]


def benchmark(root, host, previous, original, ssd_path, timing_proof, lock, topology):
    ssd, Batch, Fatal, Pending = imports(topology,original)
    root.mkdir(parents=True, exist_ok=True)
    for name in ("build","logs","inputs"):
        (root/name).mkdir(exist_ok=True)
    run = Batch(root,host,previous)
    run.task = "K8DegreeFollowup"
    full = root.parent / "full"
    while True:
        where = full / "status.json"
        state = ssd.read(where)["state"] if where.exists() else "NOT_STARTED"
        if state == "FULL_GRAPHS_READY":
            break
        if state == "FULL_BUILD_STOPPED":
            run.status("STOPPED",reason="full graph build failed")
            return
        run.status("WAITING_FOR_FULL_GRAPHS")
        time.sleep(25)

    graphs = ssd.read(full / "analysis/phase1-graphs.json")
    proof = ssd.read(timing_proof)["executables"]["result"]
    binary = Path(proof["K2"])
    if ssd.digest(binary) != proof["K2_sha256"]:
        raise Fatal("frozen square-root search executable changed")
    with lock.open("rb") as guard:
        run.status("WAITING_FOR_SSD_LOCK")
        fcntl.flock(guard,fcntl.LOCK_EX)
        staged = ssd.permit(run,"followup-copy",lambda:ssd.prepare_layouts(run,graphs,ssd_path))
        if any(ssd.digest(Path(path)) != digest for path,digest in staged["files"].items()):
            raise Fatal("staged SSD input changed")
        ssd.permit(run,"followup-queries",lambda:ssd.stage_queries(run))
        trace = ssd.permit(run,"followup-trace",
                           lambda:ssd.compile_trace(run,original/"source/native_trace.cpp"))
        if ssd.digest(Path(trace["path"])) != trace["sha256"]:
            raise Fatal("follow-up trace executable changed")
        plans = {}
        for label in graphs:
            r = ssd.permit(run,"followup-tune-"+label,
                           lambda:tuner(ssd,run,trace["path"],ssd_path/label,label))
            plans[label] = r["plan"]
            ssd.permit(run,"followup-parity-"+label,
                       lambda:ssd.gate(run,trace["path"],ssd_path/label,label,plans[label]))
        ssd.permit(run,"followup-idle",lambda:ssd.quiet(run,dict(completed=time.time())))

        out = run.root / "metrics"
        out.mkdir(exist_ok=True)
        path = run.analysis / "clean-points.json"
        rows = ssd.read(path) if path.exists() else {}
        labels = list(graphs)
        groups = dict(workers=CORES,siblings=siblings(CORES))
        for repeat in range(1,4):
            i = repeat-1
            order = labels[i:] + labels[:i]
            for alpha in reversed(ssd.WEIGHTS):
                depths = sorted(set(d for k in order for d in plans[k][str(alpha)]))
                for d in depths:
                    for label in order:
                        if d not in plans[label][str(alpha)]:
                            continue
                        key = "r%d-%s-%s-L%d" % (repeat,alpha,label,d)
                        if key in rows:
                            continue
                        for attempt in count(1):
                            tag = "%s-attempt%03d" % (key,attempt)
                            dest = out/(tag+".json")
                            if dest.exists() or dest.with_suffix(".smt.json").exists():
                                continue
                            while True:
                                preflight = snapshot(groups)
                                if max(preflight.values()) <= LIMIT:
                                    break
                                run.status("WAITING_FOR_QUIET_SSD_CPUS", item=key, preflight=preflight)
                                time.sleep(15)
                            run.status("SSD_TIMING",item=tag)
                            arm = ssd_path/label
                            cmd = [binary,arm,*ssd.report_vectors(run),run.holdout["report"],
                                   run.truth[alpha],alpha,d,ssd.DUR,dest,arm]
                            def measure():
                                try:
                                    return ssd.timed_process(run,cmd,tag,dest), None
                                except Pending as error:
                                    return None, str(error)
                            (value,error), environment = guarded_call(CORES,measure,preflight)
                            ssd.save(dest.with_suffix(".smt.json"),dict(environment=environment,error=error))
                            if not environment["valid"]:
                                run.status("RETRY_SSD_SMT_INTERFERENCE",item=tag)
                                continue
                            if error:
                                if attempt==1:
                                    run.status("RETRY_SSD_INTEGRITY",item=tag,reason=error)
                                    continue
                                raise Pending(error)
                            witness=ssd.read_native_witness(Path(str(dest)+".parity"))
                            value["expanded_nodes"]=mean(len(q["expanded"]) for q in witness)
                            rows[key]=value
                            ssd.save(path,rows)
                            break
                deliver(ssd,run,graphs,rows)
        gaps = deliver(ssd,run,graphs,rows)
        run.status("SSD_COMPLETE_WITH_GAPS" if gaps else "SSD_COMPLETE",missing95=gaps)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for item in ("root","host","previous","original","ssd_path","timing_proof","lock","topology"):
        parser.add_argument(item,type=Path)
    settings=parser.parse_args()
    try:
        benchmark(**vars(settings))
    except (Exception, KeyboardInterrupt) as error:
        from run import status
        status(settings.root,"SSD_STOPPED",reason=str(error))
        raise
