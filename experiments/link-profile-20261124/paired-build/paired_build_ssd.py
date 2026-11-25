"""Qualify and time the Phase 1 graphs after the detached construction queue."""

import argparse
import fcntl
import os
from pathlib import Path
import shlex
from statistics import mean, median
import time

import numpy as np

from instrument import instrument
from layout import digest, validate_table, write_layout
from phase0 import DEPTHS, NODES, WEIGHTS, matrix, read, save
from phase0b_common import Batch, Fatal, LayoutFailure
from revised_phase0 import active, wait_for_topology
from timing_0b import (METRICS, copy_ssd, near_targets, quiet, read_native_witness,
                       stage_queries, timed_process)
from traces import read_trace, validate_results


DUR = 10
TARGETS = (.9,.95)
TIMING_FIELDS = (*METRICS,"expanded_nodes")


def compile_trace(run, driver):
    src = run.host/"source/src/search/starling_page_rerank_search.cpp"
    archive = run.host/"build-separated"
    lib = archive/"src/libpipeann.a"
    flags = archive/"src/CMakeFiles/pipeann.dir/flags.make"
    source = instrument(src.read_text(encoding="utf-8"))
    fusion = "alpha * d_e + (1 - alpha) * d_l"
    if source.count(fusion)!=2:
        raise Fatal("Unexpected PQ source shape")
    source=source.replace(fusion,"alpha * std::sqrt(std::max(0.f, d_e)) + (1 - alpha) * std::sqrt(std::max(0.f, d_l))")
    stem=run.root/"build/trace-sqrt.cpp"
    stem.write_text(source,encoding="utf-8")
    opts=[x for ln in flags.read_text().splitlines()
          if ln.startswith(("CXX_DEFINES ","CXX_INCLUDES ","CXX_FLAGS "))
          for x in shlex.split(ln.split("=",1)[1])]
    shared=driver.parent
    out=run.root/"build/native-trace-sqrt"
    libdir=os.environ.get("HVS_OPENBLAS_LIBDIR")
    openblas=["-L"+libdir] if libdir else []
    run.command(["g++-11",*opts,"-fno-fast-math","-ffp-contract=off","-I"+str(shared),
                 stem,driver,lib,*openblas,"-laio",
                 "-ljemalloc","-lopenblas","-pthread","-o",out],"compile-paired-build-trace")
    return dict(path=str(out),sha256=digest(out),source_sha256=digest(stem))


def permit(run,name,operation):
    value=run.take(name,operation,0)
    if value is None:
        raise Fatal("Required stage failed: "+name)
    return value


def symlink(path,target):
    if path.is_symlink():
        if path.resolve()!=target.resolve():
            raise Fatal("Index symlink changed: "+str(path))
    else:
        path.symlink_to(target)


def prepare_layouts(run,graphs,ssd):
    shared=run.starling/"_partition.bin.aligned"
    raw=np.fromfile(shared,"<u4",offset=24)
    table=raw.reshape(-1,26)[:,1:].copy()
    validate_table(table,NODES)
    layouts={}
    for label,graph in graphs.items():
        if not isinstance(graph,dict):
            continue
        raw=Path(graph["folder"])
        for name,h in graph["files"].items():
            if digest(raw/name)!=h:
                raise Fatal("Validated graph changed: "+label+" "+name)
        workspace=run.root/"work"/label
        workspace.mkdir(parents=True,exist_ok=True)
        source=workspace/"original"
        source.mkdir(exist_ok=True)
        for path in (raw/"_disk.index",raw/"disk_index_graph",run.base/"disk_index_data"):
            symlink(source/path.name,path)
        for p in run.base.iterdir():
            if "_pq_" in p.name:
                symlink(source/p.name,p)
        target=workspace/"starling"
        proof=workspace/"identity.json"
        if not proof.is_file():
            if target.exists():
                raise Fatal("A partial physical layout needs readback: "+label)
            write_layout(source,target,table,NODES)
            save(proof,dict(graph=digest(target/"disk_index_graph"),
                            layout=digest(target/"_partition.bin.aligned"),source=graph["files"]))
        v=read(proof)
        if v["source"]!=graph["files"] or digest(target/"disk_index_graph")!=v["graph"] or (
            digest(target/"_partition.bin.aligned")!=v["layout"]):
            raise Fatal("Reordered output identity changed: "+label)
        layouts[label]=(target,v)
    ssd.mkdir(parents=True,exist_ok=True)
    copied={}
    start=time.time()
    def copy(src,dest,h):
        copy_ssd(run,src,dest,h)
        copied[str(dest)]=h
    coord=ssd/"coordinates.bin"
    copy(run.base/"disk_index_data",coord,run.historical_identity["coordinates_sha256"])
    for label,(root,v) in layouts.items():
        out=ssd/label
        out.mkdir(exist_ok=True)
        for name,key in (("disk_index_graph","graph"),("_partition.bin.aligned","layout")):
            copy(root/name,out/name,v[key])
        for p in root.iterdir():
            if "_pq_" in p.name or p.name=="_disk.index":
                symlink(out/p.name,p)
        symlink(out/"disk_index_data",coord)
    r=dict(labels=list(layouts),files=copied,started=start,completed=time.time(),
           permutation_sha256=digest(shared))
    save(run.analysis/"ssd-copy-identity.json",r)
    return r


def report_vectors(run):
    base=run.root/"inputs/ssd"
    return tuple(base/key for key in ("query_img_emb.fvecs","query_text_emb.fvecs"))


def trace_job(run,trace,folder,label,medium,rows):
    f=run.root/"trace"
    f.mkdir(exist_ok=True)
    spec=f/(label+".tsv")
    out=[]
    with spec.open("w") as s:
        for a,depth,ids,key in rows:
            dest=f/(label+"-"+key+".bin")
            s.write("%s %s 0 %s %s\n"%(a,depth,ids,dest))
            out.append(dest)
    run.command([trace,folder,*report_vectors(run),medium,spec],"trace-"+label)
    return out


def tune_graph(run,trace,folder,label):
    rows=[(a,depth,run.holdout["tune"],"%s-L%d"%(a,depth)) for a in WEIGHTS for depth in DEPTHS]
    raw=trace_job(run,trace,folder,"tune-"+label,"dram",rows)
    curves={str(a):[] for a in WEIGHTS}
    gt={a:matrix(run.truth[a],"<u4") for a in WEIGHTS}
    for (a,depth,_,_),f in zip(rows,raw):
        result=read_trace(f,NODES)
        hit=sum(len(set(r["result"]) & set(gt[a][r["id"],:10])) for r in result)
        curves[str(a)].append(dict(L=depth,recall=hit/(len(result)*10)))
    plan={}
    for a in WEIGHTS:
        curve=curves[str(a)]
        base=near_targets(curve)
        edge=DEPTHS.index(base[-1])
        plan[str(a)]=base+list(DEPTHS[edge+1:edge+3])
    return dict(curves=curves,plan=plan)


def gate(run,trace,folder,label,plan):
    rows=[]
    for a in WEIGHTS:
        mid=plan[str(a)][len(plan[str(a)])//2]
        rows.append((a,mid,run.holdout["report"],str(a)))
    inmem=trace_job(run,trace,folder,label+"-report-dram","dram",rows)
    ondisk=trace_job(run,trace,folder,label+"-report-disk","disk",rows)
    # Disk and DRAM witnesses use distinct filenames even when the query set is unchanged.
    vectors=[matrix(v) for v in report_vectors(run)]
    coords=np.memmap(folder/"disk_index_data",dtype="<f4",mode="r",shape=(NODES,1024))
    report={}
    for a,f,g in zip(WEIGHTS,inmem,ondisk):
        if digest(f)!=digest(g):
            raise LayoutFailure("DRAM versus SSD trace mismatch at %s alpha=%s"%(label,a))
        report[str(a)]=validate_results(read_trace(g,NODES),(coords[:,:512],coords[:,512:]),vectors,a)
    return report


def match(points,target):
    ordered=sorted(points,key=lambda p:p["L"])
    for row in ordered:
        if abs(row["recall"]-target)<1e-12:
            return {k:row[k] for k in TIMING_FIELDS}
    for l,r in zip(ordered,ordered[1:]):
        if l["recall"]<target<r["recall"]:
            t=(target-l["recall"])/(r["recall"]-l["recall"])
            return {k:l[k]+t*(r[k]-l[k]) for k in TIMING_FIELDS}
    return None


def final_report(run,graphs,labels):
    out=["# 配对分块构建 · Laion 200 万 · Phase 1","",
         "所有图使用 Merge-2、同一冻结 Starling 节点顺序和同一 mode=6 执行器。两空间完整 PQ 距离先开方再融合。"
         "使用各自图的入口、相同 PQ 和查询集。SSD 四线程，三轮交替，同批上盘，权重倒序。",
         "",
         "| 构建方案 | 分片累计点数 | 建图秒 | 分片提取秒 | 合并总秒 | 峰值 RSS GiB | 平均度数 | 不可达点 | 多分片点占比 |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in labels:
        g=graphs[label]
        v=g["verification"]
        if "build_points" in g:
            out.append("| %s | %d | %.2f | %.2f | %.2f | %.2f | %.2f | %d | %.2f%% |"%
                       (label,g["build_points"],g["shard_build_seconds"],g["shard_extract_seconds"],
                        g["merge_total_seconds"],g["peak_rss_kib_total"]/2**20,v["mean_degree"],v["unreachable"],
                        g["merge_phases"]["merged_fraction"]*100))
        else:
            out.append("| %s | %d | %.2f（既有归档） | — | — | — | %.2f | %d | — |"%
                       (label,NODES,g["archived_build_seconds"],v["mean_degree"],v["unreachable"]))
    for k,v in graphs.items():
        if isinstance(v,str):
            out.append("配置 %s 与 %s 相同，复用同一张图。"%(k,v))
    x,y=graphs["paired-main"],graphs["legacy-dual"]
    def total(g):
        return g["shard_build_seconds"]+g["shard_extract_seconds"]+g["merge_total_seconds"]
    out+=["","两组复用了 Phase 0 的聚类结果。账面构建节省 %.2f%%，分母为双路的分片提取、构建、合并总时长。"%
          (100*(1-total(x)/total(y))),
          "区间段数直方图、阶段计时和未达门槛项均在原始 JSON 中。","",
          "| α | 召回 | 图 | 完整轮次 | 均值 ms | P99 ms | QPS | 展开节点 | 拓扑页 | 坐标页 |",
          "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    results={}
    for target in TARGETS:
        for a in WEIGHTS:
            for label in labels:
                matches={}
                for r in range(1,4):
                    prefix="point-r%d-%s-%s-"%(r,a,label)
                    values=[v["result"] for k,v in run.results.items()
                            if k.startswith(prefix) and v["state"]=="complete"]
                    m=match(values,target)
                    if m is not None:
                        matches[str(r)]=m
                results["%s-%s-%s"%(target,a,label)]=matches
                keys=("latency_us","p99_us","qps","expanded_nodes","topology_pages","coordinate_pages")
                cells=["%.2f"%(median(p[k] for p in matches.values())/(1000 if k.endswith("_us") else 1))
                       for k in keys] if matches else ["—"]*len(keys)
                out.append("| %s | %d%% | %s | %d/3 | %s |"%(a,100*target,label,len(matches)," | ".join(cells)))
    if all(len(results["0.95-%s-%s"%(a,k)])==3 for a in WEIGHTS for k in ("paired-main","one-shot")):
        ratios=[]
        for a in WEIGHTS:
            p=results["0.95-%s-paired-main"%a]
            q=results["0.95-%s-one-shot"%a]
            ratios.append(median(p[r]["latency_us"]/q[r]["latency_us"]-1 for r in p))
        out+=["","主方案相对全量图的五权重延迟变化为 %s。各权重 ±5%% 门槛 %s。"%
              (" / ".join("%+.2f%%"%(100*x) for x in ratios),"通过" if all(abs(x)<=.05 for x in ratios) else "未通过")]
    path=run.analysis/"RESULTS.md"
    tmp=path.with_suffix(".partial")
    tmp.write_text("\n".join(out)+"\n",encoding="utf-8")
    tmp.replace(path)
    save(run.analysis/"phase1.json",results)
    return results


def run_ssd(run,build_root,ssd,timing_proof):
    origin=read(build_root/"analysis/phase1-graphs.json")
    pair=read(timing_proof)["executables"]["result"]
    binary=Path(pair["K2"])
    if digest(binary)!=pair["K2_sha256"]:
        raise Fatal("Corrected mode=6 executable changed")
    staged=permit(run,"paired-copy",lambda:prepare_layouts(run,origin,ssd))
    for f,h in staged["files"].items():
        if digest(Path(f))!=h:
            raise Fatal("Staged physical file changed")
    permit(run,"paired-queries",lambda:stage_queries(run))
    driver=run.root.parent/"source/native_trace.cpp"
    trace=permit(run,"paired-trace-build",lambda:compile_trace(run,driver))
    if digest(Path(trace["path"]))!=trace["sha256"]:
        raise Fatal("Trace executable changed")
    plans={}
    for label in staged["labels"]:
        arm=ssd/label
        plans[label]=permit(run,"tune-"+label,lambda:tune_graph(run,trace["path"],arm,label))["plan"]
        permit(run,"parity-"+label,lambda:gate(run,trace["path"],arm,label,plans[label]))
    permit(run,"paired-idle",lambda:quiet(run,dict(completed=time.time())))
    out=run.root/"metrics"
    out.mkdir(exist_ok=True)
    for retry in (0,1):
        for r in range(1,4):
            labels=staged["labels"]
            offset=(r-1)%len(labels)
            ordered=labels[offset:]+labels[:offset]
            for a in reversed(WEIGHTS):
                for d in sorted({d for k in labels for d in plans[k][str(a)]}):
                    for label in ordered:
                        if d not in plans[label][str(a)]:
                            continue
                        key="point-r%d-%s-%s-L%d"%(r,a,label,d)
                        def measure():
                            tag=key+"-attempt%d"%(retry+1)
                            path=out/(tag+".json")
                            arm=ssd/label
                            cmd=[binary,arm,*report_vectors(run),run.holdout["report"],
                                 run.truth[a],a,d,DUR,path,arm]
                            v=timed_process(run,cmd,tag,path)
                            witness=read_native_witness(Path(str(path)+".parity"))
                            v["expanded_nodes"]=mean(len(x["expanded"]) for x in witness)
                            return v
                        run.take(key,measure,retry)
                final_report(run,origin,labels)
    results=final_report(run,origin,staged["labels"])
    incomplete=[k for k,v in results.items() if len(v)!=3]
    run.status("PHASE1_COMPLETE" if not incomplete else "PHASE1_COMPLETE_WITH_GAPS", missing=incomplete)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ("build_root","host","previous","ssd","timing_proof"):
        p.add_argument(name,type=Path)
    p.add_argument("--build-session",required=True)
    p.add_argument("--peer-session",action="append",default=[])
    args=p.parse_args()
    root=args.build_root/"ssd"
    for part in ("build","logs","inputs"):
        (root/part).mkdir(parents=True,exist_ok=True)
    run=Batch(root,args.host,args.previous)
    run.task="PairedBuildSSD"
    try:
        wait_for_topology(run,[args.build_session,*args.peer_session])
        build=read(args.build_root/"status.json")
        if build["state"]=="PHASE0_REVISED_COMPLETE" and build.get("selected") is None:
            run.status("SKIPPED_BY_PHASE0")
            return
        if build["state"]!="PHASE1_GRAPHS_READY":
            raise Fatal("Construction did not finish: "+str(build))
        run.status("WAITING_FOR_SSD_QUEUE")
        lock=run.experiments/"budget-reorder-20261103/queue.lock"
        while True:
            wait_for_topology(run,args.peer_session)
            with lock.open("r+b") as f:
                fcntl.flock(f,fcntl.LOCK_EX)
                if any(active(name) for name in args.peer_session):
                    continue
                run_ssd(run,args.build_root,args.ssd,args.timing_proof)
                break
    except (Exception,KeyboardInterrupt) as exc:
        run.status("STOPPED",reason=str(exc))
        raise


if __name__=="__main__":
    main()
