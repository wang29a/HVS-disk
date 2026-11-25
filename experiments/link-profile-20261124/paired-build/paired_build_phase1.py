"""Build the selected paired shards and corrected legacy graph on a CPU-only lane."""

import argparse
import fcntl
from pathlib import Path
import struct
import time
from types import SimpleNamespace

import numpy as np

from revised_phase0 import materialize_mask, wait_for_topology
from run_phase0 import FILES, N, Session, digest, open_matrix, read, save

GIB = 2 ** 30
SHARD_R = 40
LEGACY_R = 2*(SHARD_R//3)


def build_program(run):
    source=run.root/"source/build_merge.cpp"
    host=run.args.host
    variant=host/"experiments/u-shape-20261113/stage2-v2"
    proof=read(variant/"validation/semantic-proof.json")
    library=variant/"bin/libpipeann-unified.a"
    if proof["status"]!="PASS" or digest(library)!=proof["binaries"]["unified"]["library_sha256"]:
        raise RuntimeError("Corrected DEG library identity mismatch")
    cfg=read(host/"experiments/metric-consistent-20261111/analysis/build.json")
    binary=run.root/"build/build-merge"
    options=cfg["flags"]+["-fno-fast-math","-ffp-contract=off"]
    command=["g++-11",*options,source,library,*cfg["link"],"-o",binary]
    run.command("build-merge",command)
    return dict(binary=str(binary),sha256=digest(binary),source_sha256=digest(source),
                library_sha256=digest(library))


def measured(run,name,cmd):
    out=run.root/"logs"/(name+".time")
    start=time.monotonic()
    run.command(name,["/usr/bin/time","-v","-o",out,*cmd])
    rss=next(int(line.rsplit(":",1)[1]) for line in out.read_text().splitlines()
             if "Maximum resident set size" in line)
    return dict(wall_seconds=time.monotonic()-start,peak_rss_kib=rss)


def checked_assignment(run,k):
    row=run.results["K%d-A1-A2"%k]["result"]
    path=Path(row["path"])
    if digest(path)!=row["assignment_sha256"]:
        raise RuntimeError("Frozen cluster labels changed")
    return path


def layout_plans(phase0):
    main=phase0["main"]
    if not main:
        return {}
    plans={"paired-main":main,"legacy-dual":dict(K=main["K"],kind="legacy")}
    no=dict(main,boundary="none",expansion_target=0.)
    no.pop("sizes",None)
    if main["expansion_target"]:
        plans["paired-no-boundary"]=no
    aux=phase0["extra_k8"]
    if aux and any(aux.get(k)!=main.get(k) for k in ("K","method","boundary","expansion_target")):
        plans["paired-k8"]=aux
    return plans


def membership_groups(run,plan):
    k=plan["K"]
    with np.load(checked_assignment(run,k)) as labels:
        if plan["kind"]=="legacy":
            for arm,group in (("x",labels["xf_first"]),("y",labels["yf_first"])):
                for i in range(k):
                    yield arm,str(i),np.flatnonzero(group==i).astype("<u4")
        else:
            mask,eps,changed,_=materialize_mask(labels,k,plan["method"],plan["expansion_target"],plan["boundary"])
            if mask is None:
                raise RuntimeError("Selected membership no longer feasible")
            if plan["expansion_target"] and (eps!=plan["epsilon"] or changed!=round(plan["additional_point_ratio"]*N)):
                raise RuntimeError("Selected memberships changed against Phase 0")
            sizes=plan.get("sizes")
            for i in range(k):
                ids=np.flatnonzero(mask&(1<<i)).astype("<u4")
                if sizes and len(ids)!=sizes[i]:
                    raise RuntimeError("Shard membership count disagrees with Phase 0")
                yield "paired",str(i),ids


def shard(run,binary,dirs,ids,r,label):
    dirs.mkdir(parents=True,exist_ok=True)
    if len(ids)<200:
        raise RuntimeError("Shard too small for the frozen DEG configuration")
    members=dirs/"members.u32"
    if members.exists() and not np.array_equal(np.fromfile(members,"<u4"),ids):
        raise RuntimeError("Resume would change shard member IDs")
    ids.tofile(members)
    run.guard(extra=(2*512*4+1024)*len(ids),memory=7*GIB)
    t=time.monotonic()
    staged=[]
    for i,name in enumerate(FILES):
        source=open_matrix(run.root/"inputs"/name)
        path=dirs/("space%d.fbin"%i)
        with path.open("wb") as out:
            out.write(struct.pack("<II",len(ids),source.shape[1]))
            for first in range(0,len(ids),8192):
                source[ids[first:first+8192]].tofile(out)
        staged.append(path)
    extract=time.monotonic()-t
    graph=dirs/"float-graph.bin"
    part=graph.with_suffix(".partial")
    result=measured(run,label,[binary,"shard",*staged,len(ids),r,part])
    part.replace(graph)
    for f in staged:
        f.unlink()
    result.update(nodes=len(ids),max_degree=r,extract_seconds=extract,
                  memberships=str(members),graph=str(graph),
                  memberships_sha256=digest(members),graph_sha256=digest(graph))
    return result


def merge_plan(path, rows):
    path.write_text("".join("%s %s\n"%(v["memberships"],v["graph"]) for v in rows),encoding="utf-8")
    for v in rows:
        if digest(Path(v["memberships"]))!=v["memberships_sha256"] or digest(Path(v["graph"]))!=v["graph_sha256"]:
            raise RuntimeError("Frozen shard changed before merge")
    return path


def relabel(run,binary,rows,folder,global_ids,label):
    folder.mkdir(exist_ok=True)
    plan=merge_plan(folder/"plan.tsv",rows)
    out=folder/"float-graph.bin"
    part=out.with_suffix(".partial")
    r=measured(run,label,[binary,"relabel",N,plan,part])
    part.replace(out)
    r.update(memberships=str(global_ids),memberships_sha256=digest(global_ids),
             graph=str(out),graph_sha256=digest(out))
    return r


def merge(run,binary,rows,folder,label):
    folder.mkdir(exist_ok=True)
    plan=merge_plan(folder/"plan.tsv",rows)
    run.guard(extra=3*GIB,memory=22*GIB)
    files=[run.root/"inputs"/f for f in FILES]
    result=measured(run,label,[binary,"merge",*files,N,plan,folder])
    result["verification"]=read(folder/"graph-audit.json")
    result["merge_phases"]=read(folder/"merge-times.json")
    result["files"]={name:digest(folder/name) for name in ("float-graph.bin","_disk.index","disk_index_graph")}
    result["folder"]=str(folder)
    return result


def fixture(run,binary):
    folder=run.root/"build/merge-fixture"
    folder.mkdir(exist_ok=True)
    n=240
    rng=np.random.RandomState(811)
    vectors=[]
    for label in ("x","y"):
        name=folder/(label+".bin")
        with name.open("wb") as f:
            f.write(struct.pack("<II",n,512))
            rng.normal(size=(n,512)).astype("<f4").tofile(f)
        vectors.append(name)
    specs=[]
    for part, ids in enumerate((np.arange(160,dtype="<u4"),np.arange(80,n,dtype="<u4"))):
        file=folder/("ids%d.u32"%part)
        graph=folder/("graph%d.bin"%part)
        ids.tofile(file)
        with graph.open("wb") as f:
            f.write(struct.pack("<QII",len(ids),512,512))
            for u in range(len(ids)):
                f.write(struct.pack("<IIIff",1,(u+1)%len(ids),1,0.,1.))
        specs.append((file,graph))
    manifest=folder/"manifest.tsv"
    manifest.write_text("".join("%s %s\n"%row for row in specs))
    run.command("merge-structure-fixture",[binary,"merge",*vectors,n,manifest,folder])
    audit=read(folder/"graph-audit.json")
    timed=read(folder/"merge-times.json")
    if (audit["nodes"]!=n or not audit["validated"] or timed["merged_nodes"]!=80
            or (folder/"disk_index_graph").stat().st_size!=8192*((n+24)//25)):
        raise RuntimeError("Merge fixture failed")
    return dict(audit=audit, merged_nodes=timed["merged_nodes"])


def existing_reference(run,binary):
    source=run.args.host/"experiments/u-shape-20261113/stage2-v2/graphs/unified"
    d=read(source/"identity.json")
    inputs=run.results["INPUTS"]["result"]
    original=source/"float-graph.bin"
    if digest(original)!=inputs["semantic_graph_sha256"] or d["graph_sha256"]!=inputs["semantic_graph_sha256"]:
        raise RuntimeError("One-shot graph identity changed")
    graph=run.args.phase0_root/"inputs/id"
    if digest(graph/"disk_index_graph")!=inputs["merge2_sha256"]:
        raise RuntimeError("One-shot Merge-2 identity changed")
    out=run.root/"analysis/ONE-SHOT-STRUCTURE.json"
    stat=measured(run,"audit-one-shot",[binary,"audit",N,original,source/"full.meta",out])
    return dict(folder=str(graph),archived_build_seconds=d["wall_seconds"],verification=read(out),
                audit_seconds=stat["wall_seconds"],files={name:digest(graph/name)
                for name in ("_disk.index","disk_index_graph")})


def run_build(run,revised):
    plans=layout_plans(revised)
    if not plans:
        run.status("PHASE0_REVISED_COMPLETE",selected=None)
        return
    program=run.step("phase1-executable",lambda:build_program(run))["result"]
    binary=program["binary"]
    if digest(Path(binary))!=program["sha256"]:
        raise RuntimeError("Frozen build binary changed")
    run.step("phase1-merge-fixture",lambda:fixture(run,binary))
    graphs=run.root/"graphs"
    graphs.mkdir(exist_ok=True)
    global_ids=graphs/"global.u32"
    expected=np.arange(N,dtype="<u4")
    if global_ids.exists() and not np.array_equal(np.fromfile(global_ids,"<u4"),expected):
        raise RuntimeError("Global ID order changed")
    if not global_ids.exists():
        expected.tofile(global_ids)
    record={"one-shot":run.step("phase1-reference",lambda:existing_reference(run,binary))["result"]}
    for label, plan in plans.items():
        root=graphs/label
        groups={"paired":[],"x":[],"y":[]}
        for arm,number,ids in membership_groups(run,plan):
            item="%s-%s-%s"%(label,arm,number)
            degree=LEGACY_R if arm in ("x","y") else SHARD_R
            result=run.step(item,lambda:shard(run,binary,root/(arm+number),ids,degree,item))["result"]
            groups[arm].append(result)
        if plan["kind"]=="legacy":
            paths=[]
            for arm in ("x","y"):
                item=label+"-relabel-"+arm
                paths.append(run.step(item,lambda:relabel(run,binary,groups[arm],root/arm,global_ids,item))["result"])
        else:
            paths=groups["paired"]
        item=label+"-merged"
        combined=run.step(item,lambda:merge(run,binary,paths,root/"merged",item))["result"]
        leaves=groups["paired"]+groups["x"]+groups["y"]
        build_time=sum(a["wall_seconds"] for a in leaves)
        extract_time=sum(a["extract_seconds"] for a in leaves)
        stage_time=combined["wall_seconds"]+sum(a["wall_seconds"] for a in paths if plan["kind"]=="legacy")
        rss=max([combined["peak_rss_kib"],*[a["peak_rss_kib"] for a in leaves+paths]])
        combined.update(shard_build_seconds=build_time,shard_extract_seconds=extract_time,
                        merge_total_seconds=stage_time,peak_rss_kib_total=rss,
                        build_points=sum(a["nodes"] for a in leaves),plans=plan,
                        afrozen_cluster_seconds=run.results["K%d-A1-A2"%plan["K"]]["seconds"])
        save(root/"build-result.json",combined)
        record[label]=combined
        save(run.root/"analysis/phase1-graphs.json",record)
    if not revised["main"]["expansion_target"]:
        record["paired-no-boundary-alias"]="paired-main"
    if revised["extra_k8"] and "paired-k8" not in record:
        record["paired-k8-alias"]="paired-main"
    save(run.root/"analysis/phase1-graphs.json",record)
    run.status("PHASE1_GRAPHS_READY",graphs=list(record))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--host",type=Path,required=True)
    p.add_argument("--phase0-root",type=Path,required=True)
    p.add_argument("--numa-node",type=int,default=0)
    p.add_argument("--phase0-session",required=True)
    p.add_argument("--peer-session",action="append",default=[])
    args=p.parse_args()
    r=Session(args)
    def queue_status(state,**extra):
        save(r.root/"phase1-queue.json",dict(state=state,**extra))
    wait_for_topology(SimpleNamespace(status=queue_status),[args.phase0_session,*args.peer_session])
    with (args.root/"queue.lock").open("a+b") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX)
        r.results=read(r.path)
        try:
            run_build(r,read(r.root/"analysis/phase0-revised.json"))
        except (Exception,KeyboardInterrupt) as exc:
            r.status("STOPPED",reason=str(exc))
            raise


if __name__=="__main__":
    main()
