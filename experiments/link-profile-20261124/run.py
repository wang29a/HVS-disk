"""Profile only the specified shards and stop after the four-point scan."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import re
import shutil
import struct
import subprocess
import time

import numpy as np
from build import read, sha


def save(path, value):
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    tmp.replace(path)


def status(root, state, **fields):
    save(root/"status.json",dict(state=state,updated_at=time.time(),**fields))
    print(state,json.dumps(fields),flush=True)


def matrix(path):
    with path.open("rb") as f:
        shape=struct.unpack("<II",f.read(8))
    if shape[1]!=512 or path.stat().st_size!=8+4*shape[0]*shape[1]:
        raise RuntimeError("invalid frozen vector matrix: "+str(path))
    return np.memmap(path,"<f4","r",offset=8,shape=shape)


def select(old):
    progress=read(old/"progress.json")
    variants=[("legacy","legacy-dual-",500000),("k8","paired-k8-paired-",400000),
              ("k4","paired-main-paired-",None)]
    result={}
    for tag,prefix,target in variants:
        candidates=[(k,v["result"]) for k,v in progress.items() if k.startswith(prefix)
                    and (v.get("result") or {}).get("max_degree")]
        key,row=min(candidates,key=lambda x:abs(x[1]["nodes"]-target) if target is not None
                    else -x[1]["nodes"])
        mem=Path(row["memberships"])
        if sha(mem)!=row["memberships_sha256"]:
            raise RuntimeError("selected membership bytes changed")
        log=old/"logs"/(key+".log")
        matches=re.findall(r"Link time: ([0-9.]+)s",log.read_text(encoding="utf-8"))
        if len(matches)!=1:
            raise RuntimeError("ambiguous original Link time: "+key)
        result[tag]=dict(name=key,memberships=str(mem),memberships_sha256=sha(mem),nodes=row["nodes"],
                         R=row["max_degree"],old_link_s=float(matches[0]),old_log_sha256=sha(log))
    return result


def stage(root, old, choices, witness):
    source=[old/"inputs"/k for k in ("base_img_emb.fvecs","base_text_emb.fvecs")]
    expected={k:sha(k) for k in source}
    # The phase-0 proof identifies the exact matrices used for these membership files.
    published=read(old/"progress.json")["INPUTS"]["result"]
    if any(published.get(path.name)!=digest for path,digest in expected.items()):
        raise RuntimeError("frozen paired-build input hashes do not match")
    source=[matrix(p) for p in source]
    for name,r in choices.items():
        ids=np.fromfile(r["memberships"],"<u4")
        if len(ids)!=r["nodes"] or np.any(ids[1:]<=ids[:-1]) or ids[-1]>=len(source[0]):
            raise RuntimeError("invalid frozen shard membership")
        folder=root/"data"/name
        folder.mkdir(parents=True,exist_ok=True)
        material={}
        for label,selected in ((name,ids),("fixture",ids[:2048])):
            if label=="fixture" and name!="legacy":
                continue
            for i,view in enumerate(source):
                path=folder/("%s-%d.bin"%(label,i))
                with path.open("wb") as out:
                    out.write(struct.pack("<II",len(selected),512))
                    for start in range(0,len(selected),8192):
                        view[selected[start:start+8192]].tofile(out)
                material[path.name]=sha(path)
        r["material_sha256"]=material
    save(witness,dict(shards=choices,source_sha256={str(k):v for k,v in expected.items()}))
    return choices


def distribution(bins):
    total=sum(bins)
    if not total:
        raise RuntimeError("missing profile observations")
    observed=[i for i,v in enumerate(bins) if v]
    def nearest(q):
        for i, count in enumerate(bins):
            q-=count
            if q<=0:
                return i
    return dict(mean=sum(i*v for i,v in enumerate(bins))/total,min=observed[0],max=observed[-1],
                p50=nearest(total*.5),p90=nearest(total*.9),p99=nearest(total*.99),total=total)


def verify_profile(row,n,L,R,threads):
    if (row["points"],row["insertions"],row["L"],row["R"],row["init_threads"])!=(n,n-1,L,R,threads):
        raise RuntimeError("profile point count, worker count, depth or degree mismatch")
    for k,bins in row["histograms"].items():
        s=distribution(bins)
        if s["total"]!=n-1 or (k=="retained_edges" and s["max"]>R):
            raise RuntimeError("profile histogram mismatch: "+k)
    row["overflow_reprunes_total"]=sum(i*v for i,v in enumerate(row["histograms"]["overflow_reprunes"]))
    if row["overflow_reprunes_total"]>row["reverse_attempts"]:
        raise RuntimeError("overflow count exceeds reverse attempts")
    if any(v<0 for v in row["worker_ns"].values()):
        raise RuntimeError("invalid worker timing")
    row["count_summary"]={k:distribution(v) for k,v in row["histograms"].items()}
    row["thread_time_percent"]={k:100*v/sum(row["worker_ns"].values())
                                for k,v in row["worker_ns"].items()}
    return row


def obtain(root,name,binary,x,y,n,L,R,C,threads=8,profile=True):
    out=root/"runs"/name
    out.mkdir(parents=True,exist_ok=True)
    f=out/"float-graph.bin"
    result=out/"result.json"
    if result.exists():
        row=read(result)
        if sha(f)!=row["graph_sha256"] or sha(binary)!=row["binary_sha256"]:
            raise RuntimeError("existing output hash changed: "+name)
        return row
    if shutil.disk_usage(root).free<20*2**30:
        raise RuntimeError("PMEM0 cannot admit shard profile")
    args=[binary,"shard",x,y,n,R,f]+([L,C] if profile else [])
    command=["/usr/bin/time","-v","-o",out/"time.txt",*args]
    env=dict(os.environ,OMP_THREAD_LIMIT=str(threads),OMP_NUM_THREADS=str(threads),OMP_DYNAMIC="FALSE",
             OPENBLAS_NUM_THREADS="1",MKL_NUM_THREADS="1",HVS_LINK_PROFILE_JSON=str(out/"raw-profile.json"))
    status(root,"RUNNING",item=name,nodes=n,L=L,R=R,C=C,threads=threads)
    with (out/"build.log").open("w") as log:
        subprocess.run(list(map(str,command)),env=env,check=True,stdout=log,stderr=subprocess.STDOUT)
    log=(out/"build.log").read_text(encoding="utf-8")
    linked=re.findall(r"Link time: ([0-9.]+)s",log)
    if len(linked)!=1:
        raise RuntimeError("missing Link log: "+name)
    with f.open("rb") as stream:
        if stream.read(16)!=struct.pack("<QII",n,512,512) or f.stat().st_size<16+4*n:
            raise RuntimeError("incomplete graph export: "+name)
    row=verify_profile(read(out/"raw-profile.json"),n,L,R,threads) if profile else {}
    row.update(run=name,nodes=n,L=L,R=R,C=C,threads=threads,graph_sha256=sha(f),
               binary_sha256=sha(binary),link_s=float(linked[0]))
    save(result,row)
    return row


def fixture(root,binaries):
    folder=root/"data/legacy"
    x,y=folder/"fixture-0.bin",folder/"fixture-1.bin"
    proof={}
    for R in (26,40):
        rows={k:obtain(root,"fixture-%s-R%d"%(k,R),executable,x,y,2048,200,R,750,
                       threads=1,profile=k=="on") for k,executable in binaries.items()}
        equality=rows["original"]["graph_sha256"]==rows["off"]["graph_sha256"]
        enabled=rows["original"]["graph_sha256"]==rows["on"]["graph_sha256"]
        if not equality or not enabled:
            raise RuntimeError("single-thread frozen graph bytes changed at R=%d"%R)
        proof[str(R)]=dict(disabled_equal=equality,profile_equal=enabled,sha256=rows["original"]["graph_sha256"])
    save(root/"analysis/parity.json",proof)


def render(root, rows):
    def cell(s):
        return "%.1f / %d / %d / %d [%d–%d]"%(s["mean"],s["p50"],s["p90"],s["p99"],s["min"],s["max"])
    head=["# Link 分阶段剖析","","Laion 200 万原分片。分片计时使用 8 物理核、NUMA0。"
          "计时比例以五项各线程累计耗时之和为分母。Link 秒数为整片墙钟。"
          "计数单元格依次为均值、P50、P90、P99 和范围。完整直方图见结果 JSON。",
          "访问指计算距离的不同节点，展开指实际读取邻接表的次数。"
          "skyline 为新点剪枝形成的全部层数。保留边取新点剪枝后、写入当前行前的数量。",
          "冻结构建器接收 C，但当前建图路径未读取该参数。不同 C 的实测差异包含多线程建图波动。","",
          "| 分片 | L/C/R | Link 秒 | 相对历史 | 搜索% | 新点剪枝% | 反向维护% | 末尾清理% | 其余% | 访问 | 展开 | 候选 | skyline | 保留边 | 溢出重剪 | 溢出总次 |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|---|---|---|---:|"]
    for r in rows:
        profile=r["thread_time_percent"]
        counts=r["count_summary"]
        pct="%.2f%%"%r["delta_pct"] if "delta_pct" in r else "—"
        row=[r["shard"],"%d/%d/%d"%(r["L"],r["C"],r["R"]),"%.2f"%r["link_s"],pct]
        row+=["%.2f"%profile[k] for k in ("candidate_search","new_point_prune","reverse_maintenance",
                                         "cleanup","bookkeeping")]
        row += [cell(counts[k]) for k in ("search_visited","search_expanded","prune_candidates",
                                         "skyline_layers","retained_edges","overflow_reprunes")]
        row.append(str(r["overflow_reprunes_total"]))
        head.append("| "+" | ".join(row)+" |")
    (root/"analysis/REPORT.md").write_text("\n".join(head)+"\n",encoding="utf-8")
    save(root/"analysis/results.json",rows)


def run(root,old):
    proof=read(root/"build/identity.json")
    bins={k:Path(v["binary"]) for k,v in dict(proof["binaries"],original=proof["original"]).items()}
    witness=root/"analysis/inputs.json"
    if witness.exists():
        choices=read(witness)["shards"]
        for tag,r in choices.items():
            for k,h in r["material_sha256"].items():
                if sha(root/"data"/tag/k)!=h:
                    raise RuntimeError("staged shard input changed")
    else:
        choices=stage(root,old,select(old),witness)
    fixture(root,bins)
    rows=[]
    def measure(tag,L,C,baseline=False):
        r=choices[tag]
        folder=root/"data"/tag
        result=obtain(root,"%s-L%d-C%d"%(tag,L,C),bins["on"],folder/(tag+"-0.bin"),
                       folder/(tag+"-1.bin"),r["nodes"],L,r["R"],C)
        result["shard"]=r["name"]
        if baseline:
            result["original_link_s"]=r["old_link_s"]
            result["delta_pct"]=(result["link_s"]/r["old_link_s"]-1)*100
        rows.append(result)
        render(root,rows)
    for tag in ("legacy","k8","k4"):
        measure(tag,200,750,True)
    misses=[r["shard"] for r in rows if abs(r["delta_pct"])>5]
    if misses:
        status(root,"STOPPED_OVERHEAD_GATE",shards=misses)
        return
    for L,C in ((200,400),(128,750),(128,400)):
        measure("k8",L,C)
    status(root,"COMPLETE",measurements=len(rows))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    for k in ("root","old","queue_lock"):
        p.add_argument(k,type=Path)
    a=p.parse_args()
    try:
        status(a.root,"WAITING_FOR_EXCLUSIVE_WINDOW")
        with a.queue_lock.open("rb") as f:
            fcntl.flock(f,fcntl.LOCK_EX)
            run(a.root,a.old)
    except (Exception,KeyboardInterrupt) as error:
        status(a.root,"STOPPED",reason=str(error))
        raise
