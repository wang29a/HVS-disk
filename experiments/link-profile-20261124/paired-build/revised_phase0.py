"""Apply the 2026-11-23 coverage-relative gates to the frozen assignments."""

import argparse
import copy
import fcntl
from pathlib import Path
import os
import subprocess
import time

import numpy as np

from clustering import bit, coverage, member_summary, optimal
from run_phase0 import N, Session, digest, read, save, validate_inputs

MAX_WORK = 1.6
SUFFIX = {"Y": "Y", "XY": "XY"}


def boundary(mask, candidates, scores, target):
    eligible = [np.where((mask & dst) == 0, score, np.inf) for dst, score in zip(candidates, scores)]
    best = np.minimum.reduce(eligible)
    number = int(np.ceil(target * len(mask)))
    count = int(np.isfinite(best).sum())
    if count < number:
        return None, None, count
    epsilon = float(np.partition(best, number - 1)[number - 1])
    extended = mask.copy()
    for dst, score in zip(candidates, eligible):
        selected = score <= epsilon
        extended[selected] |= dst[selected]
    return extended, epsilon, int(np.count_nonzero(extended != mask))


def scenarios(k):
    for p in (.05, .10, .20):
        for mode in ("Y", "XY"):
            yield "C1-t0", p, mode
    if k == 4:
        yield "C3", .10, "XY"


def assignments(file, k, method):
    if method == "C1-t0":
        side, axis = "xb", "y0"
        match = np.arange(k,dtype="<u4")
    elif method == "C3":
        side, axis = "xf", "yf"
        tab = np.bincount((file["xf_first"]*k+file["yf_first"]).astype(np.int64),
                          minlength=k*k).reshape((k,k))
        match = optimal(tab)
    elif method == "C2":
        from clustering import greedy
        side, axis = "xf", "yf"
        tab = np.bincount((file["xf_first"]*k+file["yf_first"]).astype(np.int64),
                          minlength=k*k).reshape((k,k))
        match = greedy(tab)
    elif method == "C1-t3":
        side, axis = "xb", "y3"
        match = np.arange(k,dtype="<u4")
    else:
        raise ValueError("Unknown paired cluster method "+method)
    x = file[side+"_first"]
    y = match[file[axis+"_first"]]
    mask = bit(x)|bit(y)
    dst = (bit(file[side+"_second"]),bit(match[file[axis+"_second"]]))
    scores = (file[side+"_ratio"],file[axis+"_ratio"])
    return mask,dst,scores,int((x==y).sum())


def score(case, baseline):
    paired = zip(case["coverage"], baseline["coverage"])
    margins = [dict(alpha=a["alpha"],mean=a["mean"]-b["mean"],p01=a["p01"]-b["p01"])
               for a,b in paired]
    if len(margins)!=5:
        raise ValueError("Two coverage cohorts require all five weights")
    case["G1"] = case["work_ratio"]<=MAX_WORK
    case["G2"] = all(m["mean"]>=0 and m["p01"]>=0 for m in margins)
    case["coverage_delta_vs_dual"] = margins
    case["worst_mean"] = min(v["mean"] for v in case["coverage"])
    case["worst_p01"] = min(v["p01"] for v in case["coverage"])


def select(cases):
    return max((c for c in cases if c.get("G1") and c.get("G2")),
               key=lambda c:(c["worst_mean"],-c["work_ratio"]),default=None)


def choice_key(k, method, rate, mode):
    return "K%d-%s-%s-%dpc"%(k,method,mode,round(100*rate))


def materialize_mask(file, k, method, rate, mode):
    mask,dst,scores,paired = assignments(file,k,method)
    if rate == 0:
        return mask,0.,0,paired
    if mode == "Y":
        dst,scores = dst[1:],scores[1:]
    elif mode != "XY":
        raise ValueError("Invalid boundary mode")
    grown,epsilon,changed=boundary(mask,dst,scores,rate)
    return grown,epsilon,changed,paired


def run_phase0(run):
    known=run.results["INPUTS"]["result"]
    validate_inputs(run,known)
    exact=run.root/"analysis/exact-top40.bin"
    if digest(exact)!=run.results["A3_EXACT_TOP40"]["result"]["sha256"]:
        raise RuntimeError("Frozen coverage ground truth changed")
    ids=np.fromfile(run.root/"inputs/coverage.u32","<u4")
    nbr=np.fromfile(exact,"<u4",offset=12).reshape(len(ids),5,40)
    cases=[]
    baselines={}
    for k in (4,8,16):
        frozen=run.results["K%d-A3-A4-edge0"%k]["result"]
        source=run.results["K%d-A1-A2"%k]["result"]
        path=Path(source["path"])
        if digest(path)!=source["assignment_sha256"]:
            raise RuntimeError("Frozen cluster assignments changed")
        base=frozen["baseline"]
        baselines[str(k)]=base
        for method, values in frozen["methods"].items():
            c=copy.deepcopy(values["cases"]["0.0"])
            c.update(K=k, method=method, boundary="none", pair_rate=values["pair_rate"],kind="paired")
            score(c,base)
            cases.append(c)
        if k==16:
            continue
        with np.load(path) as data:
            for method,rate,mode in scenarios(k):
                item=choice_key(k,method,rate,mode)
                def evaluate():
                    grown,eps,changed,paired=materialize_mask(data,k,method,rate,mode)
                    common=dict(K=k,method=method,boundary=mode,kind="paired",pair_rate=paired/N,
                          expansion_target=rate,name=item)
                    if grown is None:
                        return dict(infeasible=True,eligible_nodes=changed,**common)
                    result=member_summary(grown,k)
                    result.update(epsilon=eps,additional_point_ratio=changed/N,coverage=coverage(grown,ids,nbr),
                                  **common)
                    score(result,base)
                    return result
                cases.append(run.step(item,evaluate)["result"])
    overall=select(cases)
    extra=select(c for c in cases if c["K"]==8 and c["method"]=="C1-t0")
    result=dict(cases=cases,baselines=baselines,main=overall,extra_k8=extra,
         rules=dict(G1_work_ratio_max=MAX_WORK,G2="each alpha mean and P01 at least same-K legacy",
                    reference_coverage=.95,selection="highest minimum alpha mean, then lowest total build ratio"))
    save(run.root/"analysis/phase0-revised.json",result)
    render(run,result)
    return result


def render(run,result):
    lines=["# 配对分块构建 · 修订后的 Phase 0","",
           "Laion 200 万，五个 α 使用同一批 10,000 个基库点的精确 top-40。覆盖率列按 0.1 / 0.3 / 0.5 / 0.7 / 0.9 排列。",
           "峰值内存为最大分片的构图估算。95% 只作参考。",
           "",
           "| K | 方法 | 扩展 | 新增点占比 | 配对率 | 构建量 / N | 峰值 GiB | 五权重覆盖率均值 | 五权重 P01 | G1 | G2 |",
           "|---:|---|---|---:|---:|---:|---:|---|---|---|---|"]
    def show(row,key):
        return "/".join("%.2f%%"%(100*v[key]) for v in row["coverage"])
    for k in (4,8,16):
        base=result["baselines"][str(k)]
        lines.append("| %d | 双路 k_base=1 | 无 | 0%% | — | %.3f | %.2f | %s | %s | 参照 | 参照 |"%
                     (k,base["work_ratio"],base["estimated_build_peak_gib"],show(base,"mean"),show(base,"p01")))
        for c in (row for row in result["cases"] if row["K"]==k):
            label="%s %.0f%%"%(c["boundary"],c["expansion_target"]*100) if c["expansion_target"] else "无"
            if c.get("infeasible"):
                lines.append("| %d | %s | %s | 不可达 | — | — | — | — | — | 待定 | 待定 |"%(k,c["method"],label))
                continue
            lines.append("| %d | %s | %s | %.2f%% | %.2f%% | %.3f | %.2f | %s | %s | %s | %s |"%
             (k,c["method"],label,c["additional_point_ratio"]*100,c["pair_rate"]*100,c["work_ratio"],
              c["estimated_build_peak_gib"],show(c,"mean"),show(c,"p01"),
              "通过" if c["G1"] else "未通过","通过" if c["G2"] else "未通过"))
    lines+=["","## Phase 1 入选",""]
    for key,title in (("main","主方案"),("extra_k8","K=8 对照")):
        c=result[key]
        if c:
            lines.append("%s为 K=%d、%s、%s %.0f%%，最低覆盖率 %.2f%%，构建量 %.3fN。"%(
                title,c["K"],c["method"],c["boundary"],c["expansion_target"]*100,100*c["worst_mean"],c["work_ratio"]))
        else:
            lines.append(title+"没有同时通过 G1/G2 的配置。")
    path=run.root/"analysis/PHASE0-REVISED.md"
    tmp=path.with_suffix(".partial")
    tmp.write_text("\n".join(lines)+"\n",encoding="utf-8")
    tmp.replace(path)


def active(session):
    return subprocess.run(["tmux","has-session","-t",session],stdout=subprocess.DEVNULL,
                          stderr=subprocess.DEVNULL).returncode==0


def wait_for_topology(run, peers):
    last=None
    while True:
        names=[name for name in peers if active(name)]
        if not names:
            if last:
                run.status("TOPOLOGY_WINDOW_CLOSED", sessions=last)
            return
        if names!=last:
            run.status("WAITING_FOR_TOPOLOGY_WINDOW", sessions=names)
            last=names
        time.sleep(5)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("root",type=Path)
    p.add_argument("--host",type=Path,required=True)
    p.add_argument("--blas",type=Path,required=True)
    p.add_argument("--numa-node",type=int,default=0)
    p.add_argument("--peer-session",action="append",default=[])
    p.add_argument("--phase1",action="store_true")
    args=p.parse_args()
    run=Session(args)
    with (args.root/"queue.lock").open("a+b") as fd:
        fcntl.flock(fd,fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            wait_for_topology(run,args.peer_session)
            result=run_phase0(run)
            run.status("PHASE0_REVISED_COMPLETE", selected=result["main"], extra_k8=result["extra_k8"])
            if result["main"] and args.phase1:
                from paired_build_phase1 import run_build
                run_build(run,result)
        except (Exception,KeyboardInterrupt) as exc:
            run.status("STOPPED",reason=str(exc))
            raise


if __name__=="__main__":
    main()
