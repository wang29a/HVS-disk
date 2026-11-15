"""Separate approximation and traversal losses without tuning either one."""
import json
from pathlib import Path
import numpy as np

def overlap(b,c):
    b=np.asarray(b,dtype=np.float64);c=np.asarray(c,dtype=np.float64)
    k=len(b)
    if not k or len(c)!=k or len(set(map(int,b[:,0])))!=k or len(set(map(int,c[:,0])))!=k:
        raise RuntimeError('candidate count/duplicate IDs')
    cutoff=float(b[-1,1]);strict=int(np.sum(b[:,1]<cutoff));need=k-strict
    strict_found=int(np.sum(c[:,1]<cutoff));tie_found=int(np.sum(c[:,1]==cutoff))
    raw=len(set(map(int,b[:,0]))&set(map(int,c[:,0])))/k
    eligible=(strict_found+min(tie_found,need))/k
    if eligible>1+1e-12:raise RuntimeError('inconsistent exhaustive boundary')
    return dict(raw_ID_overlap=raw,candidate_coverage=eligible,B_strictly_below_cutoff=strict,
        boundary_slots=need,C_boundary_count=tie_found,cutoff=cutoff,
        missing_B_IDs=sorted(set(map(int,b[:,0]))-set(map(int,c[:,0]))),
        strict_missing_B_IDs=[int(v[0]) for v in b if v[1]<cutoff and int(v[0]) not in set(map(int,c[:,0]))])

def summary_overlap(rows,thresholds):
    means=float(np.mean([r['candidate_coverage'] for r in rows]));minimum=float(min(r['candidate_coverage'] for r in rows))
    return dict(mean_candidate_coverage=means,min_query_candidate_coverage=minimum,
        mean_raw_ID_overlap=float(np.mean([r['raw_ID_overlap'] for r in rows])),
        passed=means+1e-12>=thresholds['mean_candidate_coverage_min'] and minimum+1e-12>=thresholds['query_candidate_coverage_min'])

def recall(ids,truth):return len(set(map(int,ids))&set(map(int,truth)))/10

def rerank(ids,x,y,qx,qy,a):
    ids=np.array(sorted(set(map(int,ids))),dtype=np.uint32)
    if len(ids)<10:raise RuntimeError('too few candidates')
    dx=np.linalg.norm(np.asarray(x[ids],dtype=np.float64)-np.asarray(qx,dtype=np.float64),axis=1)
    dy=np.linalg.norm(np.asarray(y[ids],dtype=np.float64)-np.asarray(qy,dtype=np.float64),axis=1)
    scores=a*dx+(1-a)*dy;order=np.lexsort((ids,scores))[:10]
    return ids[order],scores[order]

def exact_A(folder,base,queries,query_ids,reference,weights,source_signature):
    path=Path(folder)/'exact-A.npz';plan=Path(folder)/'exact-A-plan.json'
    signature=dict(source=source_signature,query_ids=query_ids,weights=[1.,0.]+[float(np.float32(a)) for a in weights],k=320,precision='float64; ties by ID',reference_sha256=reference.digest(reference.__file__))
    if path.exists():
        if json.loads(plan.read_text())!=signature:raise RuntimeError('exact-A input identity changed')
        with np.load(str(path),allow_pickle=False) as z:return z['ids']
    x,y=(reference.matrix(Path(base)/name) for name in ('base_img_emb.fvecs','base_text_emb.fvecs'))
    qx,qy=(reference.matrix(Path(queries)/name) for name in ('query_img_emb.fvecs','query_text_emb.fvecs'))
    ids,scores=reference.exact_batch(x,y,np.asarray(qx[query_ids],dtype=np.float64),np.asarray(qy[query_ids],dtype=np.float64),signature['weights'],320,32768)
    for i,a in enumerate(signature['weights']):
        dx=np.linalg.norm(np.asarray(x[ids[i]],dtype=np.float64)-np.asarray(qx[query_ids],dtype=np.float64)[:,None,:],axis=2)
        dy=np.linalg.norm(np.asarray(y[ids[i]],dtype=np.float64)-np.asarray(qy[query_ids],dtype=np.float64)[:,None,:],axis=2)
        if not np.allclose(a*dx+(1-a)*dy,scores[i],atol=1e-10,rtol=1e-10):raise RuntimeError('exact-A independent distances differ')
    plan.write_text(json.dumps(signature,indent=2)+'\n')
    with path.with_suffix('.partial').open('wb') as f:np.savez(f,ids=ids)
    path.with_suffix('.partial').replace(path);return ids

def analyze(raw,exact,query_ids,x,y,qx,qy,truths,weights,thresholds,old_points=None):
    methods={};spatial={'image':[],'text':[]};per_query=[];native_checks=[]
    for j,qi in enumerate(query_ids):
        r=raw[qi]
        if r['pq_score_audit']['max_relative_error']>5e-6:raise RuntimeError('ADC score numerical parity failed')
        for side in spatial:spatial[side].append(overlap(r[side]['B'],r[side]['C']))
        for z,nominal in enumerate(weights):
            a=float(np.float32(nominal));truth=truths[str(nominal)][qi];native=r['weights'][z]
            if abs(native['alpha']-a)>1e-8:raise RuntimeError('query weight mismatch')
            for method in ('M','DEG_off','DEG_on'):
                if method=='M':
                    sets={'A':np.concatenate((exact[0,j],exact[1,j])),
                        'B':[int(v[0]) for side in ('image','text') for v in r[side]['B']],
                        'C':[int(v[0]) for side in ('image','text') for v in r[side]['C']]}
                    details={side:overlap(r[side]['B'],r[side]['C']) for side in ('image','text')}
                    native_ids=native['M_native_C_ids']
                else:
                    arm='unified_off' if method=='DEG_off' else 'unified_on'
                    sets={'A':exact[z+2,j],'B':[int(v[0]) for v in native['unified_B']], 'C':[int(v[0]) for v in native[arm]['C']]}
                    details={'unified':overlap(native['unified_B'],native[arm]['C'])};native_ids=native[arm]['native_ids']
                records={}
                for layer,ids in sets.items():
                    top,dist=rerank(ids,x,y,qx[qi],qy[qi],a);records[layer]=dict(recall=recall(top,truth),top10=list(map(int,top)),candidate_count=len(set(map(int,ids))))
                for side,detail in details.items():
                    detail['missing_B_true_neighbor_IDs']=sorted(set(detail['missing_B_IDs'])&set(map(int,truth)))
                    detail['strict_missing_B_true_neighbor_IDs']=sorted(set(detail['strict_missing_B_IDs'])&set(map(int,truth)))
                record=dict(query_id=qi,alpha=nominal,method=method,layers=records,overlap=details,
                    native_C_top10=native_ids,native_C_recall=recall(native_ids,truth),
                    candidate_AB_ID_coverage=len(set(map(int,sets['A']))&set(map(int,sets['B'])))/len(set(map(int,sets['A']))))
                if method=='M':
                    record['side_GT_coverage']={side:{'A':recall(exact[s,j],truth),'B':recall([v[0] for v in r[side]['B']],truth),'C':recall([v[0] for v in r[side]['C']],truth)} for s,side in enumerate(('image','text'))}
                if method!='M' and records['A']['top10']!=list(map(int,truth)):
                    raise RuntimeError('unified exact-A top10 differs from existing GT')
                per_query.append(record);methods.setdefault((nominal,method),[]).append(record)
                if method=='M' and old_points and str(nominal) in old_points:
                    prior=old_points[str(nominal)][qi]
                    if list(map(int,prior['ids']))!=list(map(int,native_ids)):raise RuntimeError('diagnostic changed original M results at alpha '+str(nominal)+' query '+str(qi))
                    native_checks.append(dict(query_id=qi,alpha=nominal,native_ids_equal=True))
    spatial_summary={side:summary_overlap(rows,thresholds) for side,rows in spatial.items()}
    for s,side in enumerate(('image','text')):
        spatial_summary[side]['mean_exact_PQ_ID_overlap']=float(np.mean([len(set(map(int,exact[s,j]))&set(int(v[0]) for v in raw[qi][side]['B']))/len(exact[s,j]) for j,qi in enumerate(query_ids)]))
    groups=[]
    for (a,method),rows in sorted(methods.items()):
        means={l:float(np.mean([r['layers'][l]['recall'] for r in rows])) for l in ('A','B','C')}
        overlap_groups={s:summary_overlap([r['overlap'][s] for r in rows],thresholds) for s in rows[0]['overlap']}
        gap=abs(means['B']-means['C'])
        group=dict(alpha=a,method=method,recall=means,PQ_recall_loss=means['A']-means['B'],traversal_recall_loss=means['B']-means['C'],
            BC_absolute_recall_gap=gap,BC_candidates=overlap_groups,
            passed=all(g['passed'] for g in overlap_groups.values()) and gap<=thresholds['max_recall_gap']+1e-12,
            mean_candidate_AB_ID_coverage=float(np.mean([r['candidate_AB_ID_coverage'] for r in rows])),
            note='A/B is a measurement, not a pass/fail rule')
        if method=='M':
            group['side_GT_coverage']={side:{layer:float(np.mean([r['side_GT_coverage'][side][layer] for r in rows])) for layer in ('A','B','C')} for side in ('image','text')}
            group['missing_B_true_neighbors']={side:sum(len(r['overlap'][side]['missing_B_true_neighbor_IDs']) for r in rows) for side in ('image','text')}
        groups.append(group)
    return dict(status='PASS' if all(g['passed'] for g in groups) else 'FAIL',groups=groups,spaces=spatial_summary,
        thresholds=thresholds,query_ids=query_ids,per_query=per_query,original_native_result_checks=native_checks,
        limits='16 preselected correctness queries, not a performance comparison; large L unchanged; all layers use independent float64 rerank')

def markdown(result):
    text='# 三层候选损失拆分\n\nA为精确距离全库候选，B为PQ距离全库候选，C为原图搜索候选；均以float64独立精排。只有B/C参与正确性门槛，A/B是近似损失测量。\n\n'
    text+='| α | 方法 | A召回 | B召回 | C召回 | PQ净损失(pp) | 遍历净损失(pp) | 门槛 |\n|---:|---|---:|---:|---:|---:|---:|---|\n'
    for r in result['groups']:
        text+='| %.1f | %s | %.3f%% | %.3f%% | %.3f%% | %.3f | %.3f | %s |\n'%(r['alpha'],r['method'],100*r['recall']['A'],100*r['recall']['B'],100*r['recall']['C'],100*r['PQ_recall_loss'],100*r['traversal_recall_loss'],'PASS' if r['passed'] else 'FAIL')
    text+='\n| 单空间图 | 精确/PQ候选ID重合 | B/C原始ID重合 | B/C平局校正覆盖 | 最差查询覆盖 |\n|---|---:|---:|---:|---:|\n'
    for side,s in result['spaces'].items():text+='| %s | %.3f%% | %.3f%% | %.3f%% | %.3f%% |\n'%(side,s['mean_exact_PQ_ID_overlap']*100,s['mean_raw_ID_overlap']*100,s['mean_candidate_coverage']*100,s['min_query_candidate_coverage']*100)
    text+='\n| α | 空间 | A覆盖GT | B覆盖GT | C覆盖GT | B找到而C未找到的GT数 |\n|---:|---|---:|---:|---:|---:|\n'
    for r in result['groups']:
        if r['method']=='M':
            for side,s in r['side_GT_coverage'].items():text+='| %.1f | %s | %.3f%% | %.3f%% | %.3f%% | %d |\n'%(r['alpha'],side,s['A']*100,s['B']*100,s['C']*100,r['missing_B_true_neighbors'][side])
    text+='\nPQ和遍历的净损失可能为负：近似候选并不严格包含于精确top-k，不能强行裁成零。逐查询漏失ID及其GT归属见summary.json，不将候选漏失自动解释成物理I/O。\n'
    return text
