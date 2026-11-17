"""Exhaustive float64 audit. Never edits source GT or graph artifacts."""
from pathlib import Path
import fcntl
import json
import time
import numpy as np
import run_g0 as c
from gt_unreachable import matrix
from quality_metrics import ceilings


def save(out,name,value):
    (out/name).write_text(json.dumps(value,indent=2)+'\n')


def main():
    out=c.ROOT/'runtime/gt-ties-v1';out.mkdir(exist_ok=True)
    lock=(c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    save(out,'status.json',dict(stage='waiting_global_lock'))
    fcntl.flock(lock,fcntl.LOCK_EX)
    E=c.HOST_ROOT/'experiments';MG=E/'multigraph-20261113/runtime/data'
    laion=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
    oi=Path('/mnt/nvme3/wz/dataset/Openimage_disk')
    configs=[('Laion2M',laion,MG,{g:c.ROOT/('runtime/revision3-diagnostics/'+g+'-native-unreachable.txt') for g in ('A1','A2','B1','B2')}),
             ('OpenImages',oi,E/'edbt-evidence-20261115/runtime/openimages/data',{g:c.ROOT/('runtime/revision3-openimages-ab/'+g+'-unreachable.txt') for g in ('historical-A','new-B')})]
    summaries=[];identities={}
    try:
        for dataset,data,split,graphs in configs:
            completed=out/(dataset+'-complete.json')
            if completed.exists():raise RuntimeError('existing exhaustive audit: refuse overwrite')
            bases=[];queries=[]
            for space in ('img','text'):
                bp=data/('base_'+space+'_emb.fvecs');qp=data/('query_'+space+'_emb.fvecs')
                identities[str(bp)]=c.sha(bp);identities[str(qp)]=c.sha(qp)
                bases.append(np.asarray(matrix(bp,'<f4'),dtype=np.float64))
                queries.append(np.asarray(matrix(qp,'<f4'),dtype=np.float64))
            bx,by=bases;qx,qy=queries;normx=(bx*bx).sum(1);normy=(by*by).sum(1)
            U={g:set(map(int,p.read_text().split())) for g,p in graphs.items()}
            for p in graphs.values():identities[str(p)]=c.sha(p)
            cohorts={'all':set(range(len(qx)))}
            for cohort in ('tune','report'):
                p=split/(cohort+'-ids.txt');cohorts[cohort]=set(map(int,p.read_text().split()));identities[str(p)]=c.sha(p)
            cases=[]
            for a in (.1,.3,.5,.7,.9):
                if dataset=='Laion2M':
                    base=MG if a in (.3,.7) else laion;gp=base/('gt-alpha-'+str(a)+'.bin')
                    gt=matrix(gp);identities[str(gp)]=c.sha(gp)
                else:gt=None
                cases.append((str(a),np.full(len(qx),np.float32(a),dtype=np.float64),gt))
            if dataset=='OpenImages':
                gp=oi/'base_gt_round_00/top10_results.ivecs';ap=oi/'base_gt_round_00/query_alpha.fvecs'
                cases.append(('native-mixed',np.asarray(matrix(ap,'<f4')[:,0],dtype=np.float64),matrix(gp)))
                identities[str(gp)]=c.sha(gp);identities[str(ap)]=c.sha(ap)
            rows={label:[] for label,_,_ in cases};generated={label:[] for label,_,gt in cases if gt is None}
            for first in range(0,len(qx),16):
                x=qx[first:first+16];y=qy[first:first+16]
                dx=np.sqrt(np.maximum(0,(x*x).sum(1)[:,None]+normx[None,:]-2*x.dot(bx.T)))
                dy=np.sqrt(np.maximum(0,(y*y).sum(1)[:,None]+normy[None,:]-2*y.dot(by.T)))
                for label,weights,gt in cases:
                    scores=weights[first:first+len(x),None]*dx+(1-weights[first:first+len(x),None])*dy
                    for j,dist in enumerate(scores):
                        qi=first+j;tau=float(np.partition(dist,9)[9]);strict=np.flatnonzero(dist<tau);boundary=np.flatnonzero(dist==tau)
                        ids=np.concatenate((strict[np.lexsort((strict,dist[strict]))],boundary[:10-len(strict)]))
                        truth=[int(v) for v in (gt[qi,:10] if gt is not None else ids)]
                        if gt is None:generated[label].append(truth)
                        provided=np.asarray(truth,dtype=np.int64)
                        row=dict(query_id=qi,alpha=float(weights[qi]),tau=tau,strict_count=len(strict),boundary_count=len(boundary),
                            GT_ids=truth,canonical_ids=ids.tolist(),GT_outside_exact_threshold=int(np.sum(dist[provided]>tau)),
                            GT_max_excess=float(max(0,np.max(dist[provided])-tau)),
                            GT_outside_1e_10_threshold=int(np.sum(dist[provided]>tau+1e-10)),
                            stored_top10_has_equal_distances=len(set(dist[provided].tolist()))<10,graphs={})
                        strict_set=set(map(int,strict));boundary_set=set(map(int,boundary))
                        # A tolerance band is diagnostic only, never used to change the score.
                        row['near_boundary_1e_10_count']=int(np.count_nonzero(np.abs(dist-tau)<=1e-10))
                        for graph,u in U.items():
                            rs=len(strict_set&u);rb=len(boundary_set&u)
                            row['graphs'][graph]=ceilings(truth,u,len(strict),rs,len(boundary),rb)
                            row['graphs'][graph].update(strict_unreachable=sorted(strict_set&u),boundary_unreachable=sorted(boundary_set&u))
                        rows[label].append(row)
                save(out,'status.json',dict(stage='exhaustive',dataset=dataset,queries_done=min(first+16,len(qx)),queries=len(qx),updated_unix=time.time()))
            for label,weights,gt in cases:
                save(out,dataset+'-'+label+'-queries.json',rows[label])
                if gt is None:
                    # New canonical GT lives in this experiment, never the dataset directory.
                    import struct
                    a=np.asarray(generated[label],dtype='<u4')
                    with (out/(dataset+'-gt-'+label+'.bin')).open('wb') as f:f.write(struct.pack('<II',*a.shape));a.tofile(f)
                for cohort,cohort_ids in cohorts.items():
                    selected=[r for r in rows[label] if r['query_id'] in cohort_ids]
                    for graph in U:
                        entry=dict(dataset=dataset,weight=label,graph=graph,cohort=cohort,queries=len(selected),
                            queries_with_boundary_alternatives=sum(r['strict_count']+r['boundary_count']>10 for r in selected),
                            queries_with_GT_distance_ties=sum(r['stored_top10_has_equal_distances'] for r in selected),
                            GT_entries_outside_true_threshold=sum(r['GT_outside_exact_threshold'] for r in selected),
                            GT_entries_outside_1e_10_threshold=sum(r['GT_outside_1e_10_threshold'] for r in selected),
                            GT_max_distance_excess=max(r['GT_max_excess'] for r in selected))
                        for metric in ('id_ceiling','rank_tie_ceiling','distance_threshold_ceiling'):
                            entry[metric]=float(np.mean([r['graphs'][graph][metric] for r in selected]));entry[metric+'_loss_pp']=100*(1-entry[metric])
                        summaries.append(entry)
            save(out,dataset+'-complete.json',dict(status='COMPLETE',rows=[r for r in summaries if r['dataset']==dataset]))
            del bx,by,bases,dx,dy,scores
        save(out,'results.json',dict(status='EXHAUSTIVE_TIE_AUDIT_COMPLETE',rows=summaries,inputs=identities,
            arithmetic='float64 exhaustive norm-dot squared L2, clamp at zero, sqrt, float32 alpha promoted to float64; exact equality',
            scope='single graph, metadata entries, no alpha filtering, node-only discovery',
            score_definitions='ID; threshold acceptance; rank-slot tie-aware. Original gates unchanged.'))
        save(out,'status.json',dict(stage='COMPLETE',updated_unix=time.time()))
    except Exception as e:
        save(out,'status.json',dict(stage='STOPPED',reason=str(e),updated_unix=time.time()));raise


if __name__=='__main__':main()
