"""Read-only audit of stored GT IDs; no graph repair or recall redefinition."""
from pathlib import Path
import fcntl
import json
import struct
import time
import numpy as np
import run_g0 as c


def matrix(p, dtype='<u4'):
    with Path(p).open('rb') as f:
        n,d=struct.unpack('<II',f.read(8))
    assert Path(p).stat().st_size==8+n*d*np.dtype(dtype).itemsize
    return np.memmap(str(p),mode='r',offset=8,dtype=dtype,shape=(n,d))


def main():
    out=c.ROOT/'runtime/gt-unreachable';out.mkdir(exist_ok=True)
    lock=(c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    E=c.HOST_ROOT/'experiments';ab=E/'backlink-ab-20261115/runtime'
    mg=E/'multigraph-20261113/runtime/data'
    laion=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
    oi=Path('/mnt/nvme3/wz/dataset/Openimage_disk/base_gt_round_00')
    graphs={g:c.ROOT/('runtime/revision3-diagnostics/'+g+'-native-unreachable.txt') for g in ('A1','A2','B1','B2')}
    cases=[]
    for a in (.1,.3,.5,.7,.9):
        base=mg if a in (.3,.7) else laion
        cases.append(('Laion2M',str(a),base/('gt-alpha-'+str(a)+'.bin'),None,graphs,mg))
    og={g:c.ROOT/('runtime/revision3-openimages-ab/'+g+'-unreachable.txt') for g in ('historical-A','new-B')}
    cases.append(('OpenImages','native-mixed',oi/'top10_results.ivecs',oi/'query_alpha.fvecs',og,E/'edbt-evidence-20261115/runtime/openimages/data'))
    rows=[];identities={}
    for dataset,weight,gp,ap,gs,split in cases:
        gt=matrix(gp);assert gt.shape[1]>=10
        alpha=matrix(ap,'<f4')[:,0] if ap else None
        identities[str(gp)]=c.sha(gp)
        if ap:identities[str(ap)]=c.sha(ap)
        cohorts={'all':np.arange(len(gt))}
        for name in ('tune','report'):
            p=split/(name+'-ids.txt');cohorts[name]=np.array([int(v) for v in p.read_text().split()])
            identities[str(p)]=c.sha(p)
        for graph,up in gs.items():
            u=np.array([int(v) for v in up.read_text().split()],dtype='<u4')
            identities[str(up)]=c.sha(up)
            missing=np.isin(gt[:,:10],u).sum(axis=1)
            for cohort,ids in cohorts.items():
                groups={'all-weights':ids}
                if alpha is not None:
                    # Native OpenImages is a mixed workload, not five frozen-alpha GTs.
                    # Fixed bins avoid thousands of singleton float weights.
                    groups.update({label:ids[(alpha[ids]>=lo)&(alpha[ids]<(hi if hi<1 else 1.00001))]
                                   for label,lo,hi in [('0-.2',0,.2),('.2-.4',.2,.4),('.4-.6',.4,.6),('.6-.8',.6,.8),('.8-1',.8,1)]})
                for band,ix in groups.items():
                    if not len(ix):continue
                    total=int(missing[ix].sum());denom=int(len(ix)*10)
                    rows.append(dict(dataset=dataset,weight=weight,alpha_band=band,graph=graph,cohort=cohort,
                        queries=len(ix),unreachable_nodes=len(u),missing_gt_entries=total,gt_entries=denom,
                        affected_queries=int(np.count_nonzero(missing[ix])),id_recall_ceiling=1-total/denom,
                        ceiling_loss_percentage_points=100*total/denom))
            details=[dict(query_id=int(q),alpha=float(alpha[q]) if alpha is not None else float(weight),
                unreachable_gt_ids=[int(v) for v in gt[q,:10] if v in set(u.tolist())]) for q in np.flatnonzero(missing)]
            (out/(dataset+'-'+weight+'-'+graph+'-affected.json')).write_text(json.dumps(details,indent=2)+'\n')
    result=dict(status='STORED_GT_ID_COUNTS_COMPLETE',rows=rows,inputs=identities,updated_unix=time.time(),
        scope='unfiltered single graph, fixed metadata entry points, node-only expansion; no auxiliary discovery',
        limitations=['ID ceiling is not a distance-tie-aware ceiling','OpenImages native mixed-alpha bins are not fixed-alpha experiments'],
        next='float64 full-base tie audit and fixed-alpha OpenImages GT')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([v for v in rows if v['cohort']=='report' and v['alpha_band']=='all-weights'],indent=2))


if __name__=='__main__':main()
