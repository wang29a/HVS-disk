"""Local float64 distances and neighbor-conditioned contribution calibration."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
ALPHAS=[.1,.3,.5,.7,.9]


def matrix(p,dtype=np.float32):
    import struct
    with Path(p).open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    if Path(p).stat().st_size!=8+n*d*4:raise RuntimeError('matrix size')
    return np.memmap(str(p),dtype=dtype,mode='r',offset=8,shape=(n,d))


def stat(v):
    v=np.asarray(v)
    finite=v[np.isfinite(v)]
    if not len(finite):return dict(valid_count=0,undefined_count=int(v.size))
    return dict(mean=float(finite.mean()),median=float(np.median(finite)),p10=float(np.percentile(finite,10)),
                p90=float(np.percentile(finite,90)),valid_count=int(finite.size),undefined_count=int(v.size-finite.size))


def shares(dx,dy,alpha):
    total=alpha*dx+(1-alpha)*dy
    if np.any(total<0):raise RuntimeError('negative combined distance')
    # Exact query/base duplicates have zero contribution in both terms.
    # Their share is undefined, not 0.5; keep their count and exclude shares.
    return np.divide(alpha*dx,total,out=np.full_like(total,np.nan),where=total>0)


def finite_mean(v,axis=None):
    good=np.isfinite(v)
    count=np.sum(good,axis=axis)
    total=np.sum(np.where(good,v,0),axis=axis)
    return np.divide(total,count,out=np.full_like(total,np.nan,dtype=np.float64),where=count>0)


def analyze(folder):
    manifest=json.loads((folder/'manifest.json').read_text())
    for name,info in manifest['files'].items():
        if hashlib.sha256((folder/name).read_bytes()).hexdigest()!=info['sha256']:raise RuntimeError('snapshot hash')
    ids=matrix(folder/'base_ids.bin',np.uint32)[:,0]
    x,y=matrix(folder/'base_img.bin'),matrix(folder/'base_text.bin')
    qx,qy=matrix(folder/'query_img.bin'),matrix(folder/'query_text.bin')
    references=[[] for _ in range(len(qx))];cases=[]
    for case in manifest['cases']:
        gt=matrix(folder/case['gt'],np.uint32)
        alpha=matrix(folder/case['alpha'])[:,0].astype(np.float64)
        indices=np.searchsorted(ids,gt)
        if np.any(ids[indices]!=gt):raise RuntimeError('missing sampled neighbor')
        dx=np.sqrt(np.sum((np.asarray(x[indices],dtype=np.float64)-np.asarray(qx,dtype=np.float64)[:,None,:])**2,axis=2))
        dy=np.sqrt(np.sum((np.asarray(y[indices],dtype=np.float64)-np.asarray(qy,dtype=np.float64)[:,None,:])**2,axis=2))
        native=shares(dx,dy,alpha[:,None])
        rows=[]
        for a in ALPHAS:
            v=shares(dx,dy,a)
            rows.append(dict(nominal_alpha=a,image_contribution=stat(finite_mean(v,axis=1)),
                ratio_of_aggregate_terms=float((a*dx).sum()/(a*dx+(1-a)*dy).sum()),
                interpretation='matched fixed-alpha GT' if case['fixed_alpha']==a else 'reweighting this case GT, not GT for the new alpha'))
        cases.append(dict(condition=case['condition'],fixed_alpha=case['fixed_alpha'],
            distance_image=stat(dx),distance_text=stat(dy),
            text_over_image_distance=stat(np.divide(dy,dx,out=np.full_like(dx,np.nan),where=dx>0)),
            zero_both_distances=int(np.sum((dx==0)&(dy==0))),
            undefined_query_ids=[manifest['query_ids'][j] for j in np.where(np.all((dx==0)&(dy==0),axis=1))[0]],
            native_image_contribution=stat(finite_mean(native,axis=1)),nominal_sweep_on_same_neighbors=rows,
            pairwise_balance_alpha=stat(np.divide(dy,dx+dy,out=np.full_like(dy,np.nan),where=dx+dy>0))))
        for j in range(len(qx)):references[j].extend(int(i) for i in gt[j])
    ref_dist=[]
    for j,selected in enumerate(references):
        unique=np.array(sorted(set(selected)),dtype=np.uint32);loc=np.searchsorted(ids,unique)
        dx=np.linalg.norm(np.asarray(x[loc],dtype=np.float64)-qx[j],axis=1)
        dy=np.linalg.norm(np.asarray(y[loc],dtype=np.float64)-qy[j],axis=1)
        ref_dist.append((dx,dy))
    def mean_share(alpha):return float(finite_mean(np.array([finite_mean(shares(dx,dy,alpha)) for dx,dy in ref_dist])))
    lo,hi=0.,1.
    for _ in range(60):
        mid=(lo+hi)/2
        if mean_share(mid)<.5:lo=mid
        else:hi=mid
    sweep=[dict(nominal_alpha=a,image_contribution=mean_share(a),text_contribution=1-mean_share(a)) for a in ALPHAS]
    return dict(dataset=manifest['dataset'],queries=len(qx),dimensions=[x.shape[1],y.shape[1]],
        cases=cases,reference='union of the supplied five fixed-alpha GT sets' if len(cases)>1 else 'supplied native mixed-alpha GT',
        reference_sweep=sweep,reference_mean_contribution_balance_alpha=(lo+hi)/2,
        source_query_ids=manifest['query_ids'],query_norms=dict(image=stat(np.linalg.norm(qx.astype(np.float64),axis=1)),
            text=stat(np.linalg.norm(qy.astype(np.float64),axis=1))),
        compute='local workstation float64 Euclidean distances; no additional normalization',
        limit='Neighbor-conditioned descriptive contribution. It does not redefine the metric or establish a unique global balanced alpha. OpenImages fixed-alpha GT is unavailable in this snapshot.')


def main(source):
    source=Path(source)
    reports=[analyze(source/name) for name in ('laion','openimage')]
    dest=ROOT/'analysis/distance-scale.json';dest.parent.mkdir(exist_ok=True)
    dest.write_text(json.dumps(dict(reports=reports),indent=2,allow_nan=False)+'\n')
    print(json.dumps([dict(dataset=x['dataset'],balance=x['reference_mean_contribution_balance_alpha'],sweep=x['reference_sweep']) for x in reports],indent=2))


if __name__=='__main__':main(sys.argv[1])
