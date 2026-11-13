"""Local ideal two-index union coverage; not a two-index runtime benchmark."""
import json
import hashlib
from pathlib import Path
import sys
import numpy as np
from analyze_distance_scale import matrix

ROOT=Path(__file__).resolve().parent.parent
BUDGETS=[10,20,40,80,160,320,640,1280]


def stable_top(scores,k):
    threshold=np.partition(scores,k-1)[k-1]
    lower=np.flatnonzero(scores<threshold)
    equal=np.flatnonzero(scores==threshold)[:k-len(lower)]
    ids=np.concatenate([lower,equal])
    return ids[np.lexsort((ids,scores[ids]))]


def main(base_path):
    folder=ROOT/'snapshots/openimage'
    meta=json.loads((folder/'manifest.json').read_text())
    qx,qy=matrix(folder/'query_img.bin'),matrix(folder/'query_text.bin')
    gt=matrix(folder/meta['cases'][0]['gt'],np.uint32)
    alpha=matrix(folder/meta['cases'][0]['alpha'])[:,0]
    base=matrix(base_path)
    if base.shape!=(456699,1536) or qx.shape[1]!=768:raise RuntimeError('full base identity')
    sampled_ids=matrix(folder/'base_ids.bin',np.uint32)[:,0]
    if not np.array_equal(base[sampled_ids,:768],matrix(folder/'base_img.bin')) or not np.array_equal(base[sampled_ids,768:],matrix(folder/'base_text.bin')):
        raise RuntimeError('full base differs from original-ID vector snapshot')
    fingerprint=hashlib.sha256()
    with Path(base_path).open('rb') as source:
        for chunk in iter(lambda:source.read(8<<20),b''):fingerprint.update(chunk)
    norms=np.empty((2,len(base)),dtype=np.float64)
    for first in range(0,len(base),32768):
        block=np.asarray(base[first:first+32768],dtype=np.float64)
        norms[0,first:first+len(block)]=np.sum(block[:,:768]**2,axis=1)
        norms[1,first:first+len(block)]=np.sum(block[:,768:]**2,axis=1)
    rows=[]
    for start in range(0,len(qx),8):
        x,y=np.asarray(qx[start:start+8],dtype=np.float64),np.asarray(qy[start:start+8],dtype=np.float64)
        sx=np.empty((len(x),len(base)),dtype=np.float64);sy=np.empty_like(sx)
        for first in range(0,len(base),32768):
            block=np.asarray(base[first:first+32768],dtype=np.float64);end=first+len(block)
            sx[:,first:end]=np.maximum(0,np.sum(x*x,axis=1)[:,None]+norms[0,first:end]-2*x.dot(block[:,:768].T))
            sy[:,first:end]=np.maximum(0,np.sum(y*y,axis=1)[:,None]+norms[1,first:end]-2*y.dot(block[:,768:].T))
        for j in range(len(x)):
            ids_x,ids_y=stable_top(sx[j],max(BUDGETS)),stable_top(sy[j],max(BUDGETS))
            truth=set(map(int,gt[start+j]));points=[]
            for M in BUDGETS:
                a,b=set(map(int,ids_x[:M])),set(map(int,ids_y[:M]))
                points.append(dict(per_space_k=M,union_candidates=len(a|b),covered_gt=len(truth&(a|b)),
                    covered_x=len(truth&a),covered_y=len(truth&b)))
            rows.append(dict(query_id=meta['query_ids'][start+j],alpha=float(alpha[start+j]),points=points))
        (ROOT/'analysis/dual-index-coverage-progress.json').write_text(json.dumps(dict(queries_done=len(rows),total=len(qx)))+'\n')
    means=[]
    for i,M in enumerate(BUDGETS):
        mean={k:float(np.mean([row['points'][i][k] for row in rows])) for k in ('union_candidates','covered_gt','covered_x','covered_y')}
        means.append(dict(per_space_k=M,mean_union_candidates=mean['union_candidates'],recall=mean['covered_gt']*10,
                          x_only_coverage=mean['covered_x']*10,y_only_coverage=mean['covered_y']*10))
    result=dict(dataset='OpenImages',queries=len(qx),cohort='same 256 held-out queries as distance-scale analysis',
        gt='existing native mixed-alpha GT',precision='float64 exhaustive per-space scores',
        base_sha256=fingerprint.hexdigest(),sampled_original_ID_vectors_checked=True,
        tie_break='ascending distance then original ID',records=means,per_query=rows,
        interpretation='Ideal exact top-M per space union coverage. Does not measure approximate index cost, I/O or latency; cannot prove one architecture necessary.')
    (ROOT/'analysis/dual-index-oracle-coverage.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(means,indent=2))


if __name__=='__main__':main(sys.argv[1])
