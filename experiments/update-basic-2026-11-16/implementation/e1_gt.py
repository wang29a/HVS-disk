"""All E1 prefix checkpoints: float64 exhaustive GT, score then original ID."""
import json
import struct
import time
import numpy as np
from pathlib import Path
from gt_unreachable import matrix
import run_g0 as c

COUNTS=[1800000,1850000,1900000,1950000,2000000]
WEIGHTS=[.1,.3,.5,.7,.9]


def generate(out,data):
    out=Path(out);out.mkdir(exist_ok=False)
    x=np.asarray(matrix(data/'base_img_emb.fvecs','<f4'),dtype=np.float64)
    y=np.asarray(matrix(data/'base_text_emb.fvecs','<f4'),dtype=np.float64)
    qx=np.asarray(matrix(data/'query_img_emb.fvecs','<f4'),dtype=np.float64)
    qy=np.asarray(matrix(data/'query_text_emb.fvecs','<f4'),dtype=np.float64)
    assert x.shape==y.shape==(2000000,512) and qx.shape==qy.shape==(1000,512)
    normx=(x*x).sum(1);normy=(y*y).sum(1)
    truths={(n,a):np.empty((1000,10),dtype='<u4') for n in COUNTS for a in WEIGHTS}
    for first in range(0,len(qx),16):
        u=qx[first:first+16];v=qy[first:first+16]
        dx=np.sqrt(np.maximum(0,(u*u).sum(1)[:,None]+normx[None,:]-2*u.dot(x.T)))
        dy=np.sqrt(np.maximum(0,(v*v).sum(1)[:,None]+normy[None,:]-2*v.dot(y.T)))
        for a in WEIGHTS:
            alpha=float(np.float32(a));dist=alpha*dx+(1-alpha)*dy
            for n in COUNTS:
                for j,row in enumerate(dist[:,:n]):
                    tau=np.partition(row,9)[9];ids=np.flatnonzero(row<=tau)
                    truths[n,a][first+j]=ids[np.lexsort((ids,row[ids]))][:10]
        (out/'status.json').write_text(json.dumps(dict(stage='exhaustive',queries_done=min(first+16,1000),updated_unix=time.time()))+'\n')
    for (n,a),truth in truths.items():
        with (out/('gt-%d-%s.bin'%(n,a))).open('wb') as f:f.write(struct.pack('<II',1000,10));truth.tofile(f)
    for a in WEIGHTS:
        with (out/('alpha-%s.bin'%a)).open('wb') as f:f.write(struct.pack('<II',1000,1));np.full((1000,1),a,dtype='<f4').tofile(f)
    identity=dict(status='COMPLETE',counts=COUNTS,weights=WEIGHTS,
        objective='float64 exhaustive norm-dot L2, clamp zero, root then fuse with float32 alpha promoted to float64; sort score then original row ID',
        inputs={str(data/name):c.sha(data/name) for name in ('base_img_emb.fvecs','base_text_emb.fvecs','query_img_emb.fvecs','query_text_emb.fvecs')},
        outputs={p.name:c.sha(p) for p in out.glob('*.bin')})
    (out/'identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    (out/'status.json').write_text(json.dumps(dict(stage='COMPLETE',updated_unix=time.time()))+'\n')


if __name__=='__main__':
    import sys
    generate(Path(sys.argv[1]),Path(sys.argv[2]))
