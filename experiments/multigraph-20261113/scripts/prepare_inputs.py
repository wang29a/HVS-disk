"""Shared XY pages/PQ and only the missing GT; no existing input is modified."""
from pathlib import Path
import json
import os
import struct
import sys
import numpy as np
import common as c
sys.path.insert(0,str(c.U/'scripts'))
import prepare_data as p

WEIGHTS=[.1,.3,.5,.7,.9]
PQ=c.E/'laion/full'

def shared():
    out=c.RUN/'data';out.mkdir(exist_ok=True)
    identity=c.load(c.D/'identity.json')
    ids=p.matrix(c.D/'source_ids.bin','<u4')[:,0]
    n,d=p.shape(c.D/'base_img_emb.fvecs')
    if (n,d)!=(2000000,512):raise RuntimeError('Laion subset identity')
    sig=dict(subset_manifest_sha256=c.sha(c.D/'subset.json'),identity_sha256=c.sha(c.D/'identity.json'),source_ids_sha256=c.sha(c.D/'source_ids.bin'))
    for cohort,values in identity['query_cohorts'].items():(out/(cohort+'-ids.txt')).write_text('\n'.join(map(str,values))+'\n')
    stamp=out/'shared.json'
    if stamp.exists():
        saved=c.load(stamp)
        if saved['identity']!=sig:raise RuntimeError('shared input identity changed')
        for name,value in saved['files'].items():
            if c.sha(out/name)!=value['sha256']:raise RuntimeError('shared data changed '+name)
        return
    c.capacity(24*2**30,'multigraph shared coordinates/PQ and six additional compact graphs')
    x,y=p.matrix(c.D/'base_img_emb.fvecs'),p.matrix(c.D/'base_text_emb.fvecs')
    target=out/'xy.pages';tmp=out/'xy.pages.partial'
    # A coordinate is 4096B; two records fit exactly in each 8192B page.
    with tmp.open('wb') as f:
        for first in range(0,n,8192):
            end=min(n,first+8192);block=np.concatenate((x[first:end],y[first:end]),axis=1).astype('<f4');block.tofile(f)
        if n%2:f.write(bytes(4096))
        f.flush();os.fsync(f.fileno())
    tmp.replace(target)
    for space in ('emb','loc'):
        src=PQ/(space+'_pq_compressed.bin')
        # The code header describes bytes, not float32 matrix elements.
        with src.open('rb') as f:rows,chunks=struct.unpack('<II',f.read(8))
        if chunks!=64 or src.stat().st_size!=8+rows*chunks or int(ids[-1])>=rows:raise RuntimeError('full PQ identity')
        codes=np.memmap(str(src),dtype='u1',mode='r',offset=8,shape=(rows,chunks))
        name=space+'_pq_compressed.bin';part=out/(name+'.partial')
        with part.open('wb') as f:
            f.write(struct.pack('<II',n,chunks))
            for first in range(0,n,32768):np.asarray(codes[ids[first:first+32768]],dtype='u1').tofile(f)
            f.flush();os.fsync(f.fileno())
        part.replace(out/name)
        pivot=PQ/(space+'_pq_pivots.bin');dest=out/pivot.name
        if dest.exists() or dest.is_symlink():
            if dest.resolve()!=pivot.resolve():raise RuntimeError('pivot link changed')
        else:dest.symlink_to(pivot)
    files={name:dict(bytes=(out/name).stat().st_size,sha256=c.sha(out/name)) for name in ('xy.pages','emb_pq_compressed.bin','loc_pq_compressed.bin','emb_pq_pivots.bin','loc_pq_pivots.bin')}
    c.save(stamp,dict(identity=sig,files=files,code_source=str(PQ),ordering='subset source_ids; no PQ retraining'))

def ground_truth():
    out=c.RUN/'data';chunks=out/'gt-chunks';chunks.mkdir(exist_ok=True)
    weights=[float(np.float32(a)) for a in (.3,.7)]
    x,y,qx,qy=(p.matrix(c.D/name) for name in p.NAMES)
    sig=dict(subset_sha256=c.sha(c.D/'subset.json'),weights=weights,k=10,query_batch=8,block_rows=32768,objective=p.OBJECTIVE,script_sha256=c.sha(__file__),reference_sha256=c.sha(p.__file__))
    manifest=out/'extra-gt.json'
    if manifest.exists():
        old=c.load(manifest)
        if old['signature']!=sig:raise RuntimeError('extra GT signature changed')
        for name,h in old['files'].items():
            if c.sha(out/name)!=h:raise RuntimeError('GT identity changed')
        return
    arrays=[[],[]]
    for first in range(0,len(qx),8):
        path=chunks/('q-%05d.npz'%first)
        if not path.exists():
            ids,scores=p.exact_batch(x,y,np.asarray(qx[first:first+8],dtype=np.float64),np.asarray(qy[first:first+8],dtype=np.float64),weights,10,32768)
            # Direct differences are independent of dot-product GT generation.
            for k,a in enumerate(weights):
                dx=np.linalg.norm(np.asarray(x[ids[k]],dtype=np.float64)-np.asarray(qx[first:first+8],dtype=np.float64)[:,None,:],axis=2)
                dy=np.linalg.norm(np.asarray(y[ids[k]],dtype=np.float64)-np.asarray(qy[first:first+8],dtype=np.float64)[:,None,:],axis=2)
                if not np.allclose(a*dx+(1-a)*dy,scores[k],rtol=1e-10,atol=1e-10):raise RuntimeError('GT direct-distance validation')
            with path.with_suffix('.partial').open('wb') as f:np.savez(f,ids=ids,signature=np.array(json.dumps(sig,sort_keys=True)))
            path.with_suffix('.partial').replace(path)
        with np.load(str(path),allow_pickle=False) as z:
            if str(z['signature'])!=json.dumps(sig,sort_keys=True):raise RuntimeError('GT checkpoint changed')
            for k in range(2):arrays[k].append(z['ids'][k])
        c.save(out/'gt-progress.json',dict(queries_done=min(first+8,len(qx)),queries=len(qx)))
    for k,a in enumerate((.3,.7)):
        p.write_matrix(out/('gt-alpha-'+str(a)+'.bin'),np.concatenate(arrays[k]),'<u4')
        p.write_matrix(out/('query-alpha-'+str(a)+'.bin'),np.full((len(qx),1),np.float32(a)),'<f4')
    files={f.name:c.sha(f) for a in (.3,.7) for f in (out/('gt-alpha-'+str(a)+'.bin'),out/('query-alpha-'+str(a)+'.bin'))}
    c.save(manifest,dict(status='PASS',signature=sig,files=files))

def gtpath(a):return (c.RUN/'data' if a in (.3,.7) else c.D)/('gt-alpha-'+str(a)+'.bin')
def alphapath(a):return (c.RUN/'data' if a in (.3,.7) else c.D)/('query-alpha-'+str(a)+'.bin')

def oracle():
    out=c.RUN/'data';path=out/'oracle.json'
    if path.exists():return c.load(path)
    identity=c.load(c.D/'identity.json');selected=sorted(identity['query_cohorts']['tune'][:8]+identity['query_cohorts']['report'][:8])
    x,y,qx,qy=(p.matrix(c.D/name) for name in p.NAMES)
    exact,_=p.exact_batch(x,y,np.asarray(qx[selected],dtype=np.float64),np.asarray(qy[selected],dtype=np.float64),[1.,0.],320,32768)
    result=dict(query_ids=selected,k=320,precision='float64 exhaustive; ties by local ID',weights={})
    for a in WEIGHTS:
        gt=p.matrix(gtpath(a),'<u4');result['weights'][str(a)]=dict(recall=float(np.mean([len(set(map(int,gt[qi]))&(set(map(int,exact[0,j]))|set(map(int,exact[1,j]))))/10 for j,qi in enumerate(selected)])))
    (out/'gate-ids.txt').write_text('\n'.join(map(str,selected))+'\n')
    c.save(path,result);return result

def contribution():
    x,y,qx,qy=(p.matrix(c.D/name) for name in p.NAMES)
    cohorts=c.load(c.D/'identity.json')['query_cohorts'];rows={}
    for a in WEIGHTS:
        gt=p.matrix(gtpath(a),'<u4');af=float(np.float32(a));vals=[]
        for i in range(len(qx)):
            dx=np.linalg.norm(np.asarray(x[gt[i]],dtype=np.float64)-qx[i].astype(np.float64),axis=1);dy=np.linalg.norm(np.asarray(y[gt[i]],dtype=np.float64)-qy[i].astype(np.float64),axis=1);den=af*dx+(1-af)*dy;good=den>0;vals.append(float(np.mean(af*dx[good]/den[good])) if good.any() else None)
        rows[str(a)]={k:dict(mean=float(np.mean([vals[i] for i in ids if vals[i] is not None])),valid_queries=sum(vals[i] is not None for i in ids),undefined_query_ids=[i for i in ids if vals[i] is None]) for k,ids in cohorts.items()}
    c.save(c.RUN/'data/contribution.json',rows)

if __name__=='__main__':
    if sys.argv[1]=='shared':shared()
    elif sys.argv[1]=='gt':ground_truth();oracle();contribution()
    else:raise RuntimeError('shared|gt')
