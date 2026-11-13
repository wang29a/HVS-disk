"""Prepare the OpenImages X-only Vamana index and its exhaustive L2 GT."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
BUILD=ROOT.parent.parent
EXP=ROOT.parent
sys.path.insert(0,str(EXP/'layout-cache-budget-20261109'/'scripts'))
import run as base
base.ROOT=ROOT
ENV=dict(base.ENV,OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         OMP_PROC_BIND='close',OMP_PLACES='cores')
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2')
import numpy as np

PAGE=8192
N=456699
DIM=768
QUERIES=1000
K=10


def disk_gate(task):
    def read():
        r=next(s.split() for s in Path('/proc/diskstats').read_text().splitlines()
               if s.split()[2]=='nvme3n1')
        return int(r[9]),int(r[12])
    while True:
        w,busy=read()
        start=time.monotonic()
        time.sleep(4)
        w2,busy2=read()
        sec=time.monotonic()-start
        activity=(busy2-busy)/(sec*10)
        write=(w2-w)*512/(sec*(1<<20))
        if activity<=10 and write<=1:
            return
        base.status('waiting_for_quiet_disk',task=task,disk_busy_pct=activity,write_mib_s=write)
        time.sleep(15)


def vec(path,expected):
    with path.open('rb') as f:
        shape=struct.unpack('<II',f.read(8))
    if shape!=expected or path.stat().st_size!=8+4*shape[0]*shape[1]:
        raise RuntimeError('float bin shape mismatch: '+str(path))
    return np.memmap(str(path),mode='r',dtype='<f4',offset=8,shape=shape)


def input_data(c):
    x=Path(c['query1']).with_name('base_img_emb.fvecs')
    q=Path(c['query1'])
    x_data=vec(x,(N,DIM))
    queries=vec(q,(QUERIES,DIM))
    coord=Path(c['coord'])
    if coord.stat().st_size!=N*PAGE:
        raise RuntimeError('unexpected dual coordinate packing')
    pages=np.memmap(str(coord),mode='r',dtype='<f4',shape=(N,PAGE//4))
    for i in range(0,N,8192):
        if not np.array_equal(x_data[i:i+8192],pages[i:i+8192,:DIM]):
            raise RuntimeError('dual X vector / base ID order differs')
    if not np.isfinite(queries).all() or base.digest(q)!=c['input_sha256']['query1']:
        raise RuntimeError('query identity mismatch')
    return x,q,x_data,queries


def validate_gt(path,x_data,queries):
    raw=path.read_bytes()
    if len(raw)!=8+QUERIES*K*12 or struct.unpack_from('<II',raw)!=(QUERIES,K):
        raise RuntimeError('unexpected GT header')
    ids=np.frombuffer(raw,dtype='<u4',offset=8,count=QUERIES*K).reshape(QUERIES,K)
    distances=np.frombuffer(raw,dtype='<f4',offset=8+QUERIES*K*4,count=QUERIES*K).reshape(QUERIES,K)
    tags=np.frombuffer(raw,dtype='<u4',offset=8+QUERIES*K*8).reshape(QUERIES,K)
    if (ids.max()>=N or not np.array_equal(ids,tags) or not np.isfinite(distances).all()
            or np.any(np.diff(distances,axis=1)<0) or np.any(np.diff(np.sort(ids,axis=1),axis=1)==0)):
        raise RuntimeError('GT IDs, tags or distance ordering invalid')
    delta=x_data[ids].astype(np.float64)-queries.astype(np.float64)[:,None,:]
    recomputed=np.einsum('qkd,qkd->qk',delta,delta)
    if not np.allclose(distances,recomputed,rtol=5e-5,atol=5e-5):
        raise RuntimeError('GT distance verification failed')
    sample_ids=[0,71,163,257,409,583,743,997]
    qs=queries[sample_ids].astype(np.float64)
    best=[[] for _ in sample_ids]
    for i in range(0,N,8192):
        block=x_data[i:i+8192].astype(np.float64)
        v=np.einsum('nd,nd->n',block,block)[:,None] + np.einsum('qd,qd->q',qs,qs)[None,:] - 2*block.dot(qs.T)
        nearest=np.argpartition(v,K-1,axis=0)[:K,:]
        for col in range(len(sample_ids)):
            best[col]=sorted(best[col]+[(float(v[k,col]),int(i+k)) for k in nearest[:,col]])[:K]
    for qi, top in zip(sample_ids,best):
        if set(ids[qi])!={idx for _,idx in top}:
            raise RuntimeError('float64 exhaustive top-10 cross-check failed for query '+str(qi))
    return dict(queries=QUERIES,k=K,metric='squared L2',returned_distances_checked=QUERIES*K,
                independent_float64_queries=sample_ids,max_abs_distance_error=float(np.max(np.abs(distances-recomputed))))


def validate_disk(path,x_data):
    with path.open('rb') as f:
        data=f.read(PAGE)
    count,columns=struct.unpack_from('<II',data)
    if count<8 or columns!=1:
        raise RuntimeError('unexpected single-space disk metadata')
    n,d,medoid,width,npp,frozen,_,size=struct.unpack_from('<8Q',data,8)
    max_degree=(width-DIM*4)//4-1
    pages=(N+npp-1)//npp
    if (n!=N or d!=DIM or medoid>=N or frozen or npp!=PAGE//width or not 0<max_degree<=64
            or size!=path.stat().st_size or size!=(1+pages)*PAGE):
        raise RuntimeError('single-space disk metadata mismatch')
    raw=np.memmap(str(path),mode='r',dtype=np.uint8)
    records=np.ndarray((pages,npp,width//4),dtype='<u4',buffer=raw,offset=PAGE,
                       strides=(PAGE,width,4))
    adjacency=np.full((N,max_degree),N,dtype=np.uint32)
    degree_min,degree_max,total=1000,0,0
    for j in range(0,pages,2048):
        lo=j*npp
        hi=min(N,(j+2048)*npp)
        rows=records[j:j+2048].reshape(-1,width//4)[:hi-lo]
        if not np.array_equal(rows[:,:DIM],x_data[lo:hi].view('<u4')):
            raise RuntimeError('serialized coordinate mismatch')
        degree=rows[:,DIM]
        degree_min=min(degree_min,int(degree.min()))
        degree_max=max(degree_max,int(degree.max()))
        if degree_min<1 or degree_max>max_degree:
            raise RuntimeError('invalid node degree')
        ids=rows[:,DIM+1:]
        active=np.arange(max_degree)<degree[:,None]
        if np.any(ids[active]>=N) or np.any(active & (ids==np.arange(lo,hi)[:,None])):
            raise RuntimeError('out-of-range neighbor or self loop')
        if np.any(ids[~active]):
            raise RuntimeError('nonzero neighbor padding')
        adjacency[lo:hi][active]=ids[active]
        sorted_ids=np.sort(adjacency[lo:hi],axis=1)
        if np.any((sorted_ids[:,1:]!=N)&(np.diff(sorted_ids,axis=1)==0)):
            raise RuntimeError('duplicate neighbor')
        total+=int(degree.sum())
    visited=np.zeros(N+1,dtype=bool)
    visited[N]=True
    frontier=np.array([medoid],dtype=np.uint32)
    visited[frontier]=True
    while len(frontier):
        nodes=adjacency[frontier].ravel()
        frontier=np.unique(nodes[~visited[nodes]])
        visited[frontier]=True
    reached=int(visited[:N].sum())
    if reached!=N:
        raise RuntimeError('entry can only reach {} of {} objects'.format(reached,N))
    return dict(nodes=N,dimension=DIM,medoid=medoid,record_bytes=width,nodes_per_page=npp,
                observed_degree_min=degree_min,observed_degree_max=degree_max,
                edges=total,entry_reachable_nodes=reached,alpha_ranges=False)


def main():
    for d in ('analysis','bin','index'):
        (ROOT/d).mkdir(exist_ok=True)
    if os.sched_getaffinity(0)!={48,49}:
        raise RuntimeError('CPU affinity differs from planned NUMA 2 physical cores')
    with (ROOT/'queue.lock').open('a') as own, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        base.status('preflight')
        if shutil.disk_usage(str(ROOT)).free<32*(1<<30):
            raise RuntimeError('insufficient nvme3 free space')
        with (ROOT/'logs/iostat.log').open('w') as stats:
            device=subprocess.Popen(['iostat','-x','-t','-y','5'],env=ENV,stdout=stats,stderr=subprocess.STDOUT)
            try:
                prepare()
            finally:
                if device.poll() is None:device.terminate()
                device.wait()
    base.status('complete',dataset='OpenImages',nodes=N,queries=QUERIES,gt_k=K,serial_lock_released=True)


def prepare():
    spec=base.read(EXP/'concurrency-alpha/analysis/build.json')['separated']
    if base.digest(spec['library'])!=spec['library_sha256']:
        raise RuntimeError('existing HVS library changed')
    src=ROOT/'scripts/build_single.cpp'
    builder=ROOT/'bin/build-single'
    base.run(['clang-format','-i','-style=LLVM',src],'format',ENV)
    base.run(['g++-11']+spec['flags']+[src,spec['library'],'-L/mnt/nvme2/wz/disk/local/openblas/lib',
             '-laio','-ljemalloc','-lopenblas','-o',builder],'compile',ENV)
    c=next(x for x in base.read(EXP/'layout-cache-budget-20261109'/'analysis/inputs.json') if x['name']=='openimage')
    disk_gate('verify-inputs')
    base.status('validating_vectors')
    x,q,x_data,queries=input_data(c)
    prefix=ROOT/'index/openimage_x'
    pq_inputs={}
    for source,suffix in (('emb_pq_compressed.bin','_pq_compressed.bin'),('emb_pq_pivots.bin','_pq_pivots.bin')):
        p=Path(c['prefix'])/source
        copy=Path(str(prefix)+suffix)
        if not copy.exists():
            shutil.copyfile(str(p),str(copy))
        if base.digest(p)!=base.digest(copy):
            raise RuntimeError('PQ copy changed')
        pq_inputs[suffix]=dict(source=str(p),sha256=base.digest(copy))
    codes=Path(str(prefix)+'_pq_compressed.bin')
    with codes.open('rb') as f:
        if struct.unpack('<II',f.read(8))!=(N,64) or codes.stat().st_size!=8+N*64:
            raise RuntimeError('one-space PQ code header')
    gt_tool=BUILD/'build-separated/tests/utils/compute_groundtruth'
    gt=ROOT/'index/openimage_x_l2_top10.bin'
    base.save(ROOT/'analysis/inputs.json',dict(dataset='OpenImages X image embeddings',space_count=1,
              base=str(x),base_sha256=base.digest(x),query=str(q),query_sha256=base.digest(q),
              same_ids_and_first_vectors_as_dual=True,pq=pq_inputs,n=N,dim=DIM,query_count=QUERIES))
    base.save(ROOT/'analysis/protocol.json',dict(cpus=[48,49],numa=2,build_threads=2,gt_blas_threads=2,
              degree=40,build_l=100,metric='L2',legacy_inline_reorder=False,pq_bytes=64,
              library_sha256=spec['library_sha256'],source_sha256=base.digest(src),
              builder_sha256=base.digest(builder),gt_tool_sha256=base.digest(gt_tool)))
    if not (gt.exists() and (ROOT/'analysis/gt-validation.json').exists()):
        disk_gate('ground-truth')
        base.status('computing_exact_gt')
        base.run([gt_tool,'float',x,q,K,gt],'groundtruth',dict(ENV,OPENBLAS_NUM_THREADS='2'))
    base.save(ROOT/'analysis/gt-validation.json',dict(status='PASS',**validate_gt(gt,x_data,queries)))
    disk_gate('single-space-vamana')
    base.status('building_vamana_graph')
    base.run([builder,x,prefix],'build-vamana',ENV)
    base.status('validating_single_space_index')
    index=Path(str(prefix)+'_disk.index')
    base.save(ROOT/'analysis/index-validation.json',dict(status='PASS',**validate_disk(index,x_data)))
    base.save(ROOT/'analysis/artifacts.json',dict(files={p.name:dict(bytes=p.stat().st_size,sha256=base.digest(p))
              for p in sorted((ROOT/'index').iterdir()) if p.is_file()}))


if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as e:
        base.status('interrupted' if isinstance(e,InterruptedError) else 'failed',error=str(e))
        traceback.print_exc()
        sys.exit(1)
