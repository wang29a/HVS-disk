"""Node3 G2: immutable full/initial B, independent mutable update fixtures."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import struct
import subprocess
import time
import numpy as np
import run_g0 as c
from run_g1 import matrix
from quality_metrics import matched_work

E=c.HOST_ROOT/'experiments'
AB=E/'backlink-ab-20261115/runtime'
MG=E/'multigraph-20261113/runtime'
DATA=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
OUT=Path('/mnt/nvme2/wz/hvs-update-basic-2026-11-16/g2')
DEPTHS=[16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048]


def status(stage,**kw):c.save('status.json',dict(stage=stage,updated_unix=time.time(),**kw))


def run(args,label,timeout=7200,extra=None):
    env=dict(c.ENV,OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4',OPENBLAS_NUM_THREADS='1')
    if extra:env.update(extra)
    cmd=['nice','-n','15','ionice','-c','2','-n','7','numactl','--physcpubind=72-75','--membind=3']+list(map(str,args))
    status(label,command=cmd);started=time.time()
    with (c.RUN/(label+'.log')).open('w') as log:
        child=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
        status(label,pid=child.pid,command=cmd)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    c.save(label+'.json',dict(command=cmd,returncode=rc,wall_s=time.time()-started,
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH')},extra=extra))
    if rc:raise RuntimeError(label+' rc='+str(rc))


def write_matrix(p,a):
    a=np.ascontiguousarray(a)
    with Path(p).open('wb') as f:f.write(struct.pack('<II',*a.shape));a.tofile(f)


def prepare():
    OUT.mkdir(parents=True,exist_ok=False)
    rng=np.random.RandomState(20261116);ids=rng.choice(2000000,100000,replace=False).astype('<u4')
    ids.tofile(str(OUT/'source-ids.u32'))
    arrays=[]
    for name in ('img','text'):
        a=np.asarray(matrix(DATA/('base_'+name+'_emb.fvecs'))[ids],dtype='<f4');arrays.append(a)
        write_matrix(OUT/('base-'+name+'.bin'),a);write_matrix(OUT/('initial-'+name+'.bin'),a[:90000])
    with (OUT/'xy.pages').open('wb') as f:
        for lo in range(0,100000,2048):
            block=np.concatenate([a[lo:lo+2048] for a in arrays],axis=1)
            block.astype('<f4').tofile(f)
    # 512+512 floats = 4096 bytes: exactly two rows/page, count is even.
    assert (OUT/'xy.pages').stat().st_size==409600000
    center=[]
    for a in arrays:
        value=np.zeros(512,dtype='<f4')
        for row in a[:90000]:np.add(value,row,out=value)
        value/=np.float32(90000);center.append(value)
    write_matrix(OUT/'centers.bin',np.concatenate(center).reshape(1,-1))
    for a in (.1,.5,.9):
        # Full float64 exhaustive distances, raw vectors, rooted fusion.
        qx=matrix(DATA/'query_img_emb.fvecs');qy=matrix(DATA/'query_text_emb.fvecs')
        alpha=float(np.float32(a));truth=[]
        xb,yb=[v.astype(np.float64) for v in arrays]
        bx=(xb*xb).sum(1);by=(yb*yb).sum(1)
        for first in range(0,len(qx),16):
            x=np.asarray(qx[first:first+16],dtype=np.float64);y=np.asarray(qy[first:first+16],dtype=np.float64)
            dx=np.sqrt(np.maximum(0,(x*x).sum(1)[:,None]+bx[None,:]-2*x.dot(xb.T)))
            dy=np.sqrt(np.maximum(0,(y*y).sum(1)[:,None]+by[None,:]-2*y.dot(yb.T)))
            scores=alpha*dx+(1-alpha)*dy
            for row in scores:
                threshold=np.partition(row,9)[9];candidates=np.flatnonzero(row<=threshold)
                ranked=candidates[np.lexsort((candidates,row[candidates]))][:10];truth.append(ranked)
        write_matrix(OUT/('gt-'+str(a)+'.bin'),np.asarray(truth,dtype='<u4'))
        write_matrix(OUT/('alpha-'+str(a)+'.bin'),np.full((len(qx),1),a,dtype='<f4'))
    c.save('protocol.json',dict(seed=20261116,n=100000,initial=90000,insertions=10000,
        subset='RandomState choice without replacement from 2M; preserve draw order',
        GT='float64 exhaustive norm-dot L2, sqrt then float32-alpha promoted to float64 fusion; score then ID ties',
        threads=4,NUMA=3,repeats=3,depths=DEPTHS,report='500 held-out query IDs',
        gate='all three alpha matched 95% ID Recall@10 expansion increase <=15%; no extrapolation',
        source_ids_sha256=c.sha(OUT/'source-ids.u32'),source_inputs={str(DATA/('base_'+s+'_emb.fvecs')):c.sha(DATA/('base_'+s+'_emb.fvecs')) for s in ('img','text')}))


def fixture(mode):
    root=OUT/mode;root.mkdir()
    for source,name in [(OUT/'initial/packed/compact.meta','index.meta'),(OUT/'initial/packed/compact.topo','graph.pages'),(OUT/'centers.bin','centers.bin')]:shutil.copyfile(source,root/name)
    with (OUT/'xy.pages').open('rb') as f,(root/'coordinates.pages').open('wb') as g:
        remaining=90000*4096
        while remaining:
            b=f.read(min(2**20,remaining));assert b;g.write(b);remaining-=len(b)
    ids=np.fromfile(str(OUT/'source-ids.u32'),dtype='<u4')
    for space in ('emb','loc'):
        shutil.copyfile(MG/'data'/(space+'_pq_pivots.bin'),root/(space+'_pq_pivots.bin'))
        write_matrix(root/(space+'_pq_compressed.bin'),np.asarray(matrix(MG/'data'/(space+'_pq_compressed.bin'),'u1')[ids[:90000]],dtype='u1'))
    for s,key in [('img','x'),('text','y')]:write_matrix(root/('insert-'+key+'.bin'),np.asarray(matrix(OUT/('base-'+s+'.bin'))[90000:],dtype='<f4'))
    (root/'fixture-identity.json').write_text(json.dumps({p.name:c.sha(p) for p in root.iterdir()},indent=2)+'\n')
    return root


def evaluate(name,meta,topo):
    manifest=OUT/(name+'-manifest.txt')
    manifest.write_text('HVS_MULTI_V1 100000 512 512 1\n'+str(OUT/'xy.pages')+'\n'+str(MG/'data')+'/\nu '+str(meta)+' '+str(topo)+' - 0\n')
    result=[]
    for a in (.1,.5,.9):
        jobs=OUT/('jobs-'+str(a)+'.txt');jobs.write_text(''.join('L%d DEG_off %d 10 0 %s\n'%(L,L,a) for L in DEPTHS))
        for cohort,repeats in [('tune',1),('report',3)]:
            for rep in range(repeats):
                label='%s-%s-%s-%d'%(name,a,cohort,rep);output=c.RUN/(label+'-results.json')
                run([MG/'bin/multigraph',manifest,DATA/'query_img_emb.fvecs',DATA/'query_text_emb.fvecs',OUT/('alpha-'+str(a)+'.bin'),OUT/('gt-'+str(a)+'.bin'),MG/'data'/(cohort+'-ids.txt'),output,'dram',4,0,'exact',jobs,'-',1],label,timeout=3600)
                points=json.loads(output.read_text());m=matched_work([dict(L=p['L'],recall_id=p['recall'],expanded=sum(g['expanded'] for g in p['graphs'])) for p in points])
                result.append(dict(alpha=a,cohort=cohort,repeat=rep,matched=m))
                if not m['covered']:raise RuntimeError('95% not bracketed '+label)
    return result


def main():
    c.RUN=c.ROOT/'runtime/g2';c.RUN.mkdir(parents=True,exist_ok=True)
    lock=(E/'budget-reorder-20261103/queue.lock').open('a+');status('waiting_global_lock');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        assert json.loads((c.ROOT/'runtime/g1-revision3/status.json').read_text())['stage']=='G1_PASSED_BOTH_DISTANCES'
        assert shutil.disk_usage('/mnt/nvme2').free>5*2**30,'need 5 GiB margin'
        if OUT.exists():raise RuntimeError('existing mutable G2: refuse overwrite')
        (c.RUN/'artifacts').symlink_to(OUT,target_is_directory=True)
        status('preparing_data_and_GT');prepare()
        proof=json.loads((AB/'validation/build.json').read_text());builder=AB/'bin/build-B'
        assert c.sha(builder)==proof['binaries']['B']['binary']
        c.save('builder-identity.json',dict(proof=proof,binary=c.sha(builder),executor=c.sha(MG/'bin/multigraph'),executor_source=c.sha(E/'multigraph-20261113/scripts/multigraph.cpp')))
        for name,n,prefix in [('full',100000,'base'),('initial',90000,'initial')]:
            root=OUT/name;root.mkdir()
            run([builder,'dual',OUT/(prefix+'-img.bin'),OUT/(prefix+'-text.bin'),n,512,512,4,root],'build-'+name,extra=dict(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1'))
            counts=json.loads((root/'reverse-insert-stats.json').read_text());assert counts['reverse_targets']==counts['B_reprunes'] and counts['reverse_always_prune']==1
            packed=root/'packed';packed.mkdir()
            run([MG/'bin/pack',root/'full.meta',root/'float-graph.bin',packed/'compact.meta',packed/'compact.topo',packed/'ranges.bin',root/'full.meta'],'pack-'+name)
            run([AB/'function-v2/bin/graph-audit',root/'float-graph.bin',packed/'compact.meta',packed/'compact.topo',packed/'ranges.bin'],'audit-'+name)
        src=c.ROOT/'implementation';files=[src/v for v in ('update_store.cpp','deg_prune.cpp','dual_update.cpp','g2_insert.cpp')]
        run(['clang-format','-style=LLVM','-i',src/'g2_insert.cpp'],'format-g2')
        flags=json.loads((c.ROOT/'runtime/g1-revision3/build-identity.json').read_text())['flags']
        run(['g++-11']+flags+files+[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',c.RUN/'g2_insert'],'compile-g2',600)
        c.save('build-identity.json',dict(flags=flags,sources={str(p):c.sha(p) for p in files},binary=c.sha(c.RUN/'g2_insert')))
        full=evaluate('full',OUT/'full/packed/compact.meta',OUT/'full/packed/compact.topo');c.save('full-matched.json',full)
        comparisons=[]
        for mode in ('pq','exact'):
            root=fixture(mode);run([c.RUN/'g2_insert',root,mode],'insert-'+mode,timeout=14400)
            rows=evaluate(mode,root/'eval.meta',root/'eval.topo');c.save(mode+'-matched.json',rows)
            for row,base in zip(rows,full):
                increase=row['matched']['expanded']/base['matched']['expanded']-1
                comparisons.append(dict(mode=mode,alpha=row['alpha'],cohort=row['cohort'],repeat=row['repeat'],increase=increase,passed=increase<=.15))
            c.save('comparisons.json',comparisons)
            if any(not r['passed'] for r in comparisons if r['cohort']=='report'):
                raise RuntimeError('G2 expansion increase exceeds 15%; see comparisons.json')
        status('G2_PASSED_BOTH_DISTANCES',G4='PASSED_G1_AND_G2_BOTH',G3='NOT_RUN')
    except Exception as e:
        status('STOPPED_AT_G2',reason=str(e),dependent_experiments='not started');raise


if __name__=='__main__':main()
