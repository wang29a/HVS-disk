"""Build the frozen E1 1.8M B baseline on node3 before insertion experiments."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import struct
import subprocess
import time
import run_g0 as c

E=c.HOST_ROOT/'experiments'
AB=E/'backlink-ab-20261115/runtime'
MG=E/'multigraph-20261113/runtime'
DATA=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
COLD=Path(os.environ.get('HVS_E1_RUN_ROOT','/mnt/pmem0/hvs-update-basic-2026-11-16/e1'))


def save(path,data):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n');temporary.replace(path)


def run(args,label,timeout=14400,threads=4):
    env={k:v for k,v in c.ENV.items() if not k.startswith(('HVS_','LAYOUT_','PTR_','METRIC_'))}
    env.update(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1')
    cpus='72-87' if threads==16 else '72-79' if threads==8 else '72-75'
    command=['nice','-n','15','ionice','-c','2','-n','7','numactl','--physcpubind='+cpus,'--membind=3']+list(map(str,args))
    started=time.time()
    with (COLD/(label+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        save(COLD/'prepare-status.json',dict(stage=label,pid=child.pid,command=command,started_unix=started))
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    record=dict(command=command,environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','HVS_SOURCE_ONLY','HVS_FULL_ONLY','LD_LIBRARY_PATH')},
                returncode=rc,wall_s=time.time()-started)
    save(COLD/(label+'.json'),record)
    if rc:raise RuntimeError(label+' failed '+str(rc))
    return record


def prefix(source,destination,n):
    with source.open('rb') as f,destination.open('xb') as g:
        count,dim=struct.unpack('<II',f.read(8));assert count==2000000 and dim==512
        g.write(struct.pack('<II',n,dim));remaining=n*dim*4
        while remaining:
            data=f.read(min(8*2**20,remaining));assert data;g.write(data);remaining-=len(data)
    return dict(source=str(source),source_sha256=c.sha(source),destination=str(destination),n=n,dim=dim,sha256=c.sha(destination))


def main():
    build_threads=int(os.environ.get('HVS_E1_BUILD_THREADS','16'));assert build_threads in (4,8,16)
    COLD.mkdir(parents=True,exist_ok=True)
    lock=(E/'budget-reorder-20261103/queue.lock').open('a+')
    save(COLD/'prepare-status.json',dict(stage='waiting_global_lock'));fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        assert json.loads((c.ROOT/'runtime/g3-entry-parity/status.json').read_text())['stage']=='G3_PASSED_ASAN_AND_TSAN'
        assert json.loads((c.ROOT/'runtime/g2-entry-parity/status.json').read_text())['stage']=='G2_PASSED_BOTH_DISTANCES'
        proof=json.loads((AB/'validation/build.json').read_text());builder=AB/'bin/build-B'
        assert c.sha(builder)==proof['binaries']['B']['binary']
        assert c.sha(AB/'build/B/libpipeann.a')==proof['binaries']['B']['library']
        assert c.sha(AB/'build/hybrid_index.cpp')==proof['variant']
        if (COLD/'initial-1800k').exists():raise RuntimeError('existing E1 build: refuse overwrite')
        assert shutil.disk_usage(str(COLD)).free>70*2**30,'cold artifact capacity requires 70 GiB'
        link=c.ROOT/'runtime/e1'
        if not link.exists():link.symlink_to(COLD,target_is_directory=True)
        else:assert link.resolve()==COLD
        save(COLD/'authorization.json',dict(user_confirmation='current B, default PQ; no connectivity repair or deduplication',
            checkpoint_records=['unreachable live count','average degree separately for pq/exact','corresponding GT unreachable-ID ceiling loss beside ID recall','PQ expansion-increase trend'],
            initial=1800000,final=2000000,batch=50000,weights=[.1,.3,.5,.7,.9],R=40,L=200,C=750,update_threads=4,
            L_ins=200,C_ins=100,reprune_dist='pq',reverse_policy='append below R, prune only at R',normalization='none',
            build_threads=build_threads,NUMA=3,GT='float64 exhaustive rooted L2 fusion on current committed live points',
            cold_storage=str(COLD),measurement_storage='topology and coordinates both nvme3; use the dedicated HVS capacity reservation when needed',
            storage_reason='NVMe3 has a verified data-free HVS capacity reservation; do not choose cross-device SSD measurements using df output alone',
            inherited_build_identity=proof,implementation={p.name:c.sha(p) for p in (c.ROOT/'implementation').glob('*') if p.suffix in ('.h','.cpp')},
            README_sha256=c.sha(c.ROOT/'README.md')))
        inputs=COLD/'inputs';inputs.mkdir(exist_ok=True)
        save(COLD/'prepare-status.json',dict(stage='materializing_1800k_prefix_inputs'))
        if (COLD/'input-identity.json').exists():
            identity=json.loads((COLD/'input-identity.json').read_text())
            for row in identity:
                assert c.sha(row['source'])==row['source_sha256'] and c.sha(row['destination'])==row['sha256'],'immutable prefix changed'
        else:identity=[prefix(DATA/('base_'+s+'_emb.fvecs'),inputs/('base-'+s+'-1800k.bin'),1800000) for s in ('img','text')]
        frozen=json.loads((c.ROOT/'runtime/gt-ties-v1/results.json').read_text())['inputs']
        for row in identity:assert row['source_sha256']==frozen[row['source']],'base vectors changed since verified gates'
        save(COLD/'input-identity.json',identity)
        target=COLD/'initial-1800k';target.mkdir()
        run([builder,'dual',inputs/'base-img-1800k.bin',inputs/'base-text-1800k.bin',1800000,512,512,build_threads,target],'build-initial-1800k-B',threads=build_threads)
        counts=json.loads((target/'reverse-insert-stats.json').read_text())
        assert counts['reverse_always_prune']==1 and counts['reverse_targets']==counts['B_reprunes']
        packed=target/'packed';packed.mkdir()
        run([MG/'bin/pack',target/'full.meta',target/'float-graph.bin',packed/'compact.meta',packed/'compact.topo',packed/'ranges.bin',target/'full.meta'],'pack-initial-1800k-B')
        run([AB/'function-v2/bin/graph-audit',target/'float-graph.bin',packed/'compact.meta',packed/'compact.topo',packed/'ranges.bin'],'validate-initial-1800k-B')
        save(COLD/'initial-build-identity.json',dict(status='B_INITIAL_GRAPH_VALIDATED',inputs=identity,binary=c.sha(builder),reverse_counts=counts,
            outputs={str(p.relative_to(target)):c.sha(p) for p in target.rglob('*') if p.is_file()}))
        save(COLD/'prepare-status.json',dict(stage='INITIAL_B_READY',updated_unix=time.time()))
    except Exception as e:
        save(COLD/'prepare-status.json',dict(stage='STOPPED_PREPARING_E1',reason=str(e),updated_unix=time.time()));raise


if __name__=='__main__':main()
