"""Incoming-degree revision: immutable replay, then a separate fresh E1 run."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import struct
import time
import numpy as np
import run_g0 as c
import run_e1 as e
import run_g1

OLD=Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1')
OUT=c.ROOT/'runtime/degree-validation'
SOURCES=('update_store.cpp','update_store.h','dual_update.cpp','dual_update.h',
         'deg_prune.cpp','deg_prune.h','entry_frontier.h','in_degree_guard.h','e1_driver.cpp')


def setup(root):
    root.mkdir(parents=True,exist_ok=False)
    for name in ('initial-1800k','inputs','gt'):
        (root/name).symlink_to(OLD/name,target_is_directory=True)
    shutil.copyfile(OLD/'prepare-status.json',root/'prepare-status.json')
    e.save(root/'reuse-inputs.json',{p.name:c.sha(p) for p in (OLD/'inputs').iterdir() if p.is_file()})
    (root/'variants').mkdir()
    e.save(root/'reused-baseline.json',dict(source=str(OLD),identity=c.sha(OLD/'initial-build-identity.json'),
        reason='same frozen initial B graph, prefix vectors and exhaustive GT; no rebuilding or overwriting'))


def raw_identity():
    raw=Path('/mnt/nvme3/linsy/EPDG/DEG/dataset/LAION')
    source=e.matrix(e.DATA/'source_ids.bin','<u4')[:,0]
    positions=np.unique(np.concatenate((np.arange(16),np.linspace(0,1999999,112).astype(int))))
    rows=[]
    for space in ('img','text'):
        base=e.matrix(e.DATA/('base_'+space+'_emb.fvecs'),'<f4')
        path=raw/('base_'+space+'_emb.fvecs')
        with path.open('rb') as f:
            for row in positions:
                original=int(source[row]);f.seek(original*513*4)
                dimension=struct.unpack('<I',f.read(4))[0];assert dimension==512
                assert f.read(2048)==base[row].tobytes(),'raw source-ID mapping mismatch'
        query=e.matrix(e.DATA/('query_'+space+'_emb.fvecs'),'<f4')
        querypath=raw/('query_'+space+'_emb.fvecs')
        with querypath.open('rb') as f:
            for row in query:
                assert struct.unpack('<I',f.read(4))[0]==512
                assert f.read(2048)==row.tobytes(),'raw query identity mismatch'
            assert not f.read(1)
        rows.append(dict(space=space,base=str(path),base_bytes=path.stat().st_size,
            sampled_original_ids=[int(source[r]) for r in positions],subset_positions=positions.tolist(),
            query=str(querypath),query_sha256=c.sha(querypath),all_query_rows_equal=True))
    e.save(OUT/'raw-data-identity.json',dict(status='SAMPLED_BASE_AND_ALL_QUERIES_BYTE_EQUAL',
        source_ids_sha256=c.sha(e.DATA/'source_ids.bin'),sampled_base_rows_per_space=len(positions),rows=rows,
        limitation='base comparison is a deterministic row sample; the exact row count and original IDs are recorded; not a complete full-base hash comparison'))


def sanitized(gate):
    src=c.ROOT/'implementation'
    base=[v for v in gate['flags'] if v not in ('-O3','-O2','-static-libstdc++','-static-libgcc')]
    link=[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
    results=[]
    for name,flag in [('asan','address'),('tsan','thread')]:
        out=OUT/name;out.mkdir()
        flags=base+['-O1','-g','-fno-omit-frame-pointer','-fsanitize='+flag]
        units=[src/'update_store.cpp',src/'deg_prune.cpp',src/'dual_update.cpp',c.SOURCE/'src/utils/linux_aligned_file_reader.cpp']
        e.run(['g++-11']+flags+units+[src/'g3_concurrent.cpp']+link+['-o',out/'concurrent'],'compile-'+name,out=out)
        e.run(['g++-11']+flags+[src/'in_degree_guard_check.cpp','-pthread','-o',out/'guard'],'compile-guard-'+name,out=out)
        env=dict(ASAN_OPTIONS='halt_on_error=1:abort_on_error=1:detect_leaks=1',TSAN_OPTIONS='halt_on_error=1:exitcode=66')
        e.run([out/'guard'],'directed-two-writer-guard',out=out,extra=env)
        old=c.RUN;c.RUN=out
        try:fixture=run_g1.fixture('pq')
        finally:c.RUN=old
        for cohort in ('tune','report'):shutil.copyfile(e.SPLIT/(cohort+'-ids.txt'),fixture/(cohort+'-ids.txt'))
        e.run([out/'concurrent',fixture,e.AB/'data-10k/base_img_emb.fvecs',e.AB/'data-10k/base_text_emb.fvecs',
            e.DATA/'query_img_emb.fvecs',e.DATA/'query_text_emb.fvecs',600],name+'-concurrent-600s',out=out,extra=env)
        result=json.loads((fixture/'concurrent-result.json').read_text());assert result['passed']
        results.append(dict(sanitizer=name,duration_seconds=600,result=result,binary_sha256=c.sha(out/'concurrent')))
    e.save(OUT/'sanitizers.json',dict(results=results,
        scope='incoming-degree regression: two complete 600-second sanitizer workloads; original prior-version gates retained separately',
        coverage='new executor, prune, harness and upstream Linux reader instrumented; other prebuilt upstream/external libraries uninstrumented'))


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    lock=(e.E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    driver=None
    try:
        passed=c.ROOT/'runtime/g1-in-degree-protection-v2'
        assert json.loads((passed/'status.json').read_text())['stage']=='G1_PASSED_BOTH_DISTANCES'
        gate=json.loads((passed/'build-identity.json').read_text())
        for name,digest in gate['sources'].items():assert c.sha(c.ROOT/'implementation'/name)==digest
        setup(e.COLD);e.owned_hot(e.HOT_COORD);e.owned_hot(e.HOT_TOPO)
        e.status('compiling_incoming_degree_replay')
        src=c.ROOT/'implementation'
        e.run(['clang-format','-style=LLVM','-i',src/'e1_driver.cpp'],'format-driver')
        units=[src/n for n in ('update_store.cpp','deg_prune.cpp','dual_update.cpp','e1_driver.cpp')]
        command=['g++-11']+gate['flags']+units+[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',e.COLD/'e1_driver']
        e.run(command,'compile-driver',timeout=900)
        sources={n:c.sha(src/n) for n in SOURCES}
        e.save(OUT/'build-identity.json',dict(command=list(map(str,command)),sources=sources,binary=c.sha(e.COLD/'e1_driver')))
        raw_identity();e.driver_smoke();sanitized(gate)
        e.status('replaying_1800k_to_1850k_with_incoming_degree_guard')
        root=e.fixture('pq-counterexample-replay')
        driver=e.Driver(root,'pq',4,100,False)
        for cp in (0,1):
            directory=root/('checkpoint-%d'%cp);directory.mkdir()
            if cp:driver.command('insert 50000 96 96 '+str(directory),'INSERT_BATCH_COMPLETE')
            driver.command('checkpoint '+str(directory),'CHECKPOINT_READY')
            shutil.copyfile(e.HOT_TOPO/'graph.pages',directory/'graph.pages')
        driver.close();driver=None
        cp=root/'checkpoint-1'
        original=set(map(int,(c.ROOT/'runtime/e1-counterexample-1850k/new-unreachable-tags.txt').read_text().split()));assert len(original)==42
        unreachable=set(map(int,(cp/'unreachable-tags.txt').read_text().split()))
        assert not original.intersection(unreachable),'an original counterexample is still unreachable'
        structure=json.loads((cp/'structure.json').read_text());assert structure['new_zero_indegree_live']==0
        tags=np.fromfile(str(cp/'id-to-tag.u32'),dtype='<u4');inverse=np.empty(len(tags),dtype='<u4');inverse[tags]=np.arange(len(tags))
        pages=np.memmap(str(cp/'graph.pages'),dtype='u1',mode='r')
        indegree=np.zeros(len(tags),dtype='<u4')
        for first in range(0,len(tags),49):
            count=min(49,len(tags)-first)
            records=pages[(first//49)*8192:(first//49)*8192+count*164].view('<u4').reshape(count,41)
            for record in records:
                np.add.at(indegree,record[1:1+int(record[0])],1)
        assert np.all(indegree[inverse[list(original)]]>0)
        np.save(str(cp/'indegree.npy'),indegree)
        rows=[dict(tag=t,indegree=int(indegree[inverse[t]]),reachable=t not in unreachable) for t in sorted(original)]
        e.save(OUT/'original-42.json',rows)
        result=dict(status='G1_AND_1850K_REPLAY_PASSED',sources=sources,structure=structure,
            original_42_became_unreachable=0,original_42_zero_indegree=0,
            protection=json.loads((cp/'protection-batch.json').read_text()),
            replay=str(root),query_threads=4,update_threads=4,
            schedule='same initial graph, vectors and 50k workload; fresh concurrent execution, not identical historical thread ordering',
            BFS='observational; nonzero indegree does not guarantee global reachability',
            g1=str(passed),sanitizers=str(OUT/'sanitizers.json'))
        e.save(OUT/'result.json',result);e.status('DEGREE_REPLAY_PASSED',result=result)
    except Exception as error:
        if driver:driver.abort()
        e.save(OUT/'failure.json',dict(reason=str(error),updated_unix=time.time()))
        raise


if __name__=='__main__':main()
