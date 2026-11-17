"""Node3 E1 controller: immutable sources, one owned SSD working set."""
from pathlib import Path
import fcntl
import json
import os
import queue
import shutil
import struct
import subprocess
import threading
import time
import numpy as np
import run_g0 as c
from prepare_e1 import COLD,E,AB,MG,DATA
from gt_unreachable import matrix
from e1_gt import generate,WEIGHTS
from e1_evaluate import collect,snapshot_codes
from e1_metrics import trend
from e1_report import render

HOT_COORD=Path(os.environ.get('HVS_E1_HOT_COORD',str(c.ROOT/'runtime/e1-degree-protected-coordinates')))
HOT_TOPO=Path(os.environ.get('HVS_E1_HOT_TOPO',str(c.ROOT/'runtime/e1-degree-protected-topology')))
SPLIT=MG/'data'
VARIANTS=[('pq-t4','pq',4,100,False),('exact-t4','exact',4,100,False),
          ('pq-C50-t4','pq',4,50,False),('pq-always-t4','pq',4,100,True),
          ('pq-t1','pq',1,100,False),('pq-t8','pq',8,100,False)]


def save(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(p)


def status(stage,**kw):save(COLD/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))


def run(args,label,timeout=14400,out=None,extra=None,threads=4):
    out=Path(out or COLD);out.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in c.ENV.items() if not k.startswith(('HVS_','LAYOUT_','PTR_','METRIC_'))}
    env.update(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if extra:env.update(extra)
    cpus='72-79' if threads==8 else '72-75'
    command=['nice','-n','15','ionice','-c','2','-n','7','numactl','--physcpubind='+cpus,'--membind=3']+list(map(str,args))
    start=time.time();samples=[]
    with (out/(label+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        status(label,pid=child.pid,output=str(out),command=command)
        while child.poll() is None:
            if time.time()-start>timeout:
                child.terminate()
                try:child.wait(timeout=15)
                except subprocess.TimeoutExpired:child.kill();child.wait()
                raise RuntimeError('timeout '+label)
            if len(samples)<5 or int(time.time()-start)%30==0:
                try:
                    proc=Path('/proc')/str(child.pid)
                    samples.append(dict(unix=time.time(),io=proc.joinpath('io').read_text(),
                        placement=[v for v in proc.joinpath('status').read_text().splitlines() if v.startswith(('Cpus_allowed_list:','Threads:','VmRSS:'))]))
                except OSError:pass
            time.sleep(1)
    record=dict(command=command,returncode=child.returncode,wall_s=time.time()-start,samples=samples,
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH')})
    save(out/(label+'.json'),record)
    if child.returncode:raise RuntimeError(label+' rc='+str(child.returncode))


def copy_prefix(source,destination,size):
    assert not destination.exists(),str(destination)+' already exists'
    with source.open('rb') as f,destination.open('xb') as g:
        remaining=size
        while remaining:
            b=f.read(min(8*2**20,remaining));assert b;g.write(b);remaining-=len(b)
    assert destination.stat().st_size==size


def owned_hot(root):
    root.mkdir(parents=True,exist_ok=True)
    marker=root/'owner.json'
    if marker.exists():raise RuntimeError('existing E1 hot namespace: explicit identity resume required')
    assert not any(root.iterdir()),'unexpected files in new hot namespace'
    save(marker,dict(owner=str(COLD),purpose='E1 sequential NVMe working files; never upstream data',created_unix=time.time()))


def fixture(name):
    root=COLD/'variants'/name;root.mkdir(parents=True,exist_ok=False)
    source=COLD/'initial-1800k/packed'
    shutil.copyfile(source/'compact.meta',root/'index.meta')
    copy_prefix(source/'compact.topo',HOT_TOPO/'graph.pages',(source/'compact.topo').stat().st_size)
    copy_prefix(MG/'data/xy.pages',HOT_COORD/'coordinates.pages',1800000*4096)
    (root/'graph.pages').symlink_to(HOT_TOPO/'graph.pages');(root/'coordinates.pages').symlink_to(HOT_COORD/'coordinates.pages')
    shutil.copyfile(COLD/'inputs/centers.bin',root/'centers.bin')
    for space in ('emb','loc'):
        (root/(space+'_pq_pivots.bin')).symlink_to(MG/'data'/(space+'_pq_pivots.bin'))
        codes=np.asarray(matrix(MG/'data'/(space+'_pq_compressed.bin'),'u1')[:1800000],dtype='u1')
        with (root/(space+'_pq_compressed.bin')).open('wb') as f:f.write(struct.pack('<II',*codes.shape));codes.tofile(f)
    for src,dst in [('insert-x.bin','insert-x.bin'),('insert-y.bin','insert-y.bin')]:
        (root/dst).symlink_to(COLD/'inputs'/src)
    for s,k in [('img','x'),('text','y')]:
        (root/('query-'+k+'.bin')).symlink_to(DATA/('query_'+s+'_emb.fvecs'))
    (root/'report-ids.txt').symlink_to(SPLIT/'report-ids.txt')
    save(root/'fixture-identity.json',{str(p):c.sha(p) for p in root.iterdir() if p.is_file()})
    return root


class Driver:
    def __init__(self,root,mode,threads,C,always,initial=1800000,batch=50000,batches=4):
        env=dict(c.ENV,OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),OPENBLAS_NUM_THREADS='1')
        cpus='72-79' if threads==8 else '72-75'
        command=['nice','-n','15','ionice','-c','2','-n','7','numactl','--physcpubind='+cpus,'--membind=3',str(COLD/'e1_driver'),str(root),mode,str(threads),str(C),str(int(always)),str(initial),str(batch),str(batches)]
        self.child=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,bufsize=1,env=env)
        self.lines=queue.Queue();self.root=root;self.log=(root/'driver.log').open('w')
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
        save(root/'driver-command.json',dict(command=command,pid=self.child.pid,environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH')}))
    def read(self):
        for line in self.child.stdout:self.log.write(line);self.log.flush();self.lines.put(line.strip())
    def command(self,text,marker,timeout=21600):
        self.child.stdin.write(text+'\n');self.child.stdin.flush();start=time.time()
        while time.time()-start<timeout:
            try:
                line=self.lines.get(timeout=5)
                if line.startswith(marker):return line
                if line.startswith('INSERT_PROGRESS'):status('inserting',variant=self.root.name,progress=line,pid=self.child.pid)
            except queue.Empty:pass
            if self.child.poll() is not None:raise RuntimeError('driver exited rc='+str(self.child.returncode)+' see '+str(self.root/'driver.log'))
        raise RuntimeError('driver timeout '+text)
    def close(self):
        self.command('quit','E1_DRIVER_COMPLETE');self.child.wait(timeout=60);self.thread.join(timeout=10);self.log.close();assert self.child.returncode==0
    def abort(self):
        if self.child.poll() is None:self.child.terminate();self.child.wait(timeout=30)


def prepare_inputs():
    if (COLD/'inputs/centers.bin').exists():
        identity=json.loads((COLD/'reuse-inputs.json').read_text())
        for name,digest in identity.items():assert c.sha(COLD/'inputs'/name)==digest
        return
    for s,key in [('img','x'),('text','y')]:
        a=matrix(DATA/('base_'+s+'_emb.fvecs'),'<f4')
        with (COLD/'inputs'/('insert-'+key+'.bin')).open('xb') as f:f.write(struct.pack('<II',200000,512));np.asarray(a[1800000:],dtype='<f4').tofile(f)
    centers=[]
    for s in ('img','text'):
        a=matrix(COLD/'inputs'/('base-'+s+'-1800k.bin'),'<f4');v=np.zeros(512,dtype='<f4')
        for row in a:np.add(v,row,out=v)
        v/=np.float32(1800000);centers.append(v)
    with (COLD/'inputs/centers.bin').open('xb') as f:f.write(struct.pack('<II',1,1024));np.concatenate(centers).astype('<f4').tofile(f)


def point_summary(rows,cohort='report'):
    return {(medium,a):[v for v in rows if v['cohort']==cohort and v['medium']==medium and v['alpha']==a] for medium in ('dram','ssd') for a in WEIGHTS}


def verify_reconstructable_coordinates(root):
    """Avoid duplicating 8GB per repeat: verify every physical record byte."""
    final=root/'checkpoint-4';tags=np.fromfile(str(final/'id-to-tag.u32'),dtype='<u4')
    canonical=np.memmap(str(MG/'data/xy.pages'),dtype='u1',mode='r',shape=(2000000,4096))
    import hashlib
    h=hashlib.sha256()
    with (HOT_COORD/'coordinates.pages').open('rb') as f:
        for first in range(0,len(tags),4096):
            expected=np.ascontiguousarray(canonical[tags[first:first+4096]]).tobytes()
            assert f.read(len(expected))==expected,'coordinate record differs from original tag vector'
            h.update(expected)
        assert f.read(1)==b''
    save(root/'coordinate-reconstruction.json',dict(status='ALL_RECORDS_BYTE_EQUAL',records=len(tags),record_bytes=4096,
        coordinates_sha256=h.hexdigest(),source=str(MG/'data/xy.pages'),source_sha256=c.sha(MG/'data/xy.pages'),
        mapping=str(final/'id-to-tag.u32'),mapping_sha256=c.sha(final/'id-to-tag.u32'),
        reconstruction='for internal_id in [0,n): copy canonical[tag[internal_id]*4096:(tag[internal_id]+1)*4096]; two records per8192B page',
        reason='retain reproducible canonical source and mapping instead of18 duplicate8GB coordinate files'))
    return h.hexdigest()


def driver_smoke():
    import run_g1
    old_run=c.RUN;c.RUN=COLD/'driver-smoke';c.RUN.mkdir()
    try:
        source_ids=np.fromfile(str(AB/'data-10k/subset_ids.u32'),dtype='<u4')
        for mode in ('pq','exact'):
            root=run_g1.fixture(mode)
            for key in ('x','y'):
                # Detach from the read-only mmap before truncating this owned
                # small-fixture file; asarray alone can retain its storage.
                a=np.array(matrix(root/('insert-'+key+'.bin'),'<f4')[:100],copy=True)
                with (root/('insert-'+key+'.bin')).open('wb') as f:f.write(struct.pack('<II',*a.shape));a.tofile(f)
            for space,key in [('img','x'),('text','y')]:
                (root/('query-'+key+'.bin')).symlink_to(DATA/('query_'+space+'_emb.fvecs'))
            shutil.copyfile(SPLIT/'report-ids.txt',root/'report-ids.txt')
            expected=root/'expected-codes';expected.mkdir()
            fresh=np.fromfile(str(root/'fresh-source-ids.u32'),dtype='<u4')[:100];order=np.concatenate((source_ids,fresh))
            for space in ('emb','loc'):
                a=np.asarray(matrix(MG/'data'/(space+'_pq_compressed.bin'),'u1')[order])
                with (expected/(space+'_pq_compressed.bin')).open('wb') as f:f.write(struct.pack('<II',*a.shape));a.tofile(f)
            driver=Driver(root,mode,4,100,False,initial=10000,batch=25,batches=4)
            try:
                for cp in range(5):
                    directory=root/('checkpoint-%d'%cp);directory.mkdir()
                    if cp:driver.command('insert %d 64 64 %s'%(cp*25,directory),'INSERT_BATCH_COMPLETE')
                    driver.command('checkpoint '+str(directory),'CHECKPOINT_READY')
                    tags,inverse=snapshot_codes(directory/'normal-exit.snapshot',directory,expected)
                    assert len(tags)==10000+cp*25 and np.array_equal(inverse[tags],np.arange(len(tags)))
                driver.close()
            except Exception:
                driver.abort();raise
        save(COLD/'driver-smoke/result.json',dict(status='E1_DRIVER_CHECKPOINT_AND_PQ_CODE_PARITY_PASS',initial=10000,insertions_each=100,
            modes=['pq','exact'],query_threads=4,update_threads=4,checkpoints_each=5,
            checks=['BFS reachability recorded as observation','no newly zero indegree','independent ledger recount matches topology','snapshot internal/tag mapping bijective','actual inserted PQ codes equal frozen full-data encoding']))
    finally:c.RUN=old_run


def main():
    status('waiting_global_lock');lock=(E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    active=None
    try:
        assert json.loads((COLD/'prepare-status.json').read_text())['stage']=='INITIAL_B_READY'
        if (COLD/'execution-identity.json').exists():raise RuntimeError('existing E1 execution: refuse unverified resume')
        shared=json.loads((MG/'data/shared.json').read_text())
        subset=json.loads((DATA/'subset.json').read_text())
        assert c.sha(DATA/'subset.json')==shared['identity']['subset_manifest_sha256']
        assert c.sha(DATA/'identity.json')==shared['identity']['identity_sha256']
        for filename,identity in shared['files'].items():assert c.sha(MG/'data'/filename)==identity['sha256'],'shared data changed '+filename
        passed=c.ROOT/'runtime/g1-in-degree-protection-v2'
        assert json.loads((passed/'status.json').read_text())['stage']=='G1_PASSED_BOTH_DISTANCES'
        gate=json.loads((passed/'build-identity.json').read_text())
        for name,digest in gate['sources'].items():assert c.sha(c.ROOT/'implementation'/name)==digest,'executor source differs from incoming-degree G1'
        replay=json.loads((c.ROOT/'runtime/degree-validation/result.json').read_text())
        assert replay['status']=='G1_AND_1850K_REPLAY_PASSED'
        for name,digest in replay['sources'].items():assert c.sha(c.ROOT/'implementation'/name)==digest,'executor source differs from 1850k replay'
        assert replay['original_42_became_unreachable']==0
        online=c.ROOT/'runtime/online-pq-audit-v1'
        online_proof=json.loads((online/'identity.json').read_text())
        assert online_proof['status']=='ONLINE_CODES_MATCH_SCALAR_ENCODER'
        for name,digest in online_proof['files'].items():assert c.sha(online/name)==digest
        assert c.sha(c.SOURCE/'include/pq_table.h')==online_proof['upstream_encoder_header']
        for path,digest in online_proof['inputs'].items():assert c.sha(path)==digest
        for space,digest in online_proof['pivots'].items():assert c.sha(MG/'data'/(space+'_pq_pivots.bin'))==digest
        assert shutil.disk_usage(str(HOT_COORD.parent)).free>8192000000+250000000,'insufficient final-coordinate NVMe3 margin; inspect the dedicated HVS reservation before choosing another device'
        assert shutil.disk_usage(str(HOT_TOPO.parent)).free>334372864+200000000,'insufficient topology NVMe3 margin'
        owned_hot(HOT_COORD);owned_hot(HOT_TOPO)
        save(COLD/'execution-identity.json',dict(variants=VARIANTS,default='pq-t4',independent_update_repetitions=3,gate_build=gate,
            R1='formal-ab-v2 B1, immutable existing 2M graph',query_split={s:c.sha(SPLIT/(s+'-ids.txt')) for s in ('tune','report')},
            query_engine=c.sha(MG/'bin/multigraph'),query_engine_source=c.sha(E/'multigraph-20261113/scripts/multigraph.cpp'),
            repair=False,deduplicate=False,recall='ID Recall@10, 95% measured brackets, no tail interpolation',
            extra_records='BFS unreachable observation, indegree-zero hard gate, independent ledger recount, degree histogram/full fraction, before-after edge totals, per-batch guard/reprune counters, GT ceiling loss and PQ expansion trend',
            incoming_degree_protection=True,global_reachability_guarantee=False,replay=replay,online_code_audit=online_proof,
            sampling='frozen depth axis; tune upward until95 bracket, report covers tune axis plus one higher depth, upward extension only for coverage; no extrapolation',
            storage=dict(topology_device='nvme3',coordinate_device='nvme3',cold_artifacts=str(COLD))))
        src=c.ROOT/'implementation';run(['clang-format','-style=LLVM','-i',src/'e1_driver.cpp'],'format-e1')
        units=[src/v for v in ('update_store.cpp','deg_prune.cpp','dual_update.cpp','e1_driver.cpp')]
        cmd=['g++-11']+gate['flags']+units+[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',COLD/'e1_driver']
        run(cmd,'compile-e1',timeout=900)
        save(COLD/'build-identity.json',dict(command=list(map(str,cmd)),sources={str(p):c.sha(p) for p in units+[src/'dual_update.h',src/'entry_frontier.h',src/'update_store.h',src/'in_degree_guard.h']},binary=c.sha(COLD/'e1_driver')))
        status('checking_e1_driver_on_small_fixture');driver_smoke()
        if (COLD/'gt/identity.json').exists():
            gt=json.loads((COLD/'gt/identity.json').read_text());assert gt['status']=='COMPLETE'
            for name,digest in gt['outputs'].items():assert c.sha(COLD/'gt'/name)==digest
            for path,digest in gt['inputs'].items():assert c.sha(path)==digest
        else:run(['python3',src/'e1_gt.py',COLD/'gt',DATA],'generate-checkpoint-GT',extra=dict(OPENBLAS_NUM_THREADS='4'))
        status('preparing_insert_inputs_and_centers');prepare_inputs()
        # R1 and every update checkpoint use the same NVMe devices; no PMEM
        # coordinate file is silently substituted into the SSD measurements.
        reuse_text=os.environ.get('HVS_E1_REUSE_MEASUREMENTS')
        if reuse_text:
            reuse=Path(reuse_text)
            previous=json.loads((reuse/'build-identity.json').read_text())
            for path,digest in previous['sources'].items():assert c.sha(path)==digest
            old_exec=json.loads((reuse/'execution-identity.json').read_text())
            assert old_exec['query_engine']==c.sha(MG/'bin/multigraph')
            assert old_exec['storage']['topology_device']==old_exec['storage']['coordinate_device']=='nvme3'
            ref_identity=json.loads((reuse/'reference-R1/identity.json').read_text())
            assert ref_identity['coordinates']==shared['files']['xy.pages']['sha256']
            (COLD/'reference-R1').symlink_to(reuse/'reference-R1',target_is_directory=True)
            reference=json.loads((reuse/'reference-R1/matched.json').read_text())
            initial_source=reuse/'variants/pq-t4-r1/checkpoint-0'
            initial_rows=json.loads((initial_source/'matched.json').read_text())
            reference_points=point_summary(reference);series={};query_L=0
            save(COLD/'reused-native-measurements.json',dict(source=str(reuse),
                reference_identity=ref_identity,initial_identity=c.sha(initial_source/'identity.json'),
                reason='same C++ executor, immutable graph, data, PQ codebooks, query engine and device placement; only snapshot-code validation was corrected'))
        else:
            status('copying_R1_coordinates_to_NVMe3');copy_prefix(MG/'data/xy.pages',HOT_COORD/'coordinates.pages',2000000*4096)
            r1=COLD/'reference-R1';r1.mkdir();rmeta=AB/'formal-ab-v2/packed/B1/compact.meta';rtopo=AB/'formal-ab-v2/packed/B1/compact.topo'
            proof=json.loads((AB/'formal-ab-v2/graphs-summary.json').read_text())['B1']
            assert proof['build']['signature']['data']==c.sha(DATA/'subset.json'),'R1 belongs to a different dataset'
            assert c.sha(HOT_COORD/'coordinates.pages')==shared['files']['xy.pages']['sha256']
            assert c.sha(AB/'formal-ab-v2/graphs/B1/float-graph.bin')==proof['build']['graph_sha256']
            assert c.sha(rmeta)==proof['pack']['files']['compact.meta'] and c.sha(rtopo)==proof['pack']['files']['compact.topo']
            unreachable=set(map(int,(c.ROOT/'runtime/revision3-diagnostics/B1-native-unreachable.txt').read_text().split()))
            reference=collect(r1,2000000,rmeta,rtopo,HOT_COORD/'coordinates.pages',MG/'data',COLD/'gt',unreachable,None,run,MG/'bin/multigraph',DATA,SPLIT)
            save(r1/'identity.json',dict(meta=c.sha(rmeta),topology=c.sha(rtopo),coordinates=c.sha(HOT_COORD/'coordinates.pages'),graph=proof))
            (HOT_COORD/'coordinates.pages').unlink()
            reference_points=point_summary(reference);series={};initial_rows=None;initial_source=None;query_L=0
        variants=[]
        schedule=[tuple(v)+(rep,) for rep in (1,2,3) for v in VARIANTS[:2]]+[tuple(v)+(rep,) for rep in (1,2,3) for v in VARIANTS[2:]]
        for base_name,mode,threads,C,always,repetition in schedule:
            name=base_name+'-r'+str(repetition)
            status('preparing_variant',variant=name);root=fixture(name);active=Driver(root,mode,threads,C,always)
            points=[]
            for checkpoint in range(5):
                cp=root/('checkpoint-%d'%checkpoint);cp.mkdir();n=1800000+checkpoint*50000
                if checkpoint:
                    active.command('insert %d %d %d %s'%(checkpoint*50000,query_L,min(query_L,512),cp),'INSERT_BATCH_COMPLETE')
                active.command('checkpoint '+str(cp),'CHECKPOINT_READY',timeout=1800)
                structure=json.loads((cp/'structure.json').read_text());mapping,inverse=snapshot_codes(cp/'normal-exit.snapshot',cp,MG/'data',online)
                assert len(mapping)==n
                for space in ('emb','loc'):(cp/(space+'_pq_pivots.bin')).symlink_to(MG/'data'/(space+'_pq_pivots.bin'))
                unreachable=set(map(int,(cp/'unreachable-tags.txt').read_text().split()))
                # Freeze physical topology before advancing the persistent driver.
                shutil.copyfile(HOT_TOPO/'graph.pages',cp/'graph.pages')
                save(cp/'identity.json',dict(n=n,mode=mode,variant=name,structure=structure,
                    topology=c.sha(cp/'graph.pages'),mapping=c.sha(cp/'id-to-tag.u32'),snapshot=c.sha(cp/'normal-exit.snapshot'),meta=c.sha(cp/'index.meta')))
                if checkpoint==0 and initial_rows is not None:
                    first=json.loads((initial_source/'identity.json').read_text())
                    current_identity=json.loads((cp/'identity.json').read_text())
                    assert all(current_identity[key]==first[key] for key in ('topology','mapping','meta','structure')),'immutable initial checkpoint differs'
                    rows=initial_rows;save(cp/'matched.json',rows);save(cp/'reused-initial-measurement.json',dict(source=str(initial_source),reason='identical initial graph, identity mapping, coordinates, PQ and entries; three measured immutable baseline rounds reused'))
                else:
                    rows=collect(cp,n,cp/'index.meta',HOT_TOPO/'graph.pages',HOT_COORD/'coordinates.pages',cp,COLD/'gt',unreachable,mapping,run,MG/'bin/multigraph',DATA,SPLIT,report_repeats=3 if checkpoint==0 else 1)
                if checkpoint==0:
                    if initial_rows is None:initial_rows=rows;initial_source=cp
                    query_L=max(v['matched']['labels'][1] for v in rows if v['medium']=='ssd' and v['cohort']=='tune')
                current=point_summary(rows);comparison=[]
                for (medium,a),values in current.items():
                    baseline=reference_points[medium,a];expanded=float(np.mean([v['matched']['expanded'] for v in values]));reference_expanded=float(np.mean([v['matched']['expanded'] for v in baseline]))
                    comparison.append(dict(n=n,inserted=checkpoint*50000,alpha=a,medium=medium,expanded=expanded,R1_expanded=reference_expanded,
                        increase_vs_R1=expanded/reference_expanded-1,GT_theoretical_limit=values[0]['GT_theoretical_limit']))
                save(cp/'comparison-to-R1.json',comparison);points.append(dict(checkpoint=checkpoint,n=n,repetition=repetition,structure=structure,comparison=comparison))
                save(root/'checkpoints.json',points)
                if base_name=='pq-t4':
                    series={str(a):trend([dict(inserted=p['checkpoint']*50000,expanded=next(v['expanded'] for v in p['comparison'] if v['alpha']==a and v['medium']=='dram')) for p in points],reference_points['dram',a][0]['matched']['expanded']) for a in WEIGHTS}
                    save(COLD/('pq-expansion-trend-r%d.json'%repetition),series)
                    if repetition==1:save(COLD/'pq-expansion-trend.json',dict(repetition=1,scope='first independent trajectory; final report must include all three',weights=series))
                render(COLD)
                # Actual codes are already archived inside the checked snapshot.
                # Preserve their hashes and exact reconstruction layout, then
                # remove only these two temporary query-prefix derivatives.
                save(cp/'pq-code-derivatives.json',dict(files={space:c.sha(cp/(space+'_pq_compressed.bin')) for space in ('emb','loc')},
                    reconstruction='e1_evaluate.snapshot_codes(normal-exit.snapshot, out, frozen_expected_codes, online_codes=runtime/online-pq-audit-v1, online_first=1800000)'))
                for space in ('emb','loc'):(cp/(space+'_pq_compressed.bin')).unlink()
                # The declared 25% rejection applies to final E1 relative to R1.
                if checkpoint==4 and any(v['increase_vs_R1']>.25 for v in comparison if v['medium']=='dram'):
                    raise RuntimeError('E1 final expansion increase exceeds25%; '+name)
            active.close();active=None
            status('archiving_variant',variant=name)
            coordinate_sha=verify_reconstructable_coordinates(root)
            save(root/'archive-identity.json',dict(coordinates_sha256=coordinate_sha,coordinates_reconstruction='coordinate-reconstruction.json',topology_checkpoints={p.parent.name:c.sha(p) for p in root.glob('checkpoint-*/graph.pages')}))
            # Delete only this run's verified hot copies, after the owning index
            # process has exited and the permanent cold copy has been checked.
            (HOT_COORD/'coordinates.pages').unlink();(HOT_TOPO/'graph.pages').unlink()
            variants.append(dict(name=name,mode=mode,repetition=repetition,threads=threads,C_ins=C,always_reprune=always,checkpoints=points));save(COLD/'variants-complete.json',variants)
        status('E1_COMPLETE',variants=[v['name'] for v in variants],next='user report checkpoint before E2/E3',default_reprune='pq',repair=False,deduplicate=False)
        render(COLD)
    except Exception as e:
        if active:active.abort()
        status('STOPPED_AT_E1',reason=str(e),dependent_experiments='not started');raise


if __name__=='__main__':main()
