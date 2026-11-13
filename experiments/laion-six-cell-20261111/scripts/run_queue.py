"""Full corrected Laion build followed only by the six filtering cells.

One global queue lock, original IDs, no reorder/density/cache/single-space arms.
Resumable checkpoints; deterministic train-only selection; held-out timings.
"""
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import struct
import sys
import time
import numpy as np
import common as c
import phase_runtime as p
from preflight import run_preflight, read, compile_tools
from gt_audit import matrix

N = 10642155
PQ = c.ROOT.parent/'laion/full'
GRAPH = c.ROOT/'scratch/full'
SOURCE = c.STORE/'full'
TARGETS = [90, 92, 94, 95, 96, 97, 98]


def await_memory(required, label, timeout=1800):
    start = time.monotonic()
    while c.node_memory()['MemFree'] < required:
        c.status('waiting_for_memory', task=label, required_bytes=required,
                 available_bytes=c.node_memory()['MemFree'])
        if time.monotonic()-start > timeout:
            raise RuntimeError(label+': insufficient NUMA3 free memory after bounded wait')
        time.sleep(30)


def build_full():
    GRAPH.mkdir(exist_ok=True)
    SOURCE.mkdir(exist_ok=True)
    estimate = read(c.ROOT/'analysis/build-cost-estimate.json')
    if not (SOURCE/'graph-audit.json').exists():
        threads = estimate['construction_threads']
        # Cost probes do not write a graph used for search. This is the full,
        # normal HybridIndex.build entry point, with the corrected library.
        required = max(80<<30, estimate['peak_rss_projection_bytes'])
        await_memory(required, 'full-graph-build')
        projected_source = estimate['probes'][-1]['float_graph_bytes']*N/1000000.
        if shutil.disk_usage(str(SOURCE)).free < projected_source*1.5+(2<<30):
            raise RuntimeError('nvme2 lacks float-graph checkpoint space; no old files deleted')
        c.quiet(c.BUILD_CPUS[:threads], timeout=1800)
        start = time.time()
        c.run([c.ROOT/'bin/build-graph', 'dual', c.DATA/'base_img_emb.fvecs',
               c.DATA/'base_text_emb.fvecs', N, 512, 512, threads, SOURCE],
              'build-laion-full-corrected', c.BUILD_CPUS[:threads],
              dict(OMP_NUM_THREADS=str(threads), OMP_THREAD_LIMIT=str(threads), HVS_SOURCE_ONLY='1'),
              timeout=max(86400, int(estimate['planning_upper_seconds']*1.5)))
        c.save(c.ROOT/'analysis/full-build-cost.json', dict(wall_seconds=time.time()-start,
               threads=threads, cpus=c.BUILD_CPUS[:threads], source_directory=str(SOURCE)))
    audit = read(SOURCE/'graph-audit.json')
    if audit['status'] != 'SOURCE_COMPLETE' or audit['nodes'] != N or audit['quantized_lost_grid_points']:
        raise RuntimeError('full source checkpoint audit failed')
    shutil.copyfile(str(SOURCE/'full.meta'), str(GRAPH/'full.meta'))
    proof = c.ROOT/'analysis/full-layout-check.json'
    if not proof.exists():
        if not (GRAPH/'full.topo').exists():
            c.run([c.ROOT/'bin/full-layout', 'write', SOURCE/'float-graph.bin', GRAPH/'full.meta',
                   GRAPH/'full.topo', c.ROOT/'analysis/full-layout-write.json'],
                  'materialize-laion-full', c.CPUS[:2], timeout=21600)
        c.run([c.ROOT/'bin/full-layout', 'check', SOURCE/'float-graph.bin', GRAPH/'full.meta',
               GRAPH/'full.topo', proof], 'verify-laion-full', c.CPUS[:2], timeout=21600)
    if read(proof)['status'] != 'PASS' or read(proof)['nodes'] != N:
        raise RuntimeError('full topology validation gate failed')
    identity = c.ROOT/'analysis/full-graph-identity.json'
    if not identity.exists():
        c.save(identity, dict(graph=audit, validation=read(proof),
            hashes={str(f):c.digest(f) for f in (SOURCE/'float-graph.bin', GRAPH/'full.meta', GRAPH/'full.topo')},
            builder_sha256=c.digest(c.ROOT/'bin/build-graph'),
            library_sha256=c.digest(c.ROOT/'bin/libpipeann-metric.a'),
            objective='alpha*sqrt(squared_l2_x)+(1-alpha)*sqrt(squared_l2_y)',
            merge_intervals='corrected final interval included',
            quantization='outward floor/ceil; continuous source coverage retained',
            order='original IDs; no Starling; no graph substitution'))


def pq_identity():
    output = c.ROOT/'analysis/pq-identity.json'
    if output.exists(): return
    files = {}
    for name in ('emb_pq_compressed.bin', 'loc_pq_compressed.bin', 'emb_pq_pivots.bin', 'loc_pq_pivots.bin'):
        path = PQ/name
        with path.open('rb') as stream:
            shape = struct.unpack('<II',stream.read(8))
        if 'compressed' in name and (shape != (N,64) or path.stat().st_size != 8+N*64):
            raise RuntimeError('PQ codes mismatch full Laion original IDs')
        files[name] = dict(path=str(path), bytes=path.stat().st_size, sha256=c.digest(path))
    c.save(output, dict(files=files, code_bytes_per_space=64, original_ids=True,
        note='Existing graph-independent codes/codebooks reused. Search loader also verifies both dimensions.'))


def prepare_workload(workload):
    root = c.ROOT/'workloads'/('alpha-'+str(workload['fixed_alpha']))
    for name in ('analysis', 'results', 'logs', 'scratch', 'bin'):
        (root/name).mkdir(parents=True,exist_ok=True)
    binary = root/'bin/metric-search'
    if not binary.exists(): binary.symlink_to(c.ROOT/'bin/metric-search')
    split_path = root/'analysis/tuning-split-v2.json'
    if not split_path.exists():
        ids = np.random.RandomState(20261111).permutation(1000)
        inputs = dict(query1=c.DATA/'query_img_emb.fvecs',query2=c.DATA/'query_text_emb.fvecs',
                      alpha=Path(workload['alpha']),gt=Path(workload['gt']))
        cohorts = {}
        for cohort, selected in [('tune',ids[:500]),('report',ids[500:])]:
            files = {}
            for key, source in inputs.items():
                dtype = np.uint32 if key=='gt' else np.float32
                values = np.asarray(matrix(source,dtype)[selected])
                path = root/'scratch'/(cohort+'-'+key+'.bin')
                with path.open('wb') as out:
                    out.write(struct.pack('<II',*values.shape))
                    values.astype('<u4' if key=='gt' else '<f4').tofile(out)
                files[key] = dict(path=str(path),sha256=c.digest(path))
            cohorts[cohort] = dict(query_ids=selected.tolist(),files=files)
        c.save(split_path,dict(seed=20261111,alpha=workload['fixed_alpha'],cohorts=cohorts,
            tuning='training queries only; latency never used for selection'))
    split = read(split_path)
    a,b = (split['cohorts'][k]['query_ids'] for k in ('tune','report'))
    if len(a)!=500 or len(b)!=500 or set(a)&set(b) or len(set(a+b))!=1000:
        raise RuntimeError('invalid query split')
    for cohort in split['cohorts'].values():
        for f in cohort['files'].values():
            if c.digest(f['path'])!=f['sha256']: raise RuntimeError('slice hash changed')
    return root,split


def layout(arm, kind):
    if arm!='full' or kind!='T_XY':
        raise RuntimeError('only six-cell Full original-order T_XY is authorized in this queue')
    directory = p.ROOT/'scratch/full-T_XY'
    directory.mkdir(exist_ok=True)
    for name,target in [('group-1.bin',GRAPH/'full.topo'),('group-6.bin',c.COORD)]:
        link = directory/name
        if not link.exists(): link.symlink_to(target)
        if link.resolve()!=target.resolve(): raise RuntimeError('layout symlink changed')
    return GRAPH/'full.meta', GRAPH/'full.topo', directory


def parity():
    proof = p.ROOT/'analysis/query-parity.json'
    if proof.exists(): return
    # Both media must return identical candidates, scores and trajectory.
    for filtering in (False,True):
        for medium in ('DRAM','SSD'):
            label='parity-f{}-{}'.format(int(filtering),medium)
            p.query('full','T_XY',medium,False,filtering,80,40,'tune',label,trace=True)
        for suffix in ('','.ids','.dists'):
            a=p.ROOT/'analysis'/('parity-f{}-DRAM.L80'.format(int(filtering))+suffix)
            b=p.ROOT/'analysis'/('parity-f{}-SSD.L80'.format(int(filtering))+suffix)
            if a.read_bytes()!=b.read_bytes(): raise RuntimeError('DRAM/SSD parity failed')
        files=p.SPLIT['cohorts']['tune']['files']
        qx,qy=matrix(files['query1']['path']),matrix(files['query2']['path'])
        weights=matrix(files['alpha']['path'])[:,0]
        coords=np.memmap(str(c.COORD),dtype=np.float32,mode='r',shape=(N+1,1024))
        stem=p.ROOT/'analysis'/('parity-f{}-DRAM.L80'.format(int(filtering)))
        ids=np.fromfile(str(stem)+'.ids',dtype=np.uint32).reshape(500,10)
        scores=np.fromfile(str(stem)+'.dists',dtype=np.float32).reshape(500,10)
        for q in range(0,500,10):
            vectors=np.asarray(coords[ids[q]],dtype=np.float64)
            expected=float(weights[q])*np.sqrt(np.sum((vectors[:,:512]-qx[q])**2,axis=1))
            expected+=(1-float(weights[q]))*np.sqrt(np.sum((vectors[:,512:]-qy[q])**2,axis=1))
            if np.max(np.abs(expected-scores[q])/np.maximum(1e-8,np.abs(expected)))>5e-6:
                raise RuntimeError('exact returned scores use wrong target')
    c.save(proof,dict(status='PASS',DRAM_SSD_equal_ids_scores_and_trajectory=True,
        independent_score_checks_per_filter=50,L=80,r=40,queries=500))


def tune(exact):
    nav='exact' if exact else 'pq'
    output=p.ROOT/'analysis'/('frozen-full-T_XY-'+nav+'.json')
    if output.exists(): return read(output)
    rows=[]
    seen=set()
    def evaluate(L,r,filtering):
        if (L,r,filtering) in seen: return
        seen.add((L,r,filtering))
        label='tune-full-T_XY-{}-f{}-L{}-r{}'.format(nav,int(filtering),L,r)
        point=p.query('full','T_XY','DRAM',exact,filtering,L,r,'tune',label)
        rows.append(dict(filtering=filtering,L=L,r=r,recall=point['recall'],
            pages=point['submitted_pages'],pq=point['pq_space_evals'],
            exact=point['navigation_exact_space_evals'],rerank_evals=point['exact_space_evals'],
            source=str(p.ROOT/'results'/(label+'.json'))))
    # Small coarse screen, then conditional extensions based solely on tuning.
    levels=[40,80,160,320]
    for L in levels:
        for r in ([L] if exact else sorted(set([min(40,L),min(80,L),L]))):
            for filtering in (False,True): evaluate(L,r,filtering)
    for L in (600,1000,1600,2400,3200):
        deficient=[f for f in (False,True) if max(x['recall'] for x in rows if x['filtering']==f)<98]
        if not deficient: break
        levels.append(L)
        for filtering in deficient:
            for r in ([L] if exact else sorted(set([80,min(160,L),L]))): evaluate(L,r,filtering)
    # Add one intermediate L around each training 95% crossing. For PQ,
    # refine r around that training crossing with bounded, predeclared depths.
    for filtering in (False,True):
        ordered=sorted([x for x in rows if x['filtering']==filtering and x['r']==x['L']],key=lambda x:x['L'])
        crossing=next(((a,b) for a,b in zip(ordered,ordered[1:]) if a['recall']<95<=b['recall']),None)
        if crossing:
            left,right=crossing
            mid=int(math.ceil((left['L']+right['L'])/40.))*20
            refinements=sorted(set([left['L'],mid,right['L']]))
        else:
            best=min(ordered,key=lambda x:abs(x['recall']-95))
            refinements=[best['L']]
        for L in refinements:
            for r in ([L] if exact else sorted(set([min(v,L) for v in (20,40,60,80,120,160,240)]+[L]))):
                evaluate(L,r,filtering)
    selected,frontier=[],[]
    for filtering in (False,True):
        candidates=[x for x in rows if x['filtering']==filtering]
        for x in candidates:
            if not any(y['recall']>=x['recall'] and y['pages']<=x['pages'] and
                y['pq']+y['exact']<=x['pq']+x['exact'] and
                (y['recall']>x['recall'] or y['pages']<x['pages'] or y['pq']+y['exact']<x['pq']+x['exact'])
                for y in candidates): frontier.append(x)
        for target in TARGETS:
            covering=[x for x in candidates if x['recall']>=target]
            if not covering: continue
            for key in (lambda x:(x['pages'],x['pq']+x['exact'],x['L'],x['r']),
                        lambda x:(x['pq']+x['exact'],x['pages'],x['L'],x['r'])):
                x=min(covering,key=key)
                if x not in selected: selected.append(x)
        # Explicit coverage limitation instead of inventing target results.
        if not any(x['filtering']==filtering for x in selected):
            selected.append(max(candidates,key=lambda x:x['recall']))
    frozen=dict(arm='full',layout='T_XY',exact=exact,selected=selected,tuning_frontier=frontier,
        all_training_points=rows,targets=TARGETS,tune_queries=500,report_queries=500,
        selection='lexicographic (pages, navigation evaluations) and reverse; no latency optimization',
        rerank_compute_note='Exact rerank work is reported separately, not equated to PQ evaluations.',
        frozen_unix=time.time(),split_sha256=c.digest(p.ROOT/'analysis/tuning-split-v2.json'))
    c.save(output,frozen)
    return frozen


def run():
    c.init()
    # If a separate preflight owns the global lock, this starts only after it
    # releases it. Markers skip already verified stages; failures stay visible.
    run_preflight()
    source_identity={name:c.digest(c.ROOT/'scripts'/name) for name in ('build_graph.cpp','full_layout.cpp')}
    build_proof=c.ROOT/'analysis/formatted-tools.json'
    if not build_proof.exists() or read(build_proof)!=source_identity:
        # Wait for all cost probes before replacing their executable.
        compile_tools()
        c.save(build_proof,source_identity)
    pq_identity()
    build_full()
    for workload in read(c.ROOT/'analysis/inventory.json')['workloads']:
        root,split=prepare_workload(workload)
        c.ACTIVE=root
        p.ROOT,p.B,p.SPLIT,p.COORD,p.PQ=root,c,split,c.COORD,str(PQ)+'/'
        p.layout=layout
        await_memory(80<<30,'six-cell-DRAM')
        identity=dict(search_sha256=c.digest(c.ROOT/'bin/metric-search'),
            runtime_sha256=c.digest(c.ROOT/'scripts/phase_runtime.py'),
            queue_sha256=c.digest(Path(__file__)),split_sha256=c.digest(root/'analysis/tuning-split-v2.json'),
            graph_identity_sha256=c.digest(c.ROOT/'analysis/full-graph-identity.json'),
            pq_identity_sha256=c.digest(c.ROOT/'analysis/pq-identity.json'))
        ip=root/'analysis/measurement-inputs.json'
        if ip.exists() and read(ip)!=identity: raise RuntimeError('measurement inputs changed')
        c.save(ip,identity)
        parity()
        for exact,media in [(True,['DRAM']),(False,['DRAM','SSD'])]:
            manifest=tune(exact)
            p.measure(manifest,media)
        from summarize import summarize
        summarize(root)
        c.save(root/'analysis/done.json',dict(completed_unix=time.time(),pending=p.pending_points()))
    c.ACTIVE=c.ROOT
    pending={str(x.parent.parent):read(x) for x in (c.ROOT/'workloads').glob('*/analysis/pending-measurements.json') if read(x)}
    c.status('complete_with_pending' if pending else 'complete',pending=pending,
             scope='Laion alpha .5 and .1; exact DRAM, PQ DRAM, PQ SSD; filter on/off')


if __name__=='__main__':
    c.init()
    c.save(c.ROOT/'queue-status.json',dict(stage='queued',pid=os.getpid(),
        task='waiting for global serial lock',updated_unix=time.time()))
    lock=open(str(c.ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    c.save(c.ROOT/'queue-status.json',dict(stage='lock_acquired',pid=os.getpid(),updated_unix=time.time()))
    try: run()
    except Exception as error:
        c.status('failed',error=str(error));raise
