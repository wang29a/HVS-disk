"""Five physical layouts, common search, shared traces and field-patch probes."""
import collections
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
from run_all import ROOT, BUILD, ENV, save, status, log, run, digest, now

LAYOUTS=['T_XY','TXY','TX_Y','TY_X','T_X_Y']
THREADS=[1,4,32,64]
LS=[20,40,60,80,100,120,160,200,300,400]
SECONDS=10
ENGINE=ROOT/'bin/layout-engine'

def execute(cmd,label,output,env=None,timed=False):
    if output.exists():return json.loads(output.read_text())
    pending=output.with_name(output.name+'.partial')
    cmd=[str(x) for x in cmd];cmd[7]=str(pending)
    e=dict(ENV);e.update(env or {})
    status(status='running',stage='layout-experiments',task=label)
    log('RUN '+label)
    pidstat=None
    with open(str(ROOT/'logs'/(label+'.log')),'w') as out,open(str(ROOT/'logs'/(label+'.pidstat')),'w') as monitor:
        start=time.time();p=subprocess.Popen(cmd,env=e,stdout=out,stderr=subprocess.STDOUT)
        if timed:pidstat=subprocess.Popen(['pidstat','-h','-u','-r','-w','-p',str(p.pid),'1'],stdout=monitor,stderr=subprocess.STDOUT)
        try:code=p.wait()
        finally:
            if pidstat:pidstat.terminate();pidstat.wait()
        if code:raise RuntimeError('{} exit {}'.format(label,code))
    data=json.loads(pending.read_text());os.replace(str(pending),str(output))
    for p in pending.parent.glob(pending.name+'.L*.queries.json'):
        os.replace(str(p),str(output.with_name(p.name.replace(pending.name,output.name,1))))
    save(ROOT/'analysis'/(label+'-run.json'),dict(command=cmd,started_unix=start,finished_unix=time.time(),source_driver_sha256=digest(ROOT/'scripts/layout_engine.cpp')))
    log('DONE '+label)
    return data

def args(c,layout,directory,operation,output):
    return [ENGINE,operation,c['meta'],c['topo'],c['coord'],directory,layout,output]

def query_args(c,layout,directory,output,threads,ls,seconds=SECONDS):
    return args(c,layout,directory,'query',output)+[c['prefix']+'/',c['query1'],c['query2'],c['alpha'],c['gt'],threads,seconds]+ls

def build_engine():
    spec=json.loads((BUILD/'experiments/concurrency-alpha/analysis/build.json').read_text())['separated']
    if digest(spec['library'])!=spec['library_sha256']:raise RuntimeError('Original library changed')
    run(['clang-format','-i','-style=LLVM',str(ROOT/'scripts/layout_engine.cpp'),str(ROOT/'scripts/layout_store.h')],'format-layout')
    sources={p.name:digest(p) for p in [ROOT/'scripts/layout_engine.cpp',ROOT/'scripts/layout_store.h']}
    prior=ROOT/'analysis/layout-build.json'
    if prior.exists():
        old=json.loads(prior.read_text())
        if old['sources']!=sources or old['library_sha256']!=spec['library_sha256']:
            raise RuntimeError('Changed layout code/library requires a new experiment directory')
    cmd=['g++-11']+spec['flags']+[str(ROOT/'scripts/layout_engine.cpp'),spec['library'],
        '-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-o',str(ENGINE)]
    run(cmd,'build-layout')
    save(ROOT/'analysis/layout-build.json',dict(command=cmd,source_commit='d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e',
        library_sha256=spec['library_sha256'],binary_sha256=digest(ENGINE),sources=sources))

def native_reference(c):
    label=c['name']+'-native-reference';output=ROOT/'analysis'/(label+'.json')
    if output.exists():return json.loads(output.read_text())
    base=BUILD/'experiments/hotdata-alpha/bin/hot-search'
    cmd=['numactl','--physcpubind=0-47,96-111','--interleave=0,1',str(base),'float',c['prefix']+'/',
         '1','4',c['query1'],c['query2'],c['alpha'],c['gt'],'10','l2','6','0','17','40','100','200']
    env=dict(ENV,HVS_ONE_PASS='1',HVS_BACKEND='ssd',HVS_MEMORY_DIR='/',HVS_SCHEME='full',HVS_DUMP_PREFIX=str(ROOT/'analysis'/label))
    run(cmd,label,env)
    text=(ROOT/'logs'/(label+'.log')).read_text()
    if 'HVS_STEADY_COMPLETE' not in text:raise RuntimeError('Native reference incomplete')
    data=[json.loads(line[len('HVS_POINT '):]) for line in text.splitlines() if line.startswith('HVS_POINT ')]
    save(output,data);return data

def validate_query(c,layout,directory,rep):
    label='{}-{}-rep{}-parity'.format(c['name'],layout,rep)
    output=ROOT/'analysis'/(label+'.json');dump=ROOT/'analysis'/label
    rows=execute(query_args(c,layout,directory,output,1,[40,100,200]),label,output,
        dict(LAYOUT_ONE_PASS='1',LAYOUT_DUMP=str(dump),LAYOUT_TRACE=str(dump)+'.trace'))
    native=native_reference(c)
    for p,n in zip(rows,native):
        if p['L']!=n['L'] or abs(p['recall']-n['recall'])>1e-7 or abs(p['pq_comparisons']-n['comparisons'])>1e-7:
            raise RuntimeError('Original-search parity mismatch '+label)
        for suffix in ('ids','dists'):
            original=ROOT/'analysis'/('{}-native-reference.L{}.{}'.format(c['name'],p['L'],suffix))
            actual=Path(str(dump)+'.L{}.{}'.format(p['L'],suffix))
            if digest(original)!=digest(actual):raise RuntimeError('Per-query result bytes differ '+label+' '+suffix)
    reference=ROOT/'analysis'/(c['name']+'-canonical.trace')
    actual=Path(str(dump)+'.trace')
    if reference.exists():
        if digest(reference)!=digest(actual):raise RuntimeError('Candidate/expansion trace mismatch '+label)
    else:shutil.copyfile(str(actual),str(reference))
    save(ROOT/'analysis'/(label+'-check.json'),dict(status='PASS',IDs_and_distances_byte_equal=True,
        common_search_trace_sha256=digest(actual),reference='unchanged d2a39e2 native Full search; physical-layout traces identical'))
    return reference

def summarize():
    groups=collections.defaultdict(list);runs=0
    for p in (ROOT/'results').glob('*__query.json'):
        name,layout,rep,threads,_=p.stem.split('__');rows=json.loads(p.read_text());runs+=1
        for row in rows:groups[name,layout,int(threads[1:]),row['L']].append(row)
    means=[]
    metrics=['recall','qps','latency_us','p50_us','p95_us','p99_us','submitted_pages','submitted_bytes',
             'requested_pages','page_hits','inspected_edges','alpha_rejected_edges','pq_comparisons',
             'expanded_nodes','reranked_nodes','prep_us','coarse_us','rerank_us','read_submit_wait_us']
    for key,rows in sorted(groups.items()):
        row=dict(zip(('dataset','layout','threads','L'),key));row['repeats']=len(rows)
        for m in metrics:
            row[m]=statistics.mean(p[m] for p in rows)
            row[m+'_stddev']=statistics.stdev(p[m] for p in rows) if len(rows)>1 else None
        means.append(row)
    curves=collections.defaultdict(list)
    for row in means:
        if row['repeats']==3:curves[row['dataset'],row['layout'],row['threads']].append(row)
    matched=[]
    for key,rows in sorted(curves.items()):
        if len(rows)!=len(LS):continue
        rows.sort(key=lambda p:p['recall'])
        for target in [90,95]:
            for a,b in zip(rows,rows[1:]):
                if a['recall']<=target<=b['recall']:
                    f=(target-a['recall'])/(b['recall']-a['recall']) if b['recall']!=a['recall'] else 0
                    v=dict(zip(('dataset','layout','threads'),key));v['target_recall']=target
                    for m in ['L','qps','latency_us','submitted_pages','submitted_bytes','pq_comparisons','reranked_nodes']:v[m]=a[m]+f*(b[m]-a[m])
                    matched.append(v);break
    save(ROOT/'analysis/layout-query-summary.json',dict(completed_query_runs=runs,mean_curves=means,matched_recall=matched,
        scope='Five layouts share a new controlled search harness; not the original coupled baseline implementation. No dynamic ANN update claim.'))

def expected_bytes(c,layout):
    raw=Path(c['meta']).read_bytes();n,d1,d2,R,M=struct.unpack_from('<IIIII',raw);widths=[4+R*(4+2*M),d1*4,d2*4]
    masks={'TXY':[7],'TX_Y':[3,4],'TY_X':[5,2],'T_XY':[1,6],'T_X_Y':[1,2,4]}[layout]
    result=0
    for mask in masks:
        width=sum(widths[k] for k in range(3) if mask&(1<<k));npp=8192//width
        if not npp:raise RuntimeError('Record overflow unsupported')
        result+=((n+npp-1)//npp)*8192
    return result

def alpha_distribution(c):
    import math
    raw=Path(c['alpha']).read_bytes();q,d=struct.unpack_from('<II',raw);values=list(struct.unpack_from('<{}f'.format(q),raw,8));hist=[0]*10
    for a in values:
        if not math.isfinite(a) or a<0 or a>1:raise RuntimeError('Query alpha')
        hist[min(9,int(a*10))]+=1
    save(ROOT/'analysis'/(c['name']+'-query-alpha.json'),dict(count=q,sha256=digest(c['alpha']),histogram=hist,
        edges=[i/10 for i in range(11)],sorted_alpha=sorted(values),mean=sum(values)/q,min=min(values),max=max(values),
        definition='Supplied weights; buckets [lo,hi), last includes 1. Raw float32 retained.'))

def main(configs):
    build_engine()
    subprocess.run(['python3',str(ROOT/'scripts/test_layout.py')],env=ENV,check=True)
    for c in configs:
        alpha_distribution(c);native_reference(c)
        original_hashes={k:digest(c[k]) for k in ('meta','topo','coord')}
        save(ROOT/'analysis'/(c['name']+'-source-identity.json'),dict(hashes=original_hashes,input_stats=c['input_stats'],
            pq={name:digest(Path(c['prefix'])/name) for name in ['emb_pq_pivots.bin','loc_pq_pivots.bin','emb_pq_compressed.bin','loc_pq_compressed.bin']}))
        for rep in (1,2,3):
            order=LAYOUTS[rep-1:]+LAYOUTS[:rep-1]
            for layout in order:
                base='{}__{}__r{}'.format(c['name'],layout,rep)
                # Finished work is reusable only if the full set of outputs exists.
                expected=[ROOT/'results'/(base+'__t{}__query.json'.format(t)) for t in THREADS]
                if all(p.exists() for p in expected) and (ROOT/'analysis'/(base+'-completed.json')).exists():continue
                for key,v in c['input_stats'].items():
                    stat=Path(c[key]).stat()
                    if stat.st_size!=v['bytes'] or stat.st_mtime_ns!=v['mtime_ns']:raise RuntimeError('Source changed '+c[key])
                directory=ROOT/'scratch'/base;directory.mkdir(exist_ok=True)
                if shutil.disk_usage(str(directory)).free<expected_bytes(c,layout)+(20<<30):raise RuntimeError('Insufficient nvme3 headroom')
                built=ROOT/'analysis'/(base+'-build.json')
                # Never resume from a potentially interrupted field-patch state.
                for p in directory.glob('group-*.bin'):p.unlink()
                if built.exists():built.unlink()
                execute(args(c,layout,directory,'build',built),base+'-build',built)
                file_hashes={p.name:digest(p) for p in directory.glob('group-*.bin')}
                save(ROOT/'analysis'/(base+'-physical-files.json'),dict(layout=layout,hashes=file_hashes,bytes=expected_bytes(c,layout),reorder=False,mapping='identity global ID order'))
                trace=validate_query(c,layout,directory,rep)
                ts=THREADS[rep-1:]+THREADS[:rep-1]
                for t in ts:
                    out=ROOT/'results'/(base+'__t{}__query.json'.format(t))
                    points=execute(query_args(c,layout,directory,out,t,LS),out.stem,out,timed=True)
                    if [p['L'] for p in points]!=LS or any(p['seconds']<SECONDS or p['samples']%1000 for p in points):raise RuntimeError('Invalid timed curve')
                    replay=ROOT/'results'/(base+'__t{}__replay.json'.format(t))
                    value=execute(args(c,layout,directory,'replay',replay)+[trace,t,SECONDS],replay.stem,replay,timed=True)
                    if value['seconds']<SECONDS:raise RuntimeError('Short replay')
                    summarize()
                for pattern in ['uniform','hot']:
                    for kind,mask in [('X',2),('Y',4),('XY',6),('T',1)]:
                        out=ROOT/'results'/(base+'__{}__{}__patch.json'.format(pattern,kind))
                        value=execute(args(c,layout,directory,'patch',out)+[20261102+rep,10000,mask,pattern],out.stem,out,timed=True)
                        if value.get('status')!='PASS' or value.get('restored_page_mismatches')!=0:raise RuntimeError('Patch probe failed')
                save(ROOT/'analysis'/(base+'-completed.json'),dict(status='complete',finished_at=now(),query_runs=4,replay_runs=4,patch_runs=8,
                    scratch_files_removed_after_byte_validated_restore=True))
                for p in directory.glob('group-*.bin'):p.unlink()
                directory.rmdir()
        # Prove original input files were never modified by probe updates.
        if any(digest(c[k])!=h for k,h in original_hashes.items()):raise RuntimeError('Original source identity changed')
        save(ROOT/'analysis'/(c['name']+'-completed.json'),dict(status='complete',finished_at=now(),original_source_hashes_preserved=True))
    summarize()
