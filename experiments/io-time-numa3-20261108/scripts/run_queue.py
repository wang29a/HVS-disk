#!/usr/bin/env python3
"""Native four-thread I/O timing on NUMA node 3."""
import collections
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
BUILD=Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
SOURCE=BUILD/'experiments/concurrency-alpha'
PREDECESSOR=BUILD/'experiments/budget-reorder-20261103'
PREVIOUS_BATCH=BUILD/'experiments/io-time-20261104'
NUMA_CPUS=set(range(72,96))|set(range(168,192))
DATASETS=['openimage','cc3m','howto100m','laion']
METHODS=['baseline','full','merge','no-alpha']
ENV=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',
         OMP_THREAD_LIMIT='4',OMP_MAX_ACTIVE_LEVELS='1',OMP_NESTED='FALSE',OMP_PROC_BIND='close',
         OMP_PLACES='cores',HVS_POINT_SECONDS='10')
ENV['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH','')
for k in ['HVS_ONE_PASS','PHASE_DUMP_PREFIX','PHASE_DISABLE_OBSERVER']:
    ENV.pop(k,None)

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,data):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,indent=2)+'\n');os.replace(str(tmp),str(path))
def status(**data):save(ROOT/'status.json',dict(updated_at=now(),numa_node=3,**data))
def log(s):print(now(),s,flush=True)
def sha(path):
    h=hashlib.sha256()
    with open(str(path),'rb')as f:
        for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
    return h.hexdigest()

def levels(points,target):
    for a,b in zip(points,points[1:]):
        if a['recall']<=target<=b['recall']:return sorted(set([a['L'],b['L']]))
    return []

def cases():
    result=[]
    for dataset in DATASETS:
        condition='fixed-0.5' if dataset=='laion' else 'mixed'
        for method in METHODS:
            path=SOURCE/'results'/('{}__{}__T4__{}__rep1.json'.format(dataset,condition,method))
            raw=json.loads(path.read_text());assert raw['threads']==4
            l90=levels(raw['points'],90);l95=levels(raw['points'],95)
            if not l90:raise RuntimeError('Missing reference 90% bracket')
            argv=raw['command'][4:17]
            if argv[2]!='4':raise RuntimeError('Unexpected thread argument')
            index=Path(argv[1])
            if not index.is_dir():raise RuntimeError('Index prefix missing '+str(index))
            stats={str(f.resolve()):dict(bytes=f.stat().st_size,mtime_ns=f.stat().st_mtime_ns)
                   for f in index.iterdir()if f.is_file()}
            if not stats:raise RuntimeError('Empty index prefix')
            result.append(dict(dataset=dataset,condition=condition,method=method,
                binary=str(ROOT/'bin'/('phase-baseline' if method=='baseline' else 'phase-separated')),
                argv=argv,L=sorted(set(l90+l95)),L90=l90,L95=l95,control_L=l90[0],
                expected_points={str(p['L']):p for p in raw['points']},input_sha256=raw['input_sha256'],
                reference_json=str(path),reference_sha256=sha(path),index_stats=stats))
    return result

def check_identity():
    build=json.loads((ROOT/'analysis/build.json').read_text())
    for name,h in build['sources'].items():
        if sha(ROOT/'scripts'/name)!=h:raise RuntimeError('Source changed: new timing experiment identity required')
    for name,h in build['binaries'].items():
        if sha(ROOT/'bin'/name)!=h:raise RuntimeError('Binary changed: '+name)
    for spec in build['libraries'].values():
        if sha(spec['path'])!=spec['sha256']:raise RuntimeError('Native library changed')

def wait_for_predecessor():
    old_lock=None
    while True:
        try:d=json.loads((PREDECESSOR/'status.json').read_text())
        except (ValueError,OSError):d=dict(status='unavailable')
        status(status='waiting_predecessor',predecessor=str(PREDECESSOR),
               predecessor_status=d.get('status'),predecessor_task=d.get('task'),
               note='No native dataset search runs until predecessor is complete and its queue lock is free.')
        if d.get('status')=='complete':
            old_lock=open(str(PREDECESSOR/'queue.lock'),'r')
            try:fcntl.flock(old_lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
            except BlockingIOError:old_lock.close();old_lock=None
        time.sleep(15)
    log('PREDECESSOR_COMPLETE_LOCK_ACQUIRED')
    return old_lock

def execute(case,label,ls,env=None):
    out=ROOT/'results'/(label+'.json');record=ROOT/'analysis'/(label+'-run.json')
    if out.exists() and record.exists():
        run=json.loads(out.read_text())
        if run.get('numa_node')!=3:raise RuntimeError('Unexpected NUMA policy in saved run')
        return run['points']
    e=dict(ENV);e.update(env or {})
    cmd=[case['binary']]+case['argv']+[str(l)for l in ls]
    status(status='running',task=label,dataset=case['dataset'],method=case['method'],threads=4)
    log('RUN '+label)
    with (ROOT/'logs'/(label+'.log')).open('w')as stdout,(ROOT/'logs'/(label+'.pidstat')).open('w')as telemetry:
        started=time.time();p=subprocess.Popen(cmd,env=e,stdout=stdout,stderr=subprocess.STDOUT)
        monitor=subprocess.Popen(['pidstat','-h','-u','-r','-w','-p',str(p.pid),'1'],stdout=telemetry,stderr=subprocess.STDOUT)
        try:code=p.wait()
        finally:monitor.terminate();monitor.wait()
        if code:raise RuntimeError('{} exit {}'.format(label,code))
    text=(ROOT/'logs'/(label+'.log')).read_text()
    if 'HVS_IO_PHASE_COMPLETE'not in text:raise RuntimeError('Incomplete phase run')
    rows=[json.loads(line[len('HVS_POINT '):])for line in text.splitlines()if line.startswith('HVS_POINT ')]
    if [r['L']for r in rows]!=ls:raise RuntimeError('Missing L point')
    for r in rows:
        expected=case['expected_points'][str(r['L'])]
        for name in ['recall','comparisons','topology_ios','rerank_ios']:
            if abs(r[name]-expected[name])>1e-7:raise RuntimeError('Original native result/counter mismatch '+name)
        if not e.get('HVS_ONE_PASS') and r['seconds']<10:raise RuntimeError('Short timing point')
        if r['reader_total_us']>r['external_service_us']+.01:raise RuntimeError('Double-counted reader time')
        if r['observer_enabled']:
            if abs(r['reader_coupled_us']+r['reader_topology_us']+r['reader_coordinate_us']-r['reader_total_us'])>1e-6:
                raise RuntimeError('Phase sum mismatch')
            total_count=r['coupled_backend_requests']+r['topology_backend_requests']+r['coordinate_backend_requests']
            total_bytes=r['coupled_backend_bytes']+r['topology_backend_bytes']+r['coordinate_backend_bytes']
            if abs(total_count*8192-total_bytes)>.01:raise RuntimeError('Unexpected request size')
    save(out,dict(dataset=case['dataset'],condition=case['condition'],method=case['method'],threads=4,numa_node=3,
        command=cmd,started_unix=started,finished_unix=time.time(),points=rows,reference_sha256=case['reference_sha256']))
    save(record,dict(command=cmd,observer_enabled='PHASE_DISABLE_OBSERVER'not in e,one_pass=bool(e.get('HVS_ONE_PASS'))))
    log('DONE '+label)
    return rows

def parity(case):
    base='{}__{}__parity'.format(case['dataset'],case['method'])
    dumps=[]
    for enabled in [True,False]:
        label=base+('__on' if enabled else '__off');prefix=ROOT/'analysis'/label
        env=dict(HVS_ONE_PASS='1',PHASE_DUMP_PREFIX=str(prefix))
        if not enabled:env['PHASE_DISABLE_OBSERVER']='1'
        execute(case,label,case['L'],env);dumps.append(prefix)
    for l in case['L']:
        for suffix in ['ids','dists']:
            if sha(str(dumps[0])+'.L{}.{}'.format(l,suffix))!=sha(str(dumps[1])+'.L{}.{}'.format(l,suffix)):
                raise RuntimeError('Observer affects search output')
    save(ROOT/'analysis'/(base+'-check.json'),dict(status='PASS',IDs_and_distances_byte_equal=True,
        original_native_counters_and_recall_equal=True,L=case['L']))

def summarize():
    rows=[];groups=collections.defaultdict(list)
    for p in (ROOT/'results').glob('*__r*__on.json'):
        run=json.loads(p.read_text())
        for target in [90,95]:
            for a,b in zip(run['points'],run['points'][1:]):
                if a['recall']<=target<=b['recall']:
                    f=(target-a['recall'])/(b['recall']-a['recall'])if b['recall']!=a['recall']else 0
                    row=dict(dataset=run['dataset'],method=run['method'],target_recall=target,run=p.stem)
                    for k in ['qps','latency_us','external_service_us','reader_total_us','reader_coupled_us','reader_coupled_backend_us',
                        'reader_topology_us','reader_coordinate_us','reader_topology_submit_us','reader_topology_wait_us',
                        'reader_topology_blocking_us','reader_coordinate_submit_us','reader_coordinate_wait_us','reader_coordinate_blocking_us',
                        'coupled_backend_requests','topology_backend_requests','coordinate_backend_requests',
                        'coupled_backend_bytes','topology_backend_bytes','coordinate_backend_bytes','normalized_candidate_evaluations',
                        'legacy_topology_wait_us','legacy_coordinate_read_us','io_us']:
                        row[k]=a[k]+f*(b[k]-a[k])
                    rows.append(row);groups[run['dataset'],run['method'],target].append(row);break
    means=[]
    for (d,m,t),rs in sorted(groups.items()):
        row=dict(dataset=d,method=m,target_recall=t,repeats=len(rs))
        for metric in rs[0]:
            if metric in ['dataset','method','target_recall','run']:continue
            v=[r[metric]for r in rs];row[metric]=statistics.mean(v);row[metric+'_sd']=statistics.stdev(v)if len(v)>1 else None
        means.append(row)
    save(ROOT/'analysis/summary.json',dict(matched_per_repeat=rows,mean_matched=means,
        status='partial_until_global_complete',definition='Sum of disjoint reader-call wall time; baseline backend time is a child, never added again.'))

def main():
    for d in ['scripts','logs','analysis','results','bin']:(ROOT/d).mkdir(exist_ok=True)
    own_lock=open(str(ROOT/'queue.lock'),'w');fcntl.flock(own_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cpus=set(os.sched_getaffinity(0))
    if not cpus or not cpus.issubset(NUMA_CPUS):
        raise RuntimeError('CPU affinity must be restricted to NUMA node 3')
    binding=subprocess.check_output(['numactl','--show'],env=ENV,universal_newlines=True)
    policy=dict(line.split(':',1) for line in binding.splitlines() if ':' in line)
    if policy.get('policy','').strip()!='bind' or policy.get('membind','').split()!=['3']:
        raise RuntimeError('Memory allocation must be bound to NUMA node 3')
    log('NUMA3_BINDING_VERIFIED '+str(sorted(cpus)))
    check_identity()
    controller={name:sha(ROOT/'scripts'/name)for name in ['run_queue.py','run.sh','analyze_phase.py']}
    controller_file=ROOT/'analysis/controller.json'
    if controller_file.exists()and json.loads(controller_file.read_text())!=controller:raise RuntimeError('Controller changed; use a new experiment identity')
    save(controller_file,controller)
    cs=cases();save(ROOT/'analysis/selected-cases.json',cs)
    save(ROOT/'analysis/protocol.json',dict(threads=4,maximum_threads=4,repeats=3,datasets=DATASETS,methods=METHODS,
        native_libraries_unchanged=True,predecessor=str(PREDECESSOR),serial=True,beam=4,top_k=10,
        numa_node=3,numa_memory_policy='bind',cpu_affinity='72-95,168-191',previous_batch=str(PREVIOUS_BATCH),
        measured_seconds_per_point=10,warmup='256-query native driver pilot; unchanged from archived measurement driver',
        queries=1000,new_GT=False,targets=[90,95],L_policy='Only measured brackets in archived curves; missing targets not extrapolated.',
        baseline='Inclusive synchronous coupled reader-call wall time; child backend time exported but not added again.',
        separated='Topology submission + blocking wait, plus coordinate blocking reader call.',
        coordinates='Original exact X/Y vectors; exact distance computation excluded from coordinate IO.',
        total_timed_points=sum(len(c['L'])for c in cs)*3+len(cs)*3,
        no_overlapping_dataset_experiments=True))
    predecessor_lock=wait_for_predecessor();check_identity()
    previous_lock=(PREVIOUS_BATCH/'queue.lock').open('r')
    fcntl.flock(previous_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if json.loads((PREVIOUS_BATCH/'status.json').read_text())['status']!='paused_by_user':
        raise RuntimeError('The previous timing batch must remain paused')
    for c in cs:
        for path,h in c['input_sha256'].items():
            if sha(path)!=h:raise RuntimeError('Query/GT input changed')
        for path,st in c['index_stats'].items():
            s=Path(path).stat()
            if s.st_size!=st['bytes']or s.st_mtime_ns!=st['mtime_ns']:raise RuntimeError('Index changed while waiting')
    processes=[];files=[]
    for cmd,name in [(['iostat','-t','-dxm','1','nvme3n1'],'iostat.log'),
                     (['mpstat','-P',','.join(map(str,sorted(cpus))),'5'],'mpstat.log')]:
        f=(ROOT/'logs'/name).open('a');files.append(f);processes.append(subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,env=ENV))
    try:
        for c in cs:parity(c)
        for rep in [1,2,3]:
            order=METHODS[rep-1:]+METHODS[:rep-1]
            for dataset in DATASETS:
                for method in order:
                    c=next(c for c in cs if c['dataset']==dataset and c['method']==method)
                    base='{}__{}__r{}'.format(dataset,method,rep)
                    tasks=[True,False]if rep%2 else[False,True]
                    for observe in tasks:
                        execute(c,base+('__on'if observe else'__off'),c['L']if observe else[c['control_L']],
                                None if observe else dict(PHASE_DISABLE_OBSERVER='1'))
                    summarize()
        subprocess.run(['python3',str(ROOT/'scripts/analyze_phase.py')],env=ENV,check=True)
        status(status='complete',completed_cases=16,completed_timed_curves=96,
               no_new_GT=True,scope='Native four-thread IO reader timing; legacy timers exported alongside uniform reader timers.')
        log('NATIVE_FOUR_THREAD_IO_TIME_COMPLETE')
    finally:
        for process in processes:process.terminate();process.wait()
        for f in files:f.close()
        previous_lock.close()
        predecessor_lock.close()

if __name__=='__main__':
    try:main()
    except Exception as e:
        status(status='failed',error=str(e));traceback.print_exc();raise
