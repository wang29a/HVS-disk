#!/usr/bin/env python3
"""Serial, resumable pointer-DRAM diagnostics with predeclared stop gates."""
import fcntl, hashlib, json, os, pathlib, signal, statistics, struct, subprocess, sys, time
import numpy as np
R=pathlib.Path(__file__).resolve().parent.parent
E=R.parent; O=R/'stage2'; D=pathlib.Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
OLD=E/'metric-consistent-20261111'
sys.path.insert(0,str(OLD/'scripts'))
import numa_guard
GRID=[20,40,60,80,100,120,160,200,300,400,600,800,1200,1600]
CHILD=None
ENV=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4')
ENV['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH','')
def save(path,x):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.partial'); tmp.write_text(json.dumps(x,indent=2)+'\n'); tmp.replace(path)
def load(path): return json.loads(path.read_text())
HASHES={}
def sha(path):
    stat=path.stat(); key=(str(path),stat.st_size,stat.st_mtime_ns)
    if key in HASHES: return HASHES[key]
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    value=h.hexdigest(); HASHES[key]=value; return value
def status(stage,**kw): save(O/'queue-status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
def stop(signum=None,frame=None):
    global CHILD
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
        try: CHILD.wait(10)
        except subprocess.TimeoutExpired: os.killpg(CHILD.pid,signal.SIGKILL); CHILD.wait()
    if signum is not None: status('interrupted',signal=signum); raise SystemExit(128+signum)
def cpu():
    values={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        key=line.split()[0]
        if key in ['cpu'+str(c) for c in [72,73,74,75,168,169,170,171]]:
            v=list(map(int,line.split()[1:9])); values[key]=sum(v)-v[3]-v[4]
    return sum(values.values())
def sample(pid):
    t=time.time(); own=None; numa=None
    try:
        p=pathlib.Path('/proc')/str(pid)
        fields=(p/'stat').read_text().rsplit(')',1)[1].split(); own=int(fields[11])+int(fields[12])
        if pathlib.Path(os.readlink(str(p/'exe'))).name=='pointer-search':
            stat=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
            policy=numa_guard.parse_maps((p/'numa_maps').read_text())
            if policy and not stat['State'].strip().startswith('Z'):
                numa=dict(time=t,cpu_mask=stat['Cpus_allowed_list'].strip(),**policy)
    except (OSError,ValueError,KeyError): pass
    disks={}
    for name in ('nvme2n1','nvme3n1'):
        x=list(map(int,(pathlib.Path('/sys/block')/name/'stat').read_text().split()))
        disks[name]=dict(write_bytes=x[6]*512,read_bytes=x[2]*512)
    return dict(time=t,cpu=cpu(),own=own,numa=numa,disks=disks)
def execute(cmd,label,extra=None,timed=False,timeout=1200,cpus='72-75'):
    global CHILD
    env=dict(ENV); env.update(extra or {})
    full=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind='+cpus,'--membind=3']+list(map(str,cmd))
    log=O/'logs'/(label+'.log'); log.parent.mkdir(exist_ok=True)
    save(O/'commands'/(label+'.json'),dict(command=full,environment=extra or {},started=time.time()))
    status('running',task=label,command=full)
    with log.open('w') as f:
        CHILD=subprocess.Popen(full,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        start=time.monotonic(); samples=[]
        while CHILD.poll() is None:
            if time.monotonic()-start>timeout: stop(); raise RuntimeError('bounded timeout '+label)
            if timed: samples.append(sample(CHILD.pid))
            time.sleep(.5 if timed else 2)
        code=CHILD.returncode; CHILD=None
    if code: raise RuntimeError(label+' failed '+str(code)+'; see '+str(log))
    return samples
def command(spec,cohort,out,grid,seconds=0):
    return [O/'bin/pointer-search',spec['meta'],spec['graph'],spec['x'],spec['y'],spec['qx'],spec['qy'],spec['alpha'],spec['gt'],spec[cohort],out,4,seconds,int(spec['filter'])]+list(grid)
def diagnostic(spec,cohort,grid,label):
    out=O/'results'/(label+'.json')
    if not out.exists(): execute(command(spec,cohort,out,grid),label,spec.get('env'))
    points=load(out)
    if [p['L'] for p in points]!=list(grid): raise RuntimeError('stale diagnostic grid '+label)
    return points
def crossing(points,target=.95):
    points=sorted(points,key=lambda x:x['L'])
    for p in points:
        if p['recall']==target: return p,p,0.
    for a,b in zip(points,points[1:]):
        if a['recall']<target<b['recall']: return a,b,(target-a['recall'])/(b['recall']-a['recall'])
    return None
def freeze(spec,label):
    path=O/'frozen'/(label+'.json')
    identity=dict(binary=sha(O/'bin/pointer-search'),meta=sha(pathlib.Path(spec['meta'])),graph=sha(pathlib.Path(spec['graph'])),spec=spec,query_files={k:sha(pathlib.Path(spec[k])) for k in ('qx','qy','alpha','gt','tune','report')})
    if path.exists():
        frozen=load(path)
        if frozen['identity']!=identity: raise RuntimeError('frozen input identity changed '+label)
        return frozen['report_L']
    p=diagnostic(spec,'tune',GRID,label+'-tune-coarse')
    pair=crossing(p)
    if not pair:
        save(path,dict(identity=identity,report_L=[],reason='training grid does not bracket 95%',max_recall=max(x['recall'] for x in p))); return []
    a,b,_=pair
    dense=sorted(set([a['L'],b['L']]+[int(round(a['L']+(b['L']-a['L'])*v)) for v in (.25,.5,.75)]))
    refined=diagnostic(spec,'tune',dense,label+'-tune-dense')
    # Frozen report points include one coarse neighbour on either side so
    # ordinary train/report variation can be shown without report-driven tuning.
    lower=[L for L in GRID if L<a['L']]; upper=[L for L in GRID if L>b['L']]
    report=sorted(set(dense+(lower[-1:] if lower else [])+(upper[:1] if upper else [])))
    save(path,dict(identity=identity,rule='training-only recall bracket, quartiles, one coarse neighbour each side; no latency selection',report_L=report,train_bracket=[a['L'],b['L']],training_dense=refined,created_unix=time.time()))
    return report
def matched(points):
    c=crossing(points)
    if not c: return dict(covered=False,observed_recall_range=[min(p['recall'] for p in points),max(p['recall'] for p in points)])
    a,b,w=c; r=dict(covered=True,L_bracket=[a['L'],b['L']],recall_bracket=[a['recall'],b['recall']],interpolation_weight=w)
    for k in ('expanded','exact_space_evals','inspected_edges','alpha_rejected'):
        r[k]=(1-w)*a[k]+w*b[k]
    qa={p['query_id']:p for p in a['queries']}; qb={p['query_id']:p for p in b['queries']}
    if set(qa)!=set(qb): raise RuntimeError('paired report query identity')
    r['query_ids']=sorted(qa)
    r['query_expanded']=[(1-w)*qa[q]['expanded']+w*qb[q]['expanded'] for q in r['query_ids']]
    r['mean_degree']=a['mean_degree']; r['reachable']=a['reachable']; r['nodes']=a['nodes']
    return r
def study(spec,label):
    grid=freeze(spec,label)
    if not grid: return dict(covered=False,reason='training recall not covered')
    report=diagnostic(spec,'report',grid,label+'-report-work')
    result=matched(report); save(O/'analysis'/(label+'-matched.json'),result); return result
def validate_window(samples,point):
    begin=point['point_start_unix']; end=begin+point['measured_seconds']; intervals=[]
    for a,b in zip(samples,samples[1:]):
        if a['time']<begin or b['time']>end or a['own'] is None or b['own'] is None: continue
        dt=b['time']-a['time']; external=max(0,b['cpu']-a['cpu']-(b['own']-a['own']))/(dt*os.sysconf('SC_CLK_TCK')*4)
        writes=sum(max(0,b['disks'][d]['write_bytes']-a['disks'][d]['write_bytes']) for d in a['disks'])/dt/2**20
        intervals.append(dict(external_cpu_fraction=external,device_write_mib_s=writes))
    live=[s['numa'] for s in samples if s['numa'] and begin<=s['time']<=end]
    numa=bool(len(live)>=3 and all(p['cpu_mask']=='72-75' and p['verified'] and not p['conflicting_policies'] for p in live))
    accepted=bool(len(intervals)>=5 and numa and statistics.mean(p['external_cpu_fraction'] for p in intervals)<=.10 and max(p['external_cpu_fraction'] for p in intervals)<=.35 and statistics.mean(p['device_write_mib_s'] for p in intervals)<=8 and max(p['device_write_mib_s'] for p in intervals)<=32)
    return dict(accepted=accepted,numa_verified=numa,intervals=intervals,numa_samples=live)
def timings(specs):
    pending=set(load(O/'timings/pending.json')) if (O/'timings/pending.json').exists() else set()
    names=list(specs)
    for rep in range(3):
        for name in names[rep:]+names[:rep]:
            spec=specs[name]; matched_file=O/'analysis'/(name+'-matched.json')
            if not matched_file.exists(): continue
            m=load(matched_file)
            if not m['covered']: continue
            for L in sorted(set(m['L_bracket'])):
                label='time-'+name+'-L'+str(L)+'-rep'+str(rep+1)
                accepted=O/'timings'/(label+'-accepted.json')
                if accepted.exists():
                    pending.discard(label); continue
                success=False
                for attempt in range(1,4):
                    out=O/'results'/(label+'-attempt'+str(attempt)+'.json')
                    samples=execute(command(spec,'report',out,[L],10),label+'-attempt'+str(attempt),spec.get('env'),True)
                    point=load(out)[0]; guard=validate_window(samples,point)
                    save(O/'timings'/(label+'-guard'+str(attempt)+'.json'),guard)
                    if guard['accepted']:
                        save(accepted,dict(point=point,guard=guard,result=str(out))); success=True; break
                    time.sleep(5)
                if not success: pending.add(label)
                else: pending.discard(label)
    save(O/'timings/pending.json',sorted(pending))
def spec_laion(a,graph,filtering=False):
    return dict(meta=str(graph/'full.meta'),graph=str(graph/'float-graph.bin'),x=str(D/'base_img_emb.fvecs'),y=str(D/'base_text_emb.fvecs'),qx=str(D/'query_img_emb.fvecs'),qy=str(D/'query_text_emb.fvecs'),alpha=str(D/('query-alpha-'+a+'.bin')),gt=str(D/('gt-alpha-'+a+'.bin')),tune=str(O/'tune-ids.txt'),report=str(O/'report-ids.txt'),filter=filtering)
def old_pointer():
    split=load(OLD/'analysis/tuning-split-v2.json'); specs={}
    data=pathlib.Path('/mnt/nvme3/wz/dataset/Openimage_disk')
    for name,stem,filtering in [('single','compact',False),('dual-noalpha','compact',False),('dual-full','full',True),('dual-full-bypass','full',False)]:
        graph=OLD/'scratch'/('single' if name=='single' else 'dual')
        s=dict(meta=str(graph/(stem+'.meta')),graph=str(graph/'float-graph.bin'),x=str(data/'base_img_emb.fvecs'),y=str(data/'base_text_emb.fvecs'),qx=str(data/'query_img_emb.fvecs'),qy=str(data/'query_text_emb.fvecs'),alpha=str(data/'base_gt_round_00/query_alpha.fvecs'),gt=str(pathlib.Path('/mnt/nvme2/wz/hvs-metric-consistent-20261111/concat-gt.bin') if name=='single' else data/'base_gt_round_00/top10_results.ivecs'),filter=filtering,env={'PTR_CONCAT':'1'} if name=='single' else {})
        for cohort in ('tune','report'):
            p=O/('openimage-'+cohort+'-ids.txt'); p.write_text('\n'.join(map(str,split['cohorts'][cohort]['query_ids']))+'\n'); s[cohort]=str(p)
        label='openimage-'+name
        result=study(s,label)
        if result['covered']: specs[label]=s
    timings(specs)
    save(O/'analysis/openimage-pointer-complete.json',dict(stage='complete',note='true pointer exact DRAM; includes filtering pair and three-arm depths; timed gaps in timings/pending.json'))
def bootstrap_gain(u,s):
    if u['query_ids']!=s['query_ids']: raise RuntimeError('paired queries differ')
    x=np.array(u['query_expanded']); y=np.array(s['query_expanded']); rng=np.random.RandomState(20261113)
    vals=[]
    for _ in range(2000):
        ix=rng.randint(0,len(x),len(x)); vals.append(1-float(y[ix].mean()/x[ix].mean()))
    return dict(gain=1-s['expanded']/u['expanded'],conditional_paired_95_ci=list(map(float,np.percentile(vals,[2.5,97.5]))),ci_note='conditional on observed fixed interpolation weights; queries resampled together')
def main():
    O.mkdir(exist_ok=True)
    for sub in ('results','analysis','frozen','timings','logs','commands'): (O/sub).mkdir(exist_ok=True)
    status('waiting_unified_build')
    deadline=time.monotonic()+50*3600
    while True:
        p=O/'build-status.json'; state=load(p) if p.exists() else {}
        if state.get('stage') in ('complete','complete_reused'): break
        if state.get('stage')=='failed': raise RuntimeError('unified build failed: '+state.get('error',''))
        if state.get('stage') in ('building_unified','waiting_global_lock'):
            try: os.kill(state['pid'],0)
            except ProcessLookupError: raise RuntimeError('unified build coordinator disappeared without a completed checkpoint')
        if time.monotonic()>deadline: raise RuntimeError('unified build wait exceeded 50h')
        time.sleep(20)
    lock=(E/'budget-reorder-20261103/queue.lock').open('a+')
    status('waiting_global_lock'); deadline=time.monotonic()+6*3600
    while True:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
        except BlockingIOError:
            if time.monotonic()>deadline: raise RuntimeError('global lock wait exceeded 6h')
            time.sleep(10)
    parity=load(O/'validation/pointer-parity.json')
    if parity['status']!='PASS' or parity['binary_sha256']!=sha(O/'bin/pointer-search'): raise RuntimeError('pointer correctness prerequisite')
    identity=load(D/'identity.json')
    for cohort,ids in identity['query_cohorts'].items(): (O/(cohort+'-ids.txt')).write_text('\n'.join(map(str,ids))+'\n')
    # Prior code-review gaps are short, independent of the U-shape gate.
    old_pointer()
    unified=O/'graphs/unified'; specs={}; values={}
    for a in ('0.1','0.5','0.9'):
        label='unified-'+a+'-off'; s=spec_laion(a,unified); specs[label]=s; values[a]=study(s,label)
    shape=all(v['covered'] for v in values.values())
    ratios={a:values[a]['expanded']/values['0.5']['expanded'] for a in ('0.1','0.9')} if shape else {}
    passed=bool(shape and all(v>=1.10 for v in ratios.values()))
    save(O/'analysis/u-shape-gate.json',dict(passed=passed,endpoint_over_center=ratios,results=values,rule='both endpoints at least 1.10x center in exact-pointer expanded nodes at held-out 95% recall'))
    timings(specs)
    # U1 remains a supplementary filtering arm, not the specialization baseline.
    supplements={}
    for a in ('0.1','0.5','0.9'):
        label='unified-'+a+'-on'; s=spec_laion(a,unified,True); result=study(s,label)
        if result['covered']: supplements[label]=s
    timings(supplements)
    if not passed:
        status('stopped_at_u_shape_gate',next='no specialized graphs or SSD routing; inspect analysis/u-shape-gate.json'); return
    fixed=load(O/'specialized-validation.json')
    if fixed['status']!='PASS' or fixed['builder_sha256']!=sha(O/'bin/build-fixed'): raise RuntimeError('specialized builder validation prerequisite')
    gains={}; specialized={}
    for a in ('0.1','0.5','0.9'):
        graph=O/'graphs'/('fixed-'+a); graph.mkdir(exist_ok=True)
        if not (graph/'graph-audit.json').exists():
            # 3 GiB output allowance + 2 GiB margin; source-only never creates
            # copied coordinate segments or per-alpha PQ codes.
            import shutil
            free=shutil.disk_usage(str(O)).free
            if free<5*2**30:
                rp=pathlib.Path('/mnt/nvme3/wz/.space-reserve')
                with (rp/'reserve.lock').open('a+') as reserve_lock:
                    fcntl.flock(reserve_lock,fcntl.LOCK_EX)
                    reserve=rp/'hvs-experiments-20261113.reserve'; st=reserve.stat()
                    take=((5*2**30-free+2**30-1)//2**30)*2**30
                    if st.st_uid!=os.getuid() or st.st_size<take+2**30: raise RuntimeError('reservation insufficient')
                    with reserve.open('r+b') as f: f.truncate(st.st_size-take); f.flush(); os.fsync(f.fileno())
                    meta=load(rp/'reservation.json'); now=reserve.stat()
                    meta.update(logical_GiB=now.st_size/2**30,allocated_GiB=now.st_blocks*512/2**30,updated_unix=time.time())
                    meta.setdefault('releases',[]).append(dict(bytes=take,purpose='U-shape fixed-'+a,time=time.time()))
                    save(rp/'reservation.json',meta)
            execute([O/'bin/build-fixed','dual',D/'base_img_emb.fvecs',D/'base_text_emb.fvecs',2000000,512,512,8,graph],'build-fixed-'+a,dict(HVS_FIXED_ALPHA=a,HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8'),False,48*3600,'72-79')
        label='fixed-'+a; s=spec_laion(a,graph)
        s['env']={'PTR_ENTRY_META':str(unified/'full.meta')}
        specialized[label]=s; result=study(s,label)
        if not result['covered']: gains[a]=dict(covered=False); continue
        g=bootstrap_gain(values[a],result)
        g['mean_degree_ratio']=result['mean_degree']/values[a]['mean_degree']
        g['budget_matched']=abs(g['mean_degree_ratio']-1)<=.05
        g['covered']=True; gains[a]=g
    timings(specialized)
    proceed=all(gains.get(a,{}).get('covered') and gains[a]['budget_matched'] and gains[a]['gain']>=.2 and gains[a]['conditional_paired_95_ci'][0]>0 for a in ('0.1','0.9'))
    save(O/'analysis/specialization-gate.json',dict(passed=proceed,results=gains,rule='both endpoint gains >=20%, paired CI positive, mean edge budget within 5%'))
    # The routing method requires a separate SSD protocol and physical budget;
    # a successful DRAM gate is not authorization to report an SSD speedup.
    status('dram_complete_ssd_design_ready' if proceed else 'stopped_at_specialization_gate',next='SSD routing implementation and physical-budget validation' if proceed else 'retain negative/one-sided results; no SSD routing queue')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop); signal.signal(signal.SIGINT,stop)
    try: main()
    except Exception as exc: stop(); status('failed',error=str(exc)); raise
