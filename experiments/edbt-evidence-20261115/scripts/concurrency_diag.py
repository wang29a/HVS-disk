"""Existing frozen configurations; diagnostic topology residency, no retuning."""
import hashlib,json,os,signal,subprocess,time
from pathlib import Path
import common as c
import native_compare as N
import edbt_monitor as M

OUT=c.RUN/'native/concurrency-diagnosis'
BUDGET=4*2**30
THREADS=(4,16,32)

def cpu_snapshot(pid):
    p=Path('/proc')/str(pid);out={}
    try:
        f=(p/'stat').read_text().rsplit(')',1)[1].split()
        out.update(minflt=int(f[7]),majflt=int(f[9]),utime=int(f[11]),stime=int(f[12]))
        out['io']={k:int(v.strip()) for k,v in (x.split(':',1) for x in (p/'io').read_text().splitlines())}
        s=dict(x.split(':',1) for x in (p/'status').read_text().splitlines() if ':' in x)
        out['context_switches']={k:int(s[k]) for k in ('voluntary_ctxt_switches','nonvoluntary_ctxt_switches')}
        out['thread_cpu_ticks']={}
        for t in (p/'task').iterdir():
            try:
                z=(t/'stat').read_text().rsplit(')',1)[1].split();out['thread_cpu_ticks'][t.name]=int(z[11])+int(z[12])
            except OSError:pass
        out['cpu_khz']={}
        for cpu in M.CPUS:
            f=Path('/sys/devices/system/cpu/cpu'+str(cpu)+'/cpufreq/scaling_cur_freq')
            if f.exists():out['cpu_khz'][str(cpu)]=int(f.read_text())
    except (OSError,ValueError,KeyError):pass
    return out

def run(args,label,out,cpus,threads):
    M.CPUS=list(cpus)
    for attempt in range(1,4):
        raw=out.with_name(out.stem+'-attempt'+str(attempt)+'.json');tag=label+'-attempt'+str(attempt)
        env=dict(c.ENV,OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads))
        aa=[str(raw) if str(x)==str(out) else str(x) for x in args]
        full=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind='+','.join(map(str,cpus)),'--membind=3']+aa
        c.save(c.RUN/'commands'/(tag+'.json'),dict(command=full,start=time.time(),timeout=300))
        samples=[];log=c.RUN/'logs'/(tag+'.log');c.status('running',task=tag)
        with log.open('w') as f:
            c.CHILD=subprocess.Popen(full,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic()
            while c.CHILD.poll() is None:
                if time.monotonic()-start>300:c.stop();raise RuntimeError('bounded diagnostic timeout '+tag)
                sample=M.sample(c.CHILD.pid);sample['process']=cpu_snapshot(c.CHILD.pid);samples.append(sample);time.sleep(.5)
            code=c.CHILD.returncode;c.CHILD=None
        if code:raise RuntimeError('diagnostic executable exit '+str(code)+' '+str(log))
        points=c.load(raw);guards=[M.validate(samples,dict(point_start_unix=p.get('start_unix',p.get('point_start_unix')),measured_seconds=p.get('seconds',p.get('measured_seconds')))) for p in points]
        c.save(out.with_name(out.stem+'-monitor-attempt'+str(attempt)+'.json'),dict(samples=samples,guards=guards,command=full))
        if points and all(g['accepted'] for g in guards) and all(p['peak_rss_kib']*1024<=BUDGET for p in points):
            return dict(points=points,guards=guards,result=str(raw),attempt=attempt,monitor=str(out.with_name(out.stem+'-monitor-attempt'+str(attempt)+'.json')))
        if any(not g.get('numa_samples') for g in guards):raise RuntimeError('missing NUMA evidence '+tag)
        time.sleep(5)
    return None

def plans():
    out=[]
    for s in N.cases():
        if s['name'] not in ('openimages','laion-0.5'):continue
        if s['name']=='openimages':s['engine']=N.OUT/'bin/openimages32'
        old=N.OUT/'search'/s['name'];p=c.load(old/'frozen.json');key='weighted_pages-0.95'
        b,cb=p['native'][key],p['compact'][key]
        if not b['covered'] or not cb['covered']:raise RuntimeError('existing95% brackets not covered')
        jobs=[j for j in p['compact_jobs'] if j['label'] in cb['labels']]
        root=OUT/s['name'];root.mkdir(parents=True,exist_ok=True);jp=root/'frozen.jobs';N.jobs(jp,jobs)
        index=N.OUT/'indexes'/('openimages' if s['name']=='openimages' else 'laion')
        out.append((s,index,old,p,b,cb,root,jp))
    return out

def compact_args(s,jp,out,mode,threads=4,seconds=10):
    a=N.compact(s,jp,out,threads=threads,seconds=seconds);a[8]=mode;return a

def gate(s,root,jp):
    proof=root/'residency-gate.json'
    if proof.exists():
        if c.load(proof)['status']!='PASS':raise RuntimeError('prior residency gate failed')
        return
    points={}
    for mode in ('ssd','topo'):
        out=root/('gate-'+mode+'.json');a=compact_args(s,jp,out,mode,seconds=0);a[13]=root/('trace-'+mode);a[14]=1
        c.run(a,'diag-residency-gate-'+s['name']+'-'+mode,cpus='80-83',timeout=600)
        points[mode]=c.load(out)
    pairs=[]
    for a,b in zip(points['ssd'],points['topo']):
        same=len(a['queries'])==len(b['queries']) and all({k:v for k,v in x.items() if k!='topology_pages'}=={k:v for k,v in y.items() if k!='topology_pages'} for x,y in zip(a['queries'],b['queries']))
        for k in ('recall','coordinate_pages','coordinate_submitted_pages','reranked','candidates_after_union'):
            same= same and a[k]==b[k]
        for x,y in zip(a['graphs'],b['graphs']):
            for k in ('expanded','pq_space_evals','exact_space_evals','topology_pages','inspected','rejected'):same=same and x[k]==y[k]
            same=same and y['topology_submitted_pages']==0
        traces={}
        for suffix in ('','.ids','.dists','.legacy'):
            x=root/('trace-ssd.'+a['label']+suffix);y=root/('trace-topo.'+b['label']+suffix)
            traces[suffix]=c.sha(x)==c.sha(y);same=same and traces[suffix]
        pairs.append(dict(label=a['label'],equal=same,queries=len(a['queries']),traces=traces))
    ok=len(pairs)==len(points['ssd'])==len(points['topo']) and all(x['equal'] for x in pairs)
    c.save(proof,dict(status='PASS' if ok else 'FAIL',pairs=pairs,binary=c.sha(s['engine'])))
    if not ok:raise RuntimeError('SSD/topology-resident trajectories differ')

def compact_match(points,cb):
    by={p['label']:p for p in points};return N.P.matched([by[x] for x in cb['labels']],cb['target'])

def main():
    OUT.mkdir(parents=True,exist_ok=True);lock=c.lock();ps=plans()
    signature=dict(protocol=c.sha(c.ROOT/'CONCURRENCY-DIAG.md'),source=c.sha(__file__),budget=BUDGET,threads=THREADS,
        inputs={s['name']:dict(frozen=c.sha(old/'frozen.json'),binary=c.sha(s['engine']),manifest=c.sha(s['manifest']),native_binary=c.sha(N.OUT/'bin/native-counted'),index=c.sha(index/'identity.json')) for s,index,old,p,b,cb,root,jp in ps})
    stamp=OUT/'plan.json'
    if stamp.exists() and c.load(stamp)!=json.loads(json.dumps(signature)):raise RuntimeError('diagnostic plan identity changed')
    c.save(stamp,signature)
    for s,index,old,p,b,cb,root,jp in ps:gate(s,root,jp)
    pending=[];total=3*len(THREADS)*len(ps)*3
    for rep in range(1,4):
      for threads in THREADS:
       for s,index,old,p,b,cb,root,jp in (ps if rep%2 else list(reversed(ps))):
        methods=['native','ssd','topo'];methods=methods[rep-1:]+methods[:rep-1]
        for mode in methods:
            tag=mode+'-T'+str(threads)+'-rep'+str(rep);saved=root/(tag+'.json')
            if saved.exists():continue
            cpus=N.affinity(threads);out=root/(tag+'-raw.json')
            args=N.native(s,index,out,sorted(set(b['L'])),threads=threads,seconds=10) if mode=='native' else compact_args(s,jp,out,mode,threads,10)
            c.save(OUT/'status.json',dict(stage='running',case=s['name'],method=mode,threads=threads,repeat=rep,total=total,pid=os.getpid()))
            value=run(args,'diag-'+s['name']+'-'+tag,out,cpus,threads)
            if value:c.save(saved,value)
            else:pending.append([s['name'],mode,threads,rep,'guard_or_budget'])
    rows=[]
    for s,index,old,p,b,cb,root,jp in ps:
      for threads in THREADS:
       for mode in ('native','ssd','topo'):
        reps=[]
        for rep in range(1,4):
            f=root/(mode+'-T'+str(threads)+'-rep'+str(rep)+'.json')
            if not f.exists():continue
            data=c.load(f);v=N.match_native(data['points'],b) if mode=='native' else compact_match(data['points'],cb)
            reps.append(dict(matched=v,peak_rss_kib=max(x['peak_rss_kib'] for x in data['points']),source=str(f)))
        rows.append(dict(case=s['name'],threads=threads,method=mode,covered=len(reps)==3,repeats=reps))
    c.save(OUT/'summary.json',dict(results=rows,pending=pending,scope='diagnostic extra topology DRAM included in common4GiB ceiling; no retuning; native unchanged'))
    c.save(OUT/'status.json',dict(stage='complete' if not pending else 'complete_with_pending',pending=pending,total=total))
    c.status('concurrency_diagnosis_finished',summary=str(OUT/'summary.json'),pending=pending)
    if pending:raise RuntimeError('diagnostic pending points block downstream full builds')

if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.save(OUT/'status.json',dict(stage='failed',error=str(e)));raise
