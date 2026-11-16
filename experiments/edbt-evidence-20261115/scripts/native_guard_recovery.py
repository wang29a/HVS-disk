"""Retain strict 2GiB evidence and add a frozen, equal-4GiB thread sweep."""
import fcntl,signal,sys,time
from pathlib import Path
import numpy as np
import common as c
import measure,native_compare as N
OUT=c.RUN/'native/guard-recovery';BUDGET=4*2**30

def plans():
    specs=N.cases();specs[0]['engine']=N.OUT/'bin/openimages32';out=[]
    for spec in specs:
        root=N.OUT/'search'/spec['name'];frozen=root/'frozen.json';gate=root/'gate.json'
        if c.load(gate)['status']!='PASS':raise RuntimeError('old dataset correctness gate missing')
        p=c.load(frozen);index=N.OUT/'indexes'/('openimages' if spec['name']=='openimages' else 'laion')
        out.append((spec,index,root,p))
    return out

def point_allowed(data,budget):
    return bool(data and data.get('points') and all(g.get('accepted') for g in data.get('guards',[])) and len(data['points'])==len(data['guards']) and all(p['peak_rss_kib']*1024<=budget for p in data['points']))

def preflight(ps):
    proof=OUT/'preflight.json'
    if proof.exists():
        d=c.load(proof)
        if d['monitor']!=c.sha(c.ROOT/'scripts/edbt_monitor.py'):raise RuntimeError('guard changed after live preflight')
        return
    results=[]
    # Both renamed executors must expose real NUMA maps in a full10s window.
    for spec,index,old,p in (ps[0],ps[1]):
        root=OUT/'preflight'/spec['name'];root.mkdir(parents=True,exist_ok=True)
        jp=root/'jobs.txt';N.jobs(jp,p['compact_jobs'][:1]);out=root/'compact.json';cpus=N.affinity(4)
        v=measure.run(N.compact(spec,jp,out,seconds=10),'guard-live-preflight-'+spec['name'],out,cpus)
        if not point_allowed(v,BUDGET):raise RuntimeError('live compact monitor gate failed')
        results.append(dict(case=spec['name'],binary=c.sha(spec['engine']),data=v))
    c.save(proof,dict(status='PASS',monitor=c.sha(c.ROOT/'scripts/edbt_monitor.py'),results=results))

def main():
    OUT.mkdir(parents=True,exist_ok=True);own=(OUT/'coordinator.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB);lock=c.lock()
    if not c.load(c.RUN/'priority2.json')['complete']:raise RuntimeError('P2 prerequisite')
    ps=plans();sig=dict(reason='observe exact managed child without executable-name gate; retain all old guards; frozen4GiB supplemental allowance based only on native32 scratch RSS',monitor=c.sha(c.ROOT/'scripts/edbt_monitor.py'),measure=c.sha(c.ROOT/'scripts/measure.py'),source=c.sha(__file__),budget=BUDGET,original_budget=2*2**30,
        inputs={s['name']:dict(frozen=c.sha(root/'frozen.json'),gate=c.sha(root/'gate.json'),native_index=c.sha(index/'identity.json'),binary_native=c.sha(N.OUT/'bin/native-counted'),binary_compact=c.sha(s['engine']),manifest=c.sha(s['manifest']),jobfile=c.sha(root/'compact-time.jobs')) for s,index,root,p in ps})
    stamp=OUT/'plan.json'
    if stamp.exists() and c.load(stamp)!=sig:raise RuntimeError('guard recovery identity changed')
    c.save(stamp,sig);preflight(ps);tasks=[];reused=[];memory_rejections=[]
    for spec,index,root,p in ps:
        for f in root.glob('native-T32-rep*-raw-attempt*.json'):
            values=c.load(f)
            if values and max(x['peak_rss_kib'] for x in values)*1024>2*2**30:memory_rejections.append(dict(case=spec['name'],file=str(f),peak_rss_kib=max(x['peak_rss_kib'] for x in values),reason='original2GiB budget exceeded; not retroactively accepted as4GiB'))
        for rep in range(1,4):
            for threads in N.THREADS:
                for method in ('native','compact'):
                    tag=method+'-T'+str(threads)+'-rep'+str(rep);folder=OUT/'runs'/spec['name'];folder.mkdir(parents=True,exist_ok=True);save=folder/(tag+'.json');old=root/(tag+'.json')
                    # Reuse only accepted original points from <=16threads;
                    # all32thread points are fresh after the budget amendment.
                    if not save.exists() and threads<=16 and old.exists():
                        v=c.load(old)
                        if not point_allowed(v,2*2**30):raise RuntimeError('old accepted point failed reuse audit')
                        c.save(save,dict(v,provenance=dict(source=str(old),sha256=c.sha(old),original_budget=2*2**30,reuse='unchanged binaries/configs/cache; original point fits both ceilings')));reused.append(str(old))
                    tasks.append((rep,threads,method,spec,index,root,p,save))
    c.save(OUT/'reuse-and-original-memory.json',dict(reused=reused,original_2GiB_rejections=memory_rejections))
    # Interleave methods and workloads at each thread count. Configuration
    # selection stays entirely frozen; only measurement guard/budget change.
    order=[]
    for rep in range(1,4):
        workloads=ps[rep-1:]+ps[:rep-1]
        for threads in N.THREADS:
            for spec,index,root,p in workloads:
                for method in (('native','compact') if rep%2 else ('compact','native')):
                    order.append(next(t for t in tasks if t[0]==rep and t[1]==threads and t[2]==method and t[3]['name']==spec['name']))
    pending=[]
    for rep,threads,method,spec,index,root,p,saved in order:
        done=sum(t[-1].exists() for t in tasks)
        c.save(OUT/'status.json',dict(stage='running',case=spec['name'],method=method,threads=threads,repeat=rep,accepted=done,total=len(tasks),pending=pending,pid=c.os.getpid()))
        if saved.exists():continue
        cpus=N.affinity(threads);out=saved.with_name(saved.stem+'-raw.json')
        args=N.native(spec,index,out,p['native_L'],threads=threads,seconds=10) if method=='native' else N.compact(spec,root/'compact-time.jobs',out,threads=threads,seconds=10)
        v=measure.run(args,'guard-recovery-'+spec['name']+'-'+saved.stem,out,cpus,extra=dict(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads)))
        if point_allowed(v,BUDGET):c.save(saved,dict(v,provenance=dict(budget=BUDGET,config_sha256=c.sha(root/'frozen.json'))))
        else:pending.append([spec['name'],threads,method,rep,'noise_or_4GiB_budget'])
    rows=[];primary2=[]
    for spec,index,root,p in ps:
      for threads in N.THREADS:
       folder=OUT/'runs'/spec['name']
       for key,b in p['native'].items():
        cb=p['compact'][key];reps=[];p2=[]
        if b['covered'] and cb['covered']:
         for rep in range(1,4):
            nf=folder/('native-T'+str(threads)+'-rep'+str(rep)+'.json');df=folder/('compact-T'+str(threads)+'-rep'+str(rep)+'.json')
            if not nf.exists() or not df.exists():continue
            nv,dv=c.load(nf),c.load(df);n=N.match_native(nv['points'],b);by={x['label']:x for x in dv['points']};d=N.P.matched([by[x] for x in cb['labels']],cb['target'])
            if not d['covered']:raise RuntimeError('deterministic compact recall changed')
            record=dict(native=n,compact=d,latency_speedup=n['latency_us']/d['latency_us'],peak_native_kib=max(x['peak_rss_kib'] for x in nv['points']),peak_compact_kib=max(x['peak_rss_kib'] for x in dv['points']))
            reps.append(record)
            if record['peak_native_kib']*1024<=2*2**30 and record['peak_compact_kib']*1024<=2*2**30:p2.append(record)
        row=dict(dataset=spec['name'],threads=threads,case=key,covered=len(reps)==3,repeats=reps)
        rows.append(row);primary2.append(dict(dataset=spec['name'],threads=threads,case=key,covered=len(p2)==3,repeats=p2,note='2GiB-uncovered if either measured peak exceeds original allowance'))
    result=dict(results=rows,pending=pending,original_2GiB=primary2,original_memory_rejections=memory_rejections,memory='4GiB supplemental common ceiling; old <=16T accepted points reused,32T newly measured; original2GiB feasibility separate',scope='unchanged source graphs/binaries/GT/frozen L,r/no-shared-cache')
    c.save(OUT/'summary.json',result);c.save(N.OUT/'summary.json',result)
    c.save(c.RUN/'priority3.json',dict(complete=not pending,summary=str(OUT/'summary.json'),budget='supplemental4GiB; original2GiB coverage retained'))
    c.save(OUT/'status.json',dict(stage='complete' if not pending else 'complete_with_pending',accepted=sum(t[-1].exists() for t in tasks),total=len(tasks),pending=pending))
    c.status('native_guard_recovery_finished',priority3=c.load(c.RUN/'priority3.json'))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.save(OUT/'status.json',dict(stage='failed',error=str(e)));raise
