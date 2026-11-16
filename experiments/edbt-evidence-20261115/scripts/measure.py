"""Bounded measured windows using the already validated NUMA3 guard."""
import subprocess,sys,time
import common as c
sys.path.insert(0,str(c.E/'metric-consistent-20261111/scripts'))
sys.path.insert(0,str(c.MG/'scripts'))
import edbt_monitor as M

def run(args,label,out,cpus,attempts=3,extra=None):
    M.CPUS=list(cpus)
    for attempt in range(1,attempts+1):
        tag=label+'-attempt'+str(attempt);env=dict(c.ENV);env.update(extra or {})
        actual_out=out.with_name(out.stem+'-attempt'+str(attempt)+out.suffix)
        actual_args=[str(actual_out) if str(x)==str(out) else str(x) for x in args]
        full=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind='+','.join(map(str,cpus)),'--membind=3']+actual_args
        c.save(c.RUN/'commands'/(tag+'.json'),dict(command=full,environment=extra or {},start=time.time(),timeout=240))
        log=c.RUN/'logs'/(tag+'.log');log.parent.mkdir(exist_ok=True)
        c.status('running',task=tag)
        with log.open('w') as f:
            c.CHILD=subprocess.Popen(full,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);start=time.monotonic();samples=[];timeout=False
            while c.CHILD.poll() is None:
                if time.monotonic()-start>240:c.stop();timeout=True;break
                samples.append(M.sample(c.CHILD.pid));time.sleep(.5)
            code=c.CHILD.returncode if c.CHILD is not None else None;c.CHILD=None
        if timeout:guards=[dict(accepted=False,reason='bounded timeout')];points=[]
        else:
            if code:raise RuntimeError('executable error '+tag+'; '+str(log))
            points=c.load(actual_out);guards=[M.validate(samples,dict(point_start_unix=p.get('start_unix',p.get('point_start_unix')),measured_seconds=p.get('seconds',p.get('measured_seconds')))) for p in points]
        c.save(c.RUN/'guards'/(tag+'.json'),dict(guards=guards,result=str(actual_out)))
        if points and all(g['accepted'] for g in guards):return dict(points=points,guards=guards,attempt=attempt,result=str(actual_out))
        if points and any(not g.get('numa_samples') for g in guards):raise RuntimeError('NUMA observation missing for a full measured point; fail fast, see '+tag)
        time.sleep(5)
    return None

def quiet():
    start=time.monotonic()
    while True:
        choice=M.choose()
        if choice['choice']['external_busy_fraction']<.10:return choice
        if time.monotonic()-start>180:raise RuntimeError('no quiet four-core block in NUMA3; bounded defer')
        time.sleep(5)
