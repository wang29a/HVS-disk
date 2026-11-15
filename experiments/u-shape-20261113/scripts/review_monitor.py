"""Preserve real affinity evidence when choosing a quiet block within NUMA3."""
import os,pathlib,time
CPUS=[84,85,86,87]
def ticks():
    out={}
    for line in pathlib.Path('/proc/stat').read_text().splitlines():
        f=line.split()
        if f[0].startswith('cpu') and f[0][3:].isdigit():
            v=list(map(int,f[1:9]));out[int(f[0][3:])]=sum(v)-v[3]-v[4]
    return out
def choose():
    global CPUS
    a=ticks();start=time.monotonic();time.sleep(1);b=ticks();dt=time.monotonic()-start
    options=[]
    for first in range(72,96,4):
        block=list(range(first,first+4))
        if not all((pathlib.Path('/sys/devices/system/cpu')/('cpu'+str(c))/'node3').exists() for c in block):raise RuntimeError('NUMA topology changed')
        busy=sum(b[c]-a[c]+b[c+96]-a[c+96] for c in block)/(dt*os.sysconf('SC_CLK_TCK')*4)
        options.append(dict(cpus=block,external_busy_fraction=busy))
    choice=min(options,key=lambda x:(x['external_busy_fraction'],x['cpus'][0]));CPUS=choice['cpus']
    return dict(choice=choice,options=options,rule='external idle sample before measurement, independent of measured QPS')
def sample(pid):
    out=dict(time=time.time(),cpu=0,own=None,numa=None,disks={});v=ticks()
    out['cpu']=sum(v[c]+v[c+96] for c in CPUS)
    p=pathlib.Path('/proc')/str(pid)
    try:
        f=(p/'stat').read_text().rsplit(')',1)[1].split();out['own']=int(f[11])+int(f[12])
        if pathlib.Path(os.readlink(str(p/'exe'))).name in ('pointer-search','metric-search'):
            status=dict(s.split(':',1) for s in (p/'status').read_text().splitlines() if ':' in s)
            from numa_guard import parse_maps
            policy=parse_maps((p/'numa_maps').read_text())
            if policy and not status['State'].strip().startswith('Z'):out['numa']=dict(time=out['time'],cpu_mask=status['Cpus_allowed_list'].strip(),**policy)
    except (OSError,ValueError,KeyError):pass
    for name in ('nvme2n1','nvme3n1'):
        x=list(map(int,(pathlib.Path('/sys/block')/name/'stat').read_text().split()));out['disks'][name]=dict(write_bytes=x[6]*512,read_bytes=x[2]*512)
    return out
def validate(samples,p):
    start=p['point_start_unix'];end=start+p['measured_seconds'];intervals=[]
    for a,b in zip(samples,samples[1:]):
        if a['time']<start or b['time']>end or a['own'] is None or b['own'] is None:continue
        dt=b['time']-a['time'];busy=max(0,b['cpu']-a['cpu']-(b['own']-a['own']))/(dt*os.sysconf('SC_CLK_TCK')*len(CPUS))
        writes=sum(max(0,b['disks'][d]['write_bytes']-a['disks'][d]['write_bytes']) for d in a['disks'])/dt/2**20
        intervals.append(dict(external_cpu_fraction=busy,device_write_mib_s=writes))
    live=[x['numa'] for x in samples if x['numa'] and start<=x['time']<=end]
    mask=str(CPUS[0])+'-'+str(CPUS[-1]);numa=bool(len(live)>=3 and all(x['cpu_mask']==mask and x['verified'] and not x['conflicting_policies'] for x in live))
    n=len(intervals);accepted=bool(n>=5 and numa and sum(x['external_cpu_fraction'] for x in intervals)/n<=.10 and max(x['external_cpu_fraction'] for x in intervals)<=.35 and sum(x['device_write_mib_s'] for x in intervals)/n<=8 and max(x['device_write_mib_s'] for x in intervals)<=32)
    return dict(accepted=accepted,numa_verified=numa,expected_cpu_mask=mask,intervals=intervals,numa_samples=live)

def timings(Q,specs):
    """Same frozen matched-recall points; bounded timeout is a pending point."""
    pending=set(Q.load(Q.O/'timings/pending.json')) if (Q.O/'timings/pending.json').exists() else set()
    names=list(specs)
    for rep in range(3):
      for name in names[rep:]+names[:rep]:
        path=Q.O/'analysis'/(name+'-matched.json')
        if not path.exists():continue
        m=Q.load(path)
        if not m['covered']:continue
        for L in sorted(set(m['L_bracket'])):
            label='time-'+name+'-L'+str(L)+'-rep'+str(rep+1);accepted=Q.O/'timings'/(label+'-accepted.json')
            if accepted.exists():pending.discard(label);continue
            success=False
            for attempt in range(1,4):
                out=Q.O/'results'/(label+'-attempt'+str(attempt)+'.json')
                try:observations=Q.execute(Q.command(specs[name],'report',out,[L],10),label+'-attempt'+str(attempt),specs[name].get('env'),True,240)
                except RuntimeError as exc:
                    if 'bounded timeout' not in str(exc):raise
                    Q.save(Q.O/'timings'/(label+'-guard'+str(attempt)+'.json'),dict(accepted=False,reason='bounded measurement timeout'));continue
                point=Q.load(out)[0];guard=Q.validate_window(observations,point)
                Q.save(Q.O/'timings'/(label+'-guard'+str(attempt)+'.json'),guard)
                if guard['accepted']:Q.save(accepted,dict(point=point,guard=guard,result=str(out)));success=True;break
                time.sleep(5)
            if success:pending.discard(label)
            else:pending.add(label)
    Q.save(Q.O/'timings/pending.json',sorted(pending))
