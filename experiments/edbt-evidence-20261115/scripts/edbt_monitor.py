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
    out['cpu']=sum(v[c] for c in (set(CPUS)|{x+96 if x<96 else x-96 for x in CPUS}))
    p=pathlib.Path('/proc')/str(pid)
    try:
        f=(p/'stat').read_text().rsplit(')',1)[1].split();out['own']=int(f[11])+int(f[12])
        # pid is the exact child launched by the measurement harness. Observe
        # its real policy regardless of executable spelling or thread-cap suffix.
        executable=os.readlink(str(p/'exe'))
        status=dict(s.split(':',1) for s in (p/'status').read_text().splitlines() if ':' in s)
        from numa_guard import parse_maps
        policy=parse_maps((p/'numa_maps').read_text())
        if policy and not status['State'].strip().startswith('Z'):out['numa']=dict(time=out['time'],cpu_mask=status['Cpus_allowed_list'].strip(),executable=executable,**policy)
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
    mask=','.join(map(str,CPUS));
    def cpuset(text):
        result=set()
        for part in text.split(','):
            a,_,b=part.partition('-');result.update(range(int(a),int(b or a)+1))
        return result
    numa=bool(len(live)>=3 and all(cpuset(x['cpu_mask'])==set(CPUS) and x['verified'] and not x['conflicting_policies'] for x in live))
    n=len(intervals);accepted=bool(n>=5 and numa and sum(x['external_cpu_fraction'] for x in intervals)/n<=.10 and max(x['external_cpu_fraction'] for x in intervals)<=.35 and sum(x['device_write_mib_s'] for x in intervals)/n<=8 and max(x['device_write_mib_s'] for x in intervals)<=32)
    return dict(accepted=accepted,numa_verified=numa,expected_cpu_mask=mask,intervals=intervals,numa_samples=live)
