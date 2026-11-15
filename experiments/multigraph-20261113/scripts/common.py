"""Bounded subprocesses, immutable identities and one owned queue namespace."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent.parent
E=ROOT.parent
RUN=ROOT/'runtime'
U=E/'u-shape-20261113'
D=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
ENV={k:v for k,v in os.environ.items() if not k.startswith(('PTR_','METRIC_','LAYOUT_','HVS_FIXED_'))}
ENV.update(LC_ALL='C',OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
ENV['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH','')
ENV['PYTHONPATH']=str(RUN/'python')+(':'+ENV['PYTHONPATH'] if ENV.get('PYTHONPATH') else '')
ENV['MPLCONFIGDIR']=str(RUN/'matplotlib-cache')
CHILD=None

def save(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.partial');temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');os.replace(str(temp),str(p))
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def status(stage,**extra):save(RUN/'status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**extra))
def stop(signum=None,frame=None):
    global CHILD
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
        try:CHILD.wait(10)
        except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
    CHILD=None
    if signum is not None:status('interrupted',signal=signum);raise SystemExit(128+signum)
def run(args,label,cpus='72-75',timeout=1200,extra=None):
    global CHILD
    env=dict(ENV);env.update(extra or {})
    cmd=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind='+cpus,'--membind=3']+list(map(str,args))
    save(RUN/'commands'/(label+'.json'),dict(command=cmd,extra=extra or {},start=time.time(),timeout=timeout))
    path=RUN/'logs'/(label+'.log');path.parent.mkdir(exist_ok=True)
    status('running',task=label)
    with path.open('w') as f:
        CHILD=subprocess.Popen(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        try:code=CHILD.wait(timeout=timeout)
        except subprocess.TimeoutExpired:stop();raise RuntimeError('timeout '+label)
        CHILD=None
    if code:raise RuntimeError(label+' exit '+str(code)+'; '+str(path))
    return path
def lock():
    p=(E/'budget-reorder-20261103/queue.lock').open('a+')
    deadline=time.monotonic()+24*3600;status('waiting_global_lock',deadline_unix=time.time()+24*3600)
    while True:
        try:fcntl.flock(p,fcntl.LOCK_EX|fcntl.LOCK_NB);return p
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('global lock wait exceeded 24h; deferred, no bypass')
            time.sleep(10)
def capacity(required,purpose):
    free=shutil.disk_usage(str(RUN)).free
    if free>=required:return
    directory=Path('/mnt/nvme3/wz/.space-reserve')
    with (directory/'reserve.lock').open('a+') as lk:
        fcntl.flock(lk,fcntl.LOCK_EX)
        p=directory/'hvs-experiments-20261113.reserve';s=p.stat();take=((required-free+2**30-1)//2**30)*2**30
        if s.st_uid!=os.getuid() or s.st_size<take+2**30:raise RuntimeError('insufficient owned disk reservation')
        with p.open('r+b') as f:f.truncate(s.st_size-take);f.flush();os.fsync(f.fileno())
        meta=load(directory/'reservation.json');s=p.stat();meta.update(logical_GiB=s.st_size/2**30,allocated_GiB=s.st_blocks*512/2**30,updated_unix=time.time());meta.setdefault('releases',[]).append(dict(bytes=take,purpose=purpose,time=time.time()));save(directory/'reservation.json',meta)
