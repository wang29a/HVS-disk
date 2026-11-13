#!/usr/bin/env python3
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

ROOT=Path(__file__).resolve().parent.parent
BUILD=Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
PREVIOUS=BUILD/'experiments/layout-preexperiments-20261102'
STAR=Path('/mnt/nvme3/wz/starling-wang29a')
CPUS='0-47,96-111'
ENV=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         OMP_NUM_THREADS='32',OMP_THREAD_LIMIT='64',OMP_MAX_ACTIVE_LEVELS='1',
         OMP_NESTED='FALSE',OMP_PROC_BIND='close',OMP_PLACES='cores')
ENV['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:'+str(STAR/'release-gcc11-abi0/gnu_11.4_cxx17_64_release')+':/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH','')
for name in list(ENV):
    if name.startswith('LAYOUT_'):del ENV[name]

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n');os.replace(str(tmp),str(path))
def status(**obj):save(ROOT/'status.json',dict(updated_at=now(),**obj))
def log(s):print(now(),s,flush=True)
def digest(path):
    h=hashlib.sha256()
    with open(str(path),'rb') as f:
        while True:
            b=f.read(8<<20)
            if not b:return h.hexdigest()
            h.update(b)
def run(args,label,env=None):
    log('RUN '+label)
    with open(str(ROOT/'logs'/(label+'.log')),'w') as f:
        subprocess.run([str(a) for a in args],env=env or ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
    log('DONE '+label)

def configs():
    cs=json.loads((PREVIOUS/'analysis/inputs.json').read_text())
    for c in cs:
        for k in ['meta','topo','coord','query1','query2','alpha','gt']:
            if not Path(c[k]).is_file():raise RuntimeError('Missing input '+c[k])
        for k,h in c['input_sha256'].items():
            if digest(c[k])!=h:raise RuntimeError('Input changed '+c[k])
        for k,s in c['input_stats'].items():
            st=Path(c[k]).stat()
            if st.st_size!=s['bytes'] or st.st_mtime_ns!=s['mtime_ns']:raise RuntimeError('Source stat changed '+c[k])
    return cs

def main():
    for d in ['scripts','bin','logs','analysis','results','scratch','inputs']:(ROOT/d).mkdir(exist_ok=True)
    lock=open(str(ROOT/'queue.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not set(os.sched_getaffinity(0)).issubset(set(range(48))|set(range(96,112))):raise RuntimeError('Affinity exceeds allocation')
    status(status='running',stage='preflight')
    import pipeline
    pipeline.main(configs())
    status(status='complete',stage='complete',datasets_completed=4,new_GT=False,
           scope='Budget/reorder queries, high recall, supplied alpha GT, trace/field controls; no dynamic ANN updates.')
    log('FOUR_DATASET_BUDGET_REORDER_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status(status='failed',error=str(e));traceback.print_exc();raise
