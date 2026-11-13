#!/usr/bin/env python3
import datetime
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT=Path(__file__).resolve().parent.parent
BUILD=Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
CPUS='0-47,96-111'
ENV=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
         OMP_THREAD_LIMIT='64',OMP_PROC_BIND='close',OMP_PLACES='cores')
ENV['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH','')

def now():return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
def save(path,obj):
    path=Path(path);tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(obj,indent=2)+'\n',encoding='utf-8');os.replace(str(tmp),str(path))
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
    with open(str(ROOT/'logs'/(label+'.log')),'w') as out:
        subprocess.run(args,env=env or ENV,stdout=out,stderr=subprocess.STDOUT,check=True)
    log('DONE '+label)

def configs():
    out=[]
    for name,title in [('openimage','OpenImages'),('cc3m','CC3M'),('howto100m','HowTo100M'),('laion','Laion')]:
        prefix=BUILD/'experiments/laion/full' if name=='laion' else Path('/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32')/name/'full'
        data=Path('/mnt/nvme2/wz/disk/dataset/Laion') if name=='laion' else Path('/mnt/nvme3/wz/dataset')/dict(openimage='Openimage_disk',cc3m='CC3M_disk',howto100m='Howto100M_disk')[name]
        original=prefix/'disk_index_graph.original'
        if not original.exists():original=(prefix/'_disk.index').resolve().parent/'disk_index_graph'
        c=dict(name=name,title=title,prefix=str(prefix),meta=str((prefix/'_disk.index').resolve()),
               topo=str(original.resolve()),coord=str((prefix/'disk_index_data').resolve()),
               query1=str(data/'query_img_emb.fvecs'),query2=str(data/'query_text_emb.fvecs'),
               alpha=str(data/'base_gt_round_00/query_alpha.fvecs'),gt=str(data/'base_gt_round_00/top10_results.ivecs'))
        for key in ('meta','topo','coord','query1','query2','alpha','gt'):
            if not Path(c[key]).is_file():raise RuntimeError('Missing '+c[key])
        c['input_stats']={key:dict(bytes=Path(c[key]).stat().st_size,mtime_ns=Path(c[key]).stat().st_mtime_ns) for key in ('meta','topo','coord')}
        c['input_sha256']={key:digest(c[key]) for key in ('meta','query1','query2','alpha','gt')}
        out.append(c)
    return out

def main():
    for d in ('logs','analysis','bin','results','scratch'):(ROOT/d).mkdir(exist_ok=True)
    lock=open(str(ROOT/'queue.lock'),'w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if not set(os.sched_getaffinity(0)).issubset(set(range(48))|set(range(96,112))):raise RuntimeError('CPU affinity exceeds 64-CPU allocation')
    cs=configs();save(ROOT/'analysis/inputs.json',cs)
    save(ROOT/'analysis/protocol.json',dict(datasets=[c['name'] for c in cs],max_threads=64,
        search_threads=[1,4,32,64],cpus=CPUS,numa_memory=[0,1],blas_threads=1,
        serial=True,new_GT=False,original_ID_order=True,alpha_budget='Full',reorder=False,
        layouts=['TXY','TX_Y','TY_X','T_XY','T_X_Y'],dynamic_ANN_updates='not implemented; storage patches only'))
    status(status='running',stage='build-alpha')
    run(['clang-format','-i','-style=LLVM',str(ROOT/'scripts/alpha_scan.cpp')],'format-alpha')
    run(['g++-11','-std=c++17','-O3','-fopenmp',str(ROOT/'scripts/alpha_scan.cpp'),'-o',str(ROOT/'bin/alpha-scan')],'build-alpha')
    for c in cs:
        output=ROOT/'analysis'/(c['name']+'-alpha.json')
        if output.exists() and json.loads(output.read_text()).get('invalid_records',1)==0:continue
        status(status='running',stage='alpha-scan',dataset=c['name'])
        run([str(ROOT/'bin/alpha-scan'),c['meta'],c['topo'],c['alpha'],'16',str(output)],c['name']+'-alpha-scan')
    status(status='running',stage='awaiting-layout-implementation',alpha_datasets_completed=4)
    while not (ROOT/'LAYOUT_READY').exists():time.sleep(15)
    spec=importlib.util.spec_from_file_location('layout_runner',str(ROOT/'scripts/layout_runner.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    module.main(cs)
    status(status='complete',stage='complete',alpha_datasets_completed=4,layout_datasets_completed=4,
           note='Static queries and storage read/write probes; not a dynamic ANN update benchmark.')
    log('FOUR_DATASET_PREEXPERIMENTS_COMPLETE')

if __name__=='__main__':
    try:main()
    except Exception as e:
        status(status='failed',error=str(e));traceback.print_exc();raise
