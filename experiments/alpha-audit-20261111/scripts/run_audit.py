"""Bounded offline format audit and real-library interval regression."""
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments')
ENV = dict(os.environ, LC_ALL='C', OMP_NUM_THREADS='1', OMP_THREAD_LIMIT='1',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH','')


def save(path, value):
    temp = path.with_suffix('.partial')
    temp.write_text(json.dumps(value,indent=2)+'\n')
    os.replace(str(temp),str(path))


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(8<<20),b''):
            h.update(block)
    return h.hexdigest()


def run(args, label, expected=0):
    with (ROOT/'logs'/(label+'.log')).open('w') as out:
        result = subprocess.call([str(a) for a in args],env=ENV,stdout=out,stderr=subprocess.STDOUT)
    if (expected==0 and result!=0) or (expected!=0 and result==0):
        raise RuntimeError(label+' unexpected exit '+str(result))
    return result


def status(stage, **fields):
    save(ROOT/'status.json',dict(stage=stage,updated_unix=time.time(),**fields))


def main():
    for name in ('analysis','logs','bin'):
        (ROOT/name).mkdir(exist_ok=True)
    status('auditing_serialized_graph')
    compiler = 'g++-11'
    run([compiler,'-std=c++17','-O2',ROOT/'scripts/audit_mem_disk.cpp','-o',ROOT/'bin/audit-mem-disk'],'compile-audit')
    paths=[Path('/mnt/nvme3/wz/Index/Openimage')/name for name in
           ('mem.index','_disk.index','disk_index_graph')]
    report=ROOT/'analysis/openimage-mem-disk.json'
    run([ROOT/'bin/audit-mem-disk']+paths+[report],'audit-openimage')
    save(ROOT/'analysis/input-hashes.json',{str(p):digest(p) for p in paths})
    spec=json.loads((EXPERIMENTS/'concurrency-alpha/analysis/build.json').read_text())['separated']
    if digest(spec['library'])!=spec['library_sha256']:
        raise RuntimeError('reference library changed')
    flags=spec['flags']
    link=['-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas']
    test=ROOT/'scripts/check_merge_intervals.cpp'
    old=ROOT/'bin/check-old'
    status('checking_reference_regression')
    run([compiler]+flags+[test,spec['library']]+link+['-o',old],'link-reference')
    reference_exit=run([old],'reference-regression',expected=1)
    status('building_changed_translation_units',max_compilers=2)
    sources=[ROOT/'sources/hybrid_index.cpp',ROOT/'sources/aux_utils.cpp']
    objects=[ROOT/'bin'/(p.name+'.o') for p in sources]
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        jobs=[pool.submit(run,[compiler]+flags+['-c',source,'-o',obj],source.stem+'-compile')
              for source,obj in zip(sources,objects)]
        for job in jobs:
            job.result()
    library=ROOT/'bin/libpipeann-audit.a'
    shutil.copyfile(spec['library'],str(library))
    members=subprocess.check_output(['ar','t',spec['library']],env=ENV).decode().splitlines()
    if not all(members.count(obj.name)==1 for obj in objects):
        raise RuntimeError('reference archive member names differ')
    run(['ar','r',library]+objects,'replace-archive-members')
    patched=ROOT/'bin/check-fixed'
    run([compiler]+flags+[test,library]+link+['-o',patched],'link-fixed')
    run([patched],'fixed-regression')
    save(ROOT/'analysis/build-and-regression.json',dict(status='PASS',reference_exit=reference_exit,
         changed_source_hashes={p.name:digest(p) for p in sources},reference_library_sha256=spec['library_sha256'],
         patched_library_sha256=digest(library),max_compilers=2,cpus=[72,73]))
    status('complete',mem_disk_audit='PASS',merge_regression='PASS')


if __name__=='__main__':
    try:
        main()
    except Exception as error:
        status('failed',error=str(error))
        raise
