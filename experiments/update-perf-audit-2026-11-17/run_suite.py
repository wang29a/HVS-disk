"""Serialized same-node/same-device profiling and upstream OdinANN reference."""
from pathlib import Path
import ctypes
import fcntl
import hashlib
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import time
import numpy as np
from prepare_profile import prepare,prepare_no_stats,FILES

ROOT=Path(__file__).resolve().parent
E=Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments')
OLD=E/'update-basic-2026-11-16'
sys.path.insert(0,str(OLD/'implementation'))
import run_g0 as c
import run_g1
from gt_unreachable import matrix
OUT=ROOT/'runtime'
COLD=Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1')
MG=E/'multigraph-20261113/runtime/data'
DATA=Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
UPSTREAM=ROOT/'upstream-odinann-0.1.1'
UPSTREAM_SHA='7840ce96ac74a461b8efe2739b6aed5b14c15b9a'


def sha(p):return c.sha(p)


def save(p,v):
    p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(v,indent=2)+'\n');q.replace(p)


def status(stage,**kw):save(OUT/'status.json',dict(stage=stage,updated_unix=time.time(),**kw))


def run(args,label,folder=None,threads=4,extra=None,timeout=7200):
    folder=Path(folder or OUT);folder.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in c.ENV.items() if not k.startswith(('HVS_','LAYOUT_','METRIC_','PTR_'))}
    env.update(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    if extra:env.update(extra)
    cpus='72-87' if threads==16 else '72-75'
    command=['nice','-n','15','ionice','-c','2','-n','7','numactl','--physcpubind='+cpus,'--membind=3']+list(map(str,args))
    start=time.time()
    with (folder/(label+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env)
        status(label,pid=child.pid,command=command)
        try:rc=child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    save(folder/(label+'.command.json'),dict(command=command,returncode=rc,wall_s=time.time()-start,
        environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','LD_LIBRARY_PATH')},
        additional_definitions=env.get('ADDITIONAL_DEFINITIONS','')))
    if rc:raise RuntimeError(label+' failed: '+str(rc))


def prefix(src,dest,size):
    assert not dest.exists()
    with Path(src).open('rb') as f,Path(dest).open('xb') as g:
        remaining=size
        while remaining:
            b=f.read(min(8*2**20,remaining));assert b;g.write(b);remaining-=len(b)
        g.flush();os.fsync(g.fileno())


def enough_space():
    free=shutil.disk_usage(str(ROOT)).free;target=40*2**30
    if free>=target:return
    base=Path('/mnt/nvme3/wz/.space-reserve');lock=(base/'reserve.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    meta=base/'reservation.json';v=json.loads(meta.read_text());p=Path(v['path']);s=p.lstat()
    import stat
    assert stat.S_ISREG(s.st_mode) and s.st_uid==os.getuid()==v['owner_uid']
    assert 'contains no experiment data' in v['purpose'] and s.st_size==int(v['logical_GiB']*2**30)
    release=(int(math.ceil((target-free)/2**30))+4)*2**30;assert release<=64*2**30 and s.st_size>release
    with p.open('r+b') as f:f.truncate(s.st_size-release);f.flush();os.fsync(f.fileno())
    event=dict(bytes=release,time=time.time(),purpose='isolated1000-insert profiling and native OdinANN benchmark; data-free HVS reservation')
    v['logical_GiB']=p.stat().st_size/2**30;v['allocated_GiB']=p.stat().st_blocks*512/2**30;v['updated_unix']=time.time();v['releases'].append(event);save(meta,v)
    save(OUT/'reservation-release.json',dict(event=event,remaining_GiB=v['logical_GiB']))


def fixture(label):
    root=OUT/'hot'/label;root.mkdir(parents=True,exist_ok=False)
    save(root/'owner.json',dict(owner=str(ROOT),purpose='disposable benchmark fixture'))
    src=COLD/'initial-1800k/packed'
    shutil.copyfile(src/'compact.meta',root/'index.meta')
    prefix(src/'compact.topo',root/'graph.pages',(src/'compact.topo').stat().st_size)
    prefix(MG/'xy.pages',root/'coordinates.pages',1800000*4096)
    for name in ('centers.bin','insert-x.bin','insert-y.bin'):(root/name).symlink_to(COLD/'inputs'/name)
    for space in ('emb','loc'):
        for suffix in ('_pq_pivots.bin','_pq_compressed.bin'):(root/(space+suffix)).symlink_to(OUT/'pq-prefix'/(space+suffix))
    for source,dest in [('img','x'),('text','y')]:(root/('query-'+dest+'.bin')).symlink_to(DATA/('query_'+source+'_emb.fvecs'))
    (root/'report-ids.txt').symlink_to(MG/'report-ids.txt')
    assert (root/'graph.pages').stat().st_dev==(root/'coordinates.pages').stat().st_dev==ROOT.stat().st_dev
    return root


def prepare_prefix():
    out=OUT/'pq-prefix';out.mkdir()
    for space in ('emb','loc'):
        (out/(space+'_pq_pivots.bin')).symlink_to(MG/(space+'_pq_pivots.bin'))
        codes=np.asarray(matrix(MG/(space+'_pq_compressed.bin'),'u1')[:1800000])
        with (out/(space+'_pq_compressed.bin')).open('wb') as f:f.write(struct.pack('<II',*codes.shape));codes.tofile(f)
    save(out/'identity.json',{p.name:sha(p) for p in out.iterdir() if p.is_file()})


def control_fixture(label):
    previous=c.RUN;c.RUN=OUT/'controls';c.RUN.mkdir(exist_ok=True)
    try:root=run_g1.fixture(label)
    finally:c.RUN=previous
    for source,dest in [('img','x'),('text','y')]:(root/('query-'+dest+'.bin')).symlink_to(DATA/('query_'+source+'_emb.fvecs'))
    (root/'report-ids.txt').symlink_to(MG/'report-ids.txt')
    for name in ('graph.pages','coordinates.pages'):
        with (root/name).open('r+b') as f:os.fsync(f.fileno())
    return root


def compile_dual():
    gate=json.loads((OLD/'runtime/g1-in-degree-protection-v2/build-identity.json').read_text())
    base=OUT/'base-source';base.mkdir()
    for name in FILES:
        p=OLD/'implementation'/name
        if name in gate['sources']:assert sha(p)==gate['sources'][name]
        shutil.copyfile(p,base/name)
    prof=OUT/'profile-source';prepare(base,prof,gate)
    nostats=OUT/'nostats-source';prepare_no_stats(base,nostats,gate)
    link=[c.BUILD/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread']
    binaries={}
    for label,source,extra in [('base',base,[]),('nostats',nostats,[]),('profile',prof,['-DHVS_PROFILE_PHASES'])]:
        # Put the frozen copy before the historical experiment include path.
        flags=['-I'+str(source)]+gate['flags']+extra
        run(['clang-format','-style=LLVM','-i',ROOT/'profile_driver.cpp'],'format-driver')
        if label in ('profile','nostats'):
            run(['clang-format','-style=LLVM','-i',source/'update_store.cpp',source/'update_store.h',source/'dual_update.cpp'],'format-'+label+'-source')
            manifest=source/('instrumentation-identity.json' if label=='profile' else 'control-identity.json')
            proof=json.loads(manifest.read_text());proof['formatted_sources']={name:sha(source/name) for name in FILES};save(manifest,proof)
        units=[source/name for name in ('update_store.cpp','deg_prune.cpp','dual_update.cpp')]+[ROOT/'profile_driver.cpp']
        binary=OUT/('driver-'+label)
        command=['g++-11']+flags+units+link+['-o',binary];run(command,'compile-'+label,timeout=900)
        save(OUT/('build-'+label+'.json'),dict(command=list(map(str,command)),sources={str(p):sha(p) for p in units},binary=sha(binary),upstream_library=sha(c.BUILD/'src/libpipeann.a')))
        binaries[label]=binary
    # Serial original/profiled equivalence: exact graph bytes and actual PQ
    # snapshot must match, while the counters themselves remain thread-private.
    run([binaries['base'],'--clock-cost'],'clock-cost')
    save(OUT/'clock-cost.json',json.loads((OUT/'clock-cost.log').read_text()))
    controls=[]
    for mode in ('pq','exact'):
        digests=[]
        for kind in ('base','nostats','profile'):
            root=control_fixture(mode+'-'+kind);out=OUT/'control-results'/(mode+'-'+kind);out.mkdir(parents=True)
            run([binaries[kind],root,mode,1,0,'e1',100,out,1],'serial-control',out)
            digests.append(dict(graph=sha(root/'graph.pages'),coordinates=sha(root/'coordinates.pages'),snapshot=sha(out/'normal-exit.snapshot')))
        assert all(v==digests[0] for v in digests),mode+' observer/control changed graph/state'
        controls.append(dict(mode=mode,status='BITWISE_IDENTICAL',digests=digests))
    save(OUT/'observer-controls.json',controls)
    return binaries


def dual_measurements(binaries):
    rows=[]
    # Each measured run inserts1000 fresh vectors into the same1.8M initialB.
    # Rotating the order reduces systematic first/last-position drift.
    kinds=[('lean','base','lean'),('logged','base','e1'),('nostats','nostats','lean'),('profile','profile','e1')]
    for rep in range(3):
        for mode in ('pq','exact'):
            order=kinds[rep:]+kinds[:rep]
            for label,kind,telemetry in order:
                name='%s-mixed-%s-r%d'%(mode,label,rep+1);out=OUT/'measurements'/name;out.mkdir(parents=True)
                root=fixture(name);run([binaries[kind],root,mode,4,4,telemetry,1000,out,0],'insert1000',out)
                result=json.loads((out/'result.json').read_text())
                row=dict(name=name,rep=rep+1,mode=mode,query_threads=4,kind=label,result=result,output=str(out),graph_sha256=sha(root/'graph.pages'))
                save(out/'identity.json',row);rows.append(row);save(OUT/'measurements.json',rows)
                assert json.loads((root/'owner.json').read_text())['owner']==str(ROOT);shutil.rmtree(str(root))
        for mode in ('pq','exact'):
            name='%s-pure-lean-r%d'%(mode,rep+1);out=OUT/'measurements'/name;out.mkdir(parents=True);root=fixture(name)
            run([binaries['base'],root,mode,4,0,'lean',1000,out,0],'insert1000',out)
            row=dict(name=name,rep=rep+1,mode=mode,query_threads=0,kind='lean',result=json.loads((out/'result.json').read_text()),output=str(out),graph_sha256=sha(root/'graph.pages'))
            save(out/'identity.json',row);rows.append(row);save(OUT/'measurements.json',rows);shutil.rmtree(str(root))
    return rows


def build_odin():
    # The pinned AE source is kept unmodified; all build outputs are separate.
    source_files=[p for p in UPSTREAM.rglob('*') if p.suffix in ('.h','.cpp') and p.is_file()]
    source_hashes={str(p.relative_to(UPSTREAM)):sha(p) for p in source_files}
    libc=ctypes.CDLL(None,use_errno=True);params=ctypes.create_string_buffer(256)
    fd=libc.syscall(425,2,ctypes.byref(params));errno=ctypes.get_errno()
    if fd>=0:os.close(fd)
    save(OUT/'kernel-io.json',dict(kernel=os.uname().release,io_uring_setup_result=fd,errno=errno,
        selected_backend='Linux AIO',reason='same backend as current executor; native supported USE_AIO option'))
    build=OUT/'odin-build'
    definitions='-DBG_IO_THREAD -DDELTA_PRUNING'
    run(['cmake','-S',UPSTREAM,'-B',build,'-DUSE_AIO=ON','-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CXX_COMPILER=g++-11','-DCMAKE_C_COMPILER=gcc-11','-DCMAKE_POLICY_VERSION_MINIMUM=3.5',
        '-DBLAS_LIBRARIES=/mnt/nvme2/wz/disk/local/openblas/lib/libopenblas.so',
        '-DCMAKE_CXX_FLAGS=-I/mnt/nvme2/wz/disk/local/openblas/include -fno-fast-math -ffp-contract=off'],
        'configure-odin',extra=dict(ADDITIONAL_DEFINITIONS=definitions),timeout=900)
    run(['cmake','--build',build,'-j16','--target','build_disk_index'],'build-odin',threads=16,extra=dict(ADDITIONAL_DEFINITIONS=definitions),timeout=3600)
    for path,digest in source_hashes.items():assert sha(UPSTREAM/path)==digest,'upstream source changed'
    flags=['-std=gnu++17','-O3','-march=native','-fopenmp','-DUSE_AIO','-DUSE_AVX512','-DBG_IO_THREAD','-DDELTA_PRUNING','-DOVERLAP_INIT','-DDYN_PIPE_WIDTH','-DNDEBUG',
        '-I'+str(UPSTREAM/'include'),'-I'+str(UPSTREAM/'include/tsl/include'),'-I'+str(UPSTREAM/'third_party/liburing/src/include'),'-I/mnt/nvme2/wz/disk/local/openblas/include','-fno-fast-math','-ffp-contract=off']
    binary=OUT/'odin-driver'
    run(['clang-format','-style=LLVM','-i',ROOT/'odin_driver.cpp'],'format-odin-harness')
    run(['g++-11']+flags+[ROOT/'odin_driver.cpp',build/'src/libpipeann.a','-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',binary],'compile-odin-driver',timeout=900)
    save(OUT/'upstream-identity.json',dict(repository='https://github.com/thustorage/PipeANN',tag='0.1.1',commit=UPSTREAM_SHA,
        source_files=source_hashes,unmodified=True,definitions=definitions,backend='Linux AIO',sector_bytes=4096,
        library_sha256=sha(build/'src/libpipeann.a'),harness_sha256=sha(ROOT/'odin_driver.cpp'),binary_sha256=sha(binary)))
    index=OUT/'odin-index';index.mkdir();prefix_path=index/'base'
    # floor(B*GiB/N)=64, retaining margin from a floating-point boundary.
    budget=(64.05*1800000)/(2**30)
    run([build/'tests/build_disk_index','float',COLD/'inputs/base-img-1800k.bin',prefix_path,40,200,str(budget),64,16,'l2',1],
        'build-native1800k',folder=index,threads=16,timeout=14400)
    codes=matrix(Path(str(prefix_path)+'_pq_compressed.bin'),'u1');assert codes.shape==(1800000,64)
    assert Path(str(prefix_path)+'_disk.index').stat().st_dev==ROOT.stat().st_dev
    for p in index.iterdir():
        if p.is_file():
            with p.open('r+b') as f:os.fsync(f.fileno())
    save(index/'identity.json',dict(initial=1800000,dimension=512,R=40,L=200,PQ_chunks=64,normalization='none',
        data_sha256=sha(COLD/'inputs/base-img-1800k.bin'),files={p.name:sha(p) for p in index.iterdir() if p.is_file()},
        note='native Vamana single-space graph/codebooks, not the dual-spaceB graph; throughput reference only'))
    return binary,prefix_path


def odin_measurements(binary,prefix_path):
    rows=[]
    for rep in (1,2,3):
        for readers in (0,):
            name='odin-q%d-r%d'%(readers,rep);out=OUT/'measurements'/name;out.mkdir(parents=True)
            run([binary,prefix_path,COLD/'inputs/insert-x.bin',DATA/'query_img_emb.fvecs',4,readers,1000,out],
                'insert1000',out,timeout=1800)
            row=dict(name=name,rep=rep,query_threads=readers,result=json.loads((out/'result.json').read_text()),output=str(out))
            shadow=Path(str(prefix_path)+'_shadow_disk.index');assert shadow.exists()
            row['shadow_sha256']=sha(shadow);save(out/'identity.json',row);rows.append(row);save(OUT/'odin-measurements.json',rows)
            # Only native default shadow copies owned by this reference are
            # overwritten/removed; the validated initial index remains intact.
            for p in prefix_path.parent.glob(prefix_path.name+'_shadow*'):
                if p.is_file() or p.is_symlink():p.unlink()
    return rows


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'suite-identity.json').exists():raise RuntimeError('existing suite: refuse unverified resume')
    lock=(E/'budget-reorder-20261103/queue.lock').open('a+');status('WAITING_GLOBAL_LOCK');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        paused=json.loads((OUT/'pause-monitor.json').read_text());assert paused['stage']=='PAUSED_AFTER_EXACT_R1_USER_REQUEST'
        enough_space()
        save(OUT/'suite-identity.json',dict(initial=1800000,insertions_each=1000,update_threads=4,
            mixed_query_threads=4,pure_query_threads=0,repetitions=3,NUMA=3,measurement_cpus='72-75',builder_cpus='72-87',
            source_commit='2fa563f9e0717b4b3b526c190a58943e890b85ec',topology_device='nvme3',coordinate_device='nvme3',
            timed='thread startup, insert calls and optional E1 in-memory sampling/progress output',
            excluded='fixture copy/fsync, load, structural audit, GT, JSON serialization and file writes, snapshots, final fsync',
            overlap='I/O and locks are overlays; full reprune is nested inside reverse updates; phase percentages use summed insert latency, not batch wall'))
        prepare_prefix();binaries=compile_dual();dual_measurements(binaries)
        status('DUAL_PROFILES_COMPLETE');run(['python3',ROOT/'analyse.py'],'analyse-dual')
        native,prefix_path=build_odin();odin_measurements(native,prefix_path)
        run(['python3',ROOT/'analyse.py'],'analyse-complete');status('PERF_AUDIT_COMPLETE')
    except Exception as error:
        status('STOPPED_AT_PERF_AUDIT',reason=str(error));raise


if __name__=='__main__':main()
