#!/usr/bin/env python3
"""Bounded, source-only unified build; never cancels another queue."""
import fcntl, hashlib, json, os, pathlib, shutil, subprocess, time
R = pathlib.Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/u-shape-20261113')
E = R.parent
D = pathlib.Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
O = R/'stage2'
O.mkdir(exist_ok=True)
def save(stage, **kw):
    x = dict(stage=stage, pid=os.getpid(), updated_unix=time.time(), **kw)
    p=O/'build-status.json.partial'; p.write_text(json.dumps(x,indent=2)+'\n'); p.replace(O/'build-status.json')
def sha(p):
    h=hashlib.sha256()
    with open(str(p),'rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
try:
    save('waiting_global_lock')
    lock=open(str(E/'budget-reorder-20261103/queue.lock'),'a+')
    deadline=time.monotonic()+6*3600
    while True:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
        except BlockingIOError:
            if time.monotonic()>deadline: raise RuntimeError('global lock timeout after 6h')
            time.sleep(10)
    assert json.loads((D/'status.json').read_text())['stage']=='complete'
    assert json.loads((D/'gt.json').read_text())
    builder=E/'laion-six-cell-20261111/bin/build-graph'
    check=json.loads((E/'laion-six-cell-20261111/analysis/tool-validation.json').read_text())
    assert check['status']=='PASS' and sha(builder)==check['builder_sha256']
    dest=O/'graphs/unified'; dest.mkdir(parents=True,exist_ok=True)
    if (dest/'graph-audit.json').exists():
        assert json.loads((dest/'graph-audit.json').read_text())['nodes']==2000000
        save('complete_reused'); raise SystemExit(0)
    # 8 GiB incremental output budget + 2 GiB margin. Release only a measured
    # shortfall from the user's actual reservation, never fill the filesystem.
    required=10*1024**3
    free=shutil.disk_usage(str(O)).free
    if free<required:
        rp=pathlib.Path('/mnt/nvme3/wz/.space-reserve')
        with open(str(rp/'reserve.lock'),'a+') as lk:
            fcntl.flock(lk,fcntl.LOCK_EX)
            reserve=rp/'hvs-experiments-20261113.reserve'
            st=reserve.stat(); take=((required-free+1024**3-1)//1024**3)*1024**3
            assert st.st_uid==os.getuid() and st.st_size>take+2*1024**3
            with open(str(reserve),'r+b') as f: f.truncate(st.st_size-take); f.flush(); os.fsync(f.fileno())
            meta=json.loads((rp/'reservation.json').read_text()); now=reserve.stat()
            meta.update(logical_GiB=now.st_size/1024**3,allocated_GiB=now.st_blocks*512/1024**3,updated_unix=time.time())
            meta.setdefault('releases',[]).append(dict(bytes=take,purpose='U-shape unified graph',time=time.time()))
            (rp/'reservation.json').write_text(json.dumps(meta,indent=2)+'\n')
    cmd=['numactl','--all','--physcpubind=72-79','--membind=3',str(builder),'dual',str(D/'base_img_emb.fvecs'),str(D/'base_text_emb.fvecs'),'2000000','512','512','8',str(dest)]
    env=os.environ.copy(); env.update(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',LC_ALL='C')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    save('building_unified',command=cmd,builder_sha256=sha(builder),started_unix=time.time())
    with open(str(O/'unified-build.log'),'a') as log:
        result=subprocess.run(['nice','-n','10','ionice','-c','2','-n','7','timeout','--signal=TERM','--kill-after=30s','48h']+cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('builder exit '+str(result.returncode))
    audit=json.loads((dest/'graph-audit.json').read_text()); assert audit['nodes']==2000000
    save('complete',audit=audit,graph_sha256=sha(dest/'float-graph.bin'),next='pointer-engine validation and three-alpha diagnostic; specialized graphs gated')
except Exception as exc:
    save('failed',error=str(exc)); raise
