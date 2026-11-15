#!/usr/bin/env python3
"""Read-only final-graph audit, bounded and serialized with experiments."""
import fcntl,hashlib,json,os,pathlib,signal,subprocess,time
R=pathlib.Path(__file__).resolve().parent.parent;O=R/'stage2'
D=pathlib.Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
CHILD=None
def save(x):
    x.update(pid=os.getpid(),updated_unix=time.time())
    p=O/'audit-control.json.partial';p.write_text(json.dumps(x,indent=2)+'\n');p.replace(O/'audit-control.json')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def stop(signum,frame):
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid,signal.SIGTERM)
        try:CHILD.wait(10)
        except subprocess.TimeoutExpired:os.killpg(CHILD.pid,signal.SIGKILL);CHILD.wait()
    save(dict(stage='interrupted',signal=signum));raise SystemExit(128+signum)
def main():
    global CHILD
    assert json.loads((O/'validation/audit-fixture-proof.json').read_text())['status']=='PASS'
    save(dict(stage='waiting_global_lock'))
    lock=(R.parent/'budget-reorder-20261103/queue.lock').open('a+')
    end=time.monotonic()+6*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>end:raise RuntimeError('global lock wait exceeded 6 hours')
            time.sleep(10)
    graph=O/'graphs/unified/float-graph.bin'
    assert sha(graph)=='6cd0d29779c18a871453b077e2a7eccca8d0414b3b4881b4df582a6e2d70a8ee'
    out=O/'analysis/unified-final-audit-v1.json'
    args=['nice','-n','15','ionice','-c','2','-n','7','timeout','--signal=TERM','--kill-after=30s','6h','numactl','--all','--physcpubind=72-75','--membind=3',str(O/'bin/audit-unified'),str(graph),str(D/'base_img_emb.fvecs'),str(D/'base_text_emb.fvecs'),str(out),'4']
    env=dict(os.environ,LC_ALL='C',OMP_NUM_THREADS='4',OMP_THREAD_LIMIT='4',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    plan=dict(stage='auditing',command=args,binary_sha256=sha(O/'bin/audit-unified'),scope='final retained-edge pairs only; not historical build branch counts',identity_sha256=sha(D/'identity.json'),started_unix=time.time())
    (O/'analysis/unified-audit-plan.json').write_text(json.dumps(plan,indent=2)+'\n');save(plan)
    with (O/'logs/unified-audit.log').open('w') as log:
        CHILD=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        code=CHILD.wait();CHILD=None
    if code:raise RuntimeError('audit exit '+str(code))
    result=json.loads(out.read_text());assert result['nodes']==2000000
    save(dict(stage='complete',result=str(out),susceptible_edges=result['susceptible_edges'],scope=plan['scope'],historical_validity='not certified by final-graph audit alone'))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except Exception as e:save(dict(stage='failed',error=str(e)));raise
