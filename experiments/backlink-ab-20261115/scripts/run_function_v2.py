"""Bounded tmux coordinator: diagnosis -> revised gates -> A/B -> DRAM -> SSD."""
import fcntl,hashlib,json,os,signal,sys,time
from pathlib import Path
import common as c
import function_compile as C
import function_check as V
import stage_checks as S
F=c.RUN/'function-v2'
QUEUE=c.RUN/'function-queue-status.json'
DIAG=c.E/'edbt-evidence-20261115/runtime/native/concurrency-diagnosis'

def status(stage,**kw):c.save(QUEUE,dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
def predecessor():
    deadline=time.monotonic()+24*3600
    while True:
        f=DIAG/'status.json';state=c.load(f) if f.exists() else {}
        stage=state.get('stage')
        if stage in ('complete','complete_with_pending','failed'):
            return stage=='complete' and (DIAG/'summary.json').exists() and not c.load(DIAG/'summary.json')['pending']
        status('waiting_concurrency_diagnosis',predecessor=state,deadline_unix=time.time()+deadline-time.monotonic())
        if time.monotonic()>deadline:raise RuntimeError('diagnostic predecessor wait exceeded24h')
        time.sleep(10)

def verify_protocol():
    b=(c.ROOT/'README.md').read_bytes()
    for name in ('FREEZE.json','FUNCTION-FREEZE.json'):
        f=c.load(c.ROOT/name)
        if hashlib.sha256(b[:f['bytes']]).hexdigest()!=f['sha256']:raise RuntimeError('frozen protocol overwritten '+name)
    for p in ('logs','commands','guards'):(c.RUN/p).mkdir(exist_ok=True)

def capture():
    root=F/'capture-graph';root.mkdir(exist_ok=True);data=c.RUN/'data-10k';out=root/'float-graph.bin';stamp=root/'identity.json'
    binary=F/'bin/build-B-capture';signature=dict(binary=c.sha(binary),data=c.sha(data/'identity.json'),sample_every=23,samples=10000,threads=1)
    if stamp.exists():
        old=c.load(stamp)
        if old['signature']!=signature or old['capture_sha256']!=c.sha(F/'capture.bin') or old['graph_sha256']!=c.sha(out):raise RuntimeError('capture identity changed')
    else:
        start=time.time();c.run([binary,'dual',data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',10000,512,512,1,root],'fn-capture-build',cpus='80',timeout=3600,extra=dict(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',HVS_FUNCTION_CAPTURE=str(F/'capture.bin')))
        c.save(stamp,dict(signature=signature,capture_sha256=c.sha(F/'capture.bin'),graph_sha256=c.sha(out),wall_seconds=time.time()-start))
    same=c.sha(out)==c.sha(c.RUN/'gate-graphs/B/float-graph.bin');c.save(F/'validation/capture-regression.json',dict(status='PASS' if same else 'FAIL',captured_graph=c.sha(out),original_B=c.sha(c.RUN/'gate-graphs/B/float-graph.bin')))
    if not same:raise RuntimeError('instrumentation changed actual B graph')

def run_replays():
    data=c.RUN/'data-10k';base=[data/'base_img_emb.fvecs',data/'base_text_emb.fvecs']
    for mode in ['B']+['author-'+str(m) for m in range(8)]:
        out=F/'results'/(mode+'.bin');args=[F/'bin'/('replay-'+mode),F/'capture.bin']+base+[out]
        S.run_once(args,'fn-replay-'+mode,out,cpus='80,81',timeout=3600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))

def stage_report():
    d=c.load(S.AB/'dram-summary.json');s=c.load(S.S/'gate.json')
    rows=['# Reverse-edge A/B DRAM stage','',
          'Function and corrected-author statistical gates passed. This file precedes the authorised SSD key-cell stage; DRAM work cannot establish SSD latency superiority.','',
          'Function gate: '+str(F/'validation/function-gate.json'),
          'Small-graph gate: '+str(S.S/'gate.json'),
          'Formal build/work summary: '+str(S.AB/'dram-summary.json'),'',
          '| Graph | build seconds | mean degree | edges | range sidecar bytes |',
          '|---|---:|---:|---:|---:|']
    for rep in (1,2):
      for mode in ('A','B'):
        name=mode+str(rep);root=S.AB/'graphs'/name;stats=c.load(root/'stats.json');build=c.load(root/'build.json');side=S.AB/'packed'/name/'ranges.bin'
        rows.append('|'+name+'|'+str(build['wall_seconds'])+'|'+str(stats['mean_degree'])+'|'+str(stats['edges'])+'|'+str(side.stat().st_size)+'|')
    rows+=['','DRAM status: '+d['status']+'. Within-version two-build variation and all three selection rules are retained in the JSON, not treated as confidence intervals.']
    (S.AB/'DRAM-RESULTS.md').write_text('\n'.join(rows)+'\n')

def main():
    verify_protocol();F.mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'function-coordinator.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    diagnostic_ok=predecessor();lock=c.lock();c.capacity(3*2**30,'independent reverse-insertion function trace and minimal author fix variants')
    status('compiling_function_adapters',diagnostic_complete=diagnostic_ok)
    proof=F/'validation/build.json'
    if not proof.exists():C.main()
    else:
        old=c.load(proof)
        for name,h in old['source_scripts'].items():
            if c.sha(c.ROOT/'scripts'/name)!=h:raise RuntimeError('compiled function source changed '+name)
    status('capturing_real_B_inputs');capture();status('function_replays');run_replays();V.main()
    # Standalone structure/export validator uses bounded memory even for2M.
    binary=F/'bin/graph-audit'
    if not binary.exists():c.run(['g++-11','-O3','-std=c++17','-fno-fast-math','-ffp-contract=off','-I'+str(c.ROOT/'scripts'),c.ROOT/'scripts/graph_audit.cpp','-o',binary],'fn-compile-graph-audit',cpus='80,81',timeout=300)
    status('small_corrected_author_statistical_gate');S.statistical_gate()
    if not diagnostic_ok:
        status('gates_passed_full_build_deferred',reason='concurrency diagnosis has unresolved/pending result; no formal2M builds',function=str(F/'validation/function-gate.json'),small=str(S.S/'gate.json'));return
    status('formal_AB_builds_and_DRAM');S.formal_dram();stage_report()
    status('DRAM_complete_SSD_key_cells',report=str(S.AB/'DRAM-RESULTS.md'));ok=S.ssd_stage()
    status('complete' if ok else 'SSD_complete_with_pending',DRAM=str(S.AB/'dram-summary.json'),SSD=str(S.AB/'ssd-summary.json'),scope='Laion2M only; no OpenImages/fullLaion rebuild')

if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();status('stopped_at_gate_or_failure',error=str(e));raise
