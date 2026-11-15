#!/usr/bin/env python3
"""Own one queue lock; concurrent build/work diagnostics, serial timing."""
import fcntl,json,os,pathlib,signal,struct,subprocess,time
import stage2_queue as Q
R=Q.R;V1=R/'stage2';V2=R/'stage2-v2';Q.O=R/'review-20261113';O=Q.O
BUILDER=None
BASE_EXECUTE=Q.execute
def status(stage,**kw):Q.save(O/'status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),builder_pid=BUILDER.pid if BUILDER and BUILDER.poll() is None else None,**kw))
Q.status=status
def stop(signum=None,frame=None):
    global BUILDER
    if BUILDER and BUILDER.poll() is None:
        os.killpg(BUILDER.pid,signal.SIGTERM)
        try:BUILDER.wait(10)
        except subprocess.TimeoutExpired:os.killpg(BUILDER.pid,signal.SIGKILL);BUILDER.wait()
    Q.stop()
    if signum is not None:status('interrupted',signal=signum);raise SystemExit(128+signum)
def main():
    global BUILDER
    for d in ('analysis','logs','graphs','results','frozen','timings','commands','bin'):(O/d).mkdir(parents=True,exist_ok=True)
    Q.save(O/'protocol-identity.json',dict(sha256=Q.sha(R/'NEXT-CHECKS-PROTOCOL.md'),recorded_unix=time.time()))
    lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+');deadline=time.monotonic()+6*3600
    status('waiting_global_lock')
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('lock wait >6h')
            time.sleep(10)
    assert Q.load(V2/'validation/integration-proof.json')['status']=='PASS'
    binary=O/'bin/pointer-search'
    if not binary.exists():binary.symlink_to(V1/'bin/pointer-search')
    assert Q.sha(binary)==Q.load(V1/'validation/pointer-parity.json')['binary_sha256']
    for cohort,ids in Q.load(Q.D/'identity.json')['query_cohorts'].items():(O/(cohort+'-ids.txt')).write_text('\n'.join(map(str,ids))+'\n')
    graph=O/'graphs/openimage';graph.mkdir(exist_ok=True)
    builder=V2/'bin/build-unified';assert Q.sha(builder)==Q.load(V2/'validation/semantic-proof.json')['binaries']['unified']['builder_sha256']
    data=pathlib.Path('/mnt/nvme3/wz/dataset/Openimage_disk')
    cmd=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind=72-79','--membind=3',str(builder),'dual',str(data/'base_img_emb.fvecs'),str(data/'base_text_emb.fvecs'),'456699','768','768','8',str(graph)]
    started=time.time()
    if not (graph/'identity.json').exists():
        env=dict(Q.ENV,HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8')
        Q.save(O/'analysis/openimage-build-command.json',dict(command=cmd,builder_sha256=Q.sha(builder),started_unix=started,concurrent_work_diagnostics=True))
        log=(O/'logs/build-openimage.log').open('w');BUILDER=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        status('building_openimage',command=cmd)
    # These runs supply deterministic work only; their latency is discarded.
    def diagnostic_execute(cmd,label,extra=None,timed=False,timeout=1200,cpus='72-75'):
        return BASE_EXECUTE(cmd,label,extra,timed,timeout,'80-83' if not timed else cpus)
    Q.execute=diagnostic_execute
    u=Q.spec_laion('0.5',V2/'graphs/unified',True)
    s=Q.spec_laion('0.5',V2/'graphs/fixed-0.5');s['env']={'PTR_ENTRY_META':str(V2/'graphs/unified/full.meta')}
    Q.GRID=[20,40,60,80,100,120,160,200,300,400,600,800,1200,1600]
    um=Q.study(u,'centre-U1')
    saved=Q.load(V2/'frozen/fixed-0.5.json');grid=saved['report_L']
    Q.save(O/'frozen/centre-S.json',dict(source=str(V2/'frozen/fixed-0.5.json'),source_sha256=Q.sha(V2/'frozen/fixed-0.5.json'),report_L=grid,selection='reused v2 training-only frozen configurations'))
    sm=Q.matched(Q.diagnostic(s,'report',grid,'centre-S-report-work'));Q.save(O/'analysis/centre-S-matched.json',sm)
    compare=O/'bin/compare-float-graphs';assert compare.exists()
    out=O/'analysis/centre-active-edges.json'
    Q.execute([compare,'--active',V2/'graphs/unified/float-graph.bin',V2/'graphs/fixed-0.5/float-graph.bin',out,.5],'compare-centre-activity')
    edge=Q.load(out)
    er=sm['expanded']/um['expanded'] if um['covered'] and sm['covered'] else None
    ar=edge['new_mean_degree']/edge['old_mean_degree']
    cg=dict(passed=bool(er is not None and .8<=er<=1.2 and .8<=ar<=1.2),expansion_ratio_S_over_U1=er,active_degree_ratio_S_over_U1=ar,unified_on=um,specialized=sm,activity=edge,note='Full-graph closeness is diagnostic; controlled same-pool predicate equivalence is a separate test.')
    Q.save(O/'analysis/centre-gate.json',cg)
    # Fixed-L traces distinguish graph/search trajectory from matched-recall tuning.
    for label,spec in [('U1',u),('S',s)]:
        trace=dict(spec);trace['env']=dict(spec.get('env',{}),PTR_TRACE=str(O/'analysis'/('centre-'+label+'-trace')))
        Q.diagnostic(trace,'report',[60],'centre-'+label+'-L60-trace')
    Q.save(O/'centre-status.json',dict(stage='work_complete',gate_passed=cg['passed']))
    Q.execute=BASE_EXECUTE
    if BUILDER:
        status('waiting_openimage_build')
        while BUILDER.poll() is None:
            if time.time()-started>7200:raise RuntimeError('OpenImages build timeout 2h')
            time.sleep(5)
        code=BUILDER.returncode;BUILDER=None;log.close()
        if code:raise RuntimeError('OpenImages builder exit '+str(code))
        audit=Q.load(graph/'graph-audit.json');assert audit['nodes']==456699 and audit['quantized_lost_grid_points']==0
        Q.save(graph/'identity.json',dict(builder_sha256=Q.sha(builder),graph_sha256=Q.sha(graph/'float-graph.bin'),wall_seconds=time.time()-started,graph=audit))
    old=Q.E/'metric-consistent-20261111/scratch/dual'
    Q.execute([compare,old/'float-graph.bin',graph/'float-graph.bin',O/'analysis/openimage-edge-comparison.json'],'compare-openimage')
    diff=Q.load(O/'analysis/openimage-edge-comparison.json')
    degree_change=abs(diff['new_mean_degree']/diff['old_mean_degree']-1)
    coverage=(diff['common_added_width']+diff['common_removed_width'])/diff['old_total_width']
    def meta(path):
        b=path.read_bytes();ne=struct.unpack_from('<I',b,28)[0];return dict(slots=struct.unpack_from('<I',b,16)[0],entries=list(struct.unpack_from('<'+'I'*ne,b,32)))
    om,nm=meta(old/'full.meta'),meta(graph/'full.meta')
    material=diff['edge_symmetric_difference_over_union']>.01 or degree_change>.01 or coverage>.01 or om['slots']!=nm['slots']
    Q.save(O/'analysis/openimage-change-gate.json',dict(material=material,degree_relative_change=degree_change,common_coverage_change_over_old_width=coverage,old_meta=om,new_meta=nm,diff=diff))
    Q.timings({'centre-U1':u,'centre-S':s})
    Q.save(O/'centre-status.json',dict(stage='complete',gate_passed=cg['passed']))
    # The following module owns conditional SSD rechecks and no new GT.
    import ssd_priority_review
    ssd_priority_review.run(Q,old,graph,material)
    if not cg['passed']:
        status('stopped_before_strength_sweep',reason='centre U1/S closeness gate requires review',openimage_ssd='see SSD status');return
    status('priority_checks_complete_strength_protocol_ready',note='density pilot and measured-page pipeline require their declared resource preparation; no curve is claimed yet')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:main()
    except Exception as e:stop();status('failed',error=str(e));raise
