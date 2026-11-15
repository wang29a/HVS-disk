#!/usr/bin/env python3
"""Rebuild the affected unified graph, then gate fixed-centre before endpoints."""
import fcntl,json,os,pathlib,shutil,signal,struct,time
import stage2_queue as Q
R=Q.R;V1=Q.O;Q.O=R/'stage2-v2';O=Q.O
GRID=[10,12,15,20,40,60,80,100,120,160,200,300,400,600,800,1200,1600]
def reserve_capacity(required,purpose):
    free=shutil.disk_usage(str(O)).free
    if free>=required:return
    p=pathlib.Path('/mnt/nvme3/wz/.space-reserve')
    with (p/'reserve.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        reserve=p/'hvs-experiments-20261113.reserve';s=reserve.stat()
        take=((required-free+2**30-1)//2**30)*2**30
        if s.st_uid!=os.getuid() or s.st_size<take+2**30:raise RuntimeError('insufficient owned reservation')
        with reserve.open('r+b') as f:f.truncate(s.st_size-take);f.flush();os.fsync(f.fileno())
        m=Q.load(p/'reservation.json');s=reserve.stat()
        m.update(logical_GiB=s.st_size/2**30,allocated_GiB=s.st_blocks*512/2**30,updated_unix=time.time())
        m.setdefault('releases',[]).append(dict(bytes=take,purpose=purpose,time=time.time()))
        Q.save(p/'reservation.json',m)
def build(mode,a=None):
    folder=O/'graphs'/('unified' if mode=='unified' else 'fixed-'+a);folder.mkdir(exist_ok=True)
    proof=Q.load(O/'validation/semantic-proof.json')
    if Q.sha(O/'bin'/('build-'+mode))!=proof['binaries'][mode]['builder_sha256']:raise RuntimeError('builder changed after semantic validation')
    stamp=folder/'identity.json'
    if stamp.exists():
        old=Q.load(stamp)
        if old['builder_sha256']!=proof['binaries'][mode]['builder_sha256'] or Q.sha(folder/'float-graph.bin')!=old['graph_sha256']:raise RuntimeError('saved graph identity changed')
        return folder
    reserve_capacity(5*2**30,'pruning-v2 '+mode+' '+str(a))
    env=dict(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8')
    if a is not None:env['HVS_FIXED_ALPHA']=a
    started=time.time()
    Q.execute([O/'bin'/('build-'+mode),'dual',Q.D/'base_img_emb.fvecs',Q.D/'base_text_emb.fvecs',2000000,512,512,8,folder], 'build-'+mode+('-'+a if a else ''),env,False,48*3600,'72-79')
    audit=Q.load(folder/'graph-audit.json');assert audit['nodes']==2000000 and audit['quantized_lost_grid_points']==0
    counts=Q.load(folder/'pruning-counters.json')
    if mode=='unified' and not counts['inequalities']:raise RuntimeError('unified instrumentation not reached')
    if mode=='fixed' and not counts['fixed_calls']:raise RuntimeError('fixed direct pruning not reached')
    Q.save(stamp,dict(builder_sha256=proof['binaries'][mode]['builder_sha256'],graph_sha256=Q.sha(folder/'float-graph.bin'),wall_seconds=time.time()-started,audit=audit,pruning_counts=counts))
    return folder
def study(spec,label,entries):
    Q.GRID=sorted(set([max(10,entries)]+[x for x in GRID if x>=entries]))
    return Q.study(spec,label)
def main():
    for name in ('analysis','frozen','results','timings','commands','graphs','logs'):(O/name).mkdir(exist_ok=True)
    validation=Q.load(O/'validation/integration-proof.json')
    assert validation['status']=='PASS'
    assert validation['semantic_proof_sha256']==Q.sha(O/'validation/semantic-proof.json')
    assert Q.load(V1/'audit-control.json')['stage']=='complete'
    # No valid historical build-branch census can be inferred from a final graph.
    # The full retained-edge audit is sufficient to reject a negligible-impact
    # assumption and require this corrected unified rebuild.
    audit=Q.load(V1/'analysis/unified-final-audit-v1.json')
    assert audit['nodes']==2000000
    lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+')
    Q.status('waiting_global_lock');deadline=time.monotonic()+6*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('global lock wait >6h')
            time.sleep(10)
    reserve_capacity(12*2**30,'corrected unified and gated fixed graphs, pruning-v2')
    binary=O/'bin/pointer-search'
    if not binary.exists():binary.symlink_to(V1/'bin/pointer-search')
    if Q.sha(binary)!=Q.load(V1/'validation/pointer-parity.json')['binary_sha256']:raise RuntimeError('search binary not validated')
    for cohort,ids in Q.load(Q.D/'identity.json')['query_cohorts'].items():(O/(cohort+'-ids.txt')).write_text('\n'.join(map(str,ids))+'\n')
    unified=build('unified')
    with (unified/'full.meta').open('rb') as f:f.seek(28);entries=struct.unpack('<I',f.read(4))[0]
    uspec={};u={}
    for a in ('0.1','0.5','0.9'):
        label='unified-'+a+'-off';s=Q.spec_laion(a,unified);uspec[label]=s;u[a]=study(s,label,entries)
    ratios={a:u[a]['expanded']/u['0.5']['expanded'] for a in ('0.1','0.9')} if all(v['covered'] for v in u.values()) else {}
    passed=bool(len(ratios)==2 and all(v>=1.1 for v in ratios.values()))
    Q.save(O/'analysis/u-shape-gate.json',dict(passed=passed,endpoint_over_center=ratios,results=u,note='corrected graph; v1 observations are not mixed into this gate'))
    if not passed:
        Q.timings(uspec);Q.status('stopped_at_corrected_u_shape_gate');return
    centre=build('fixed','0.5');cs=Q.spec_laion('0.5',centre);cs['env']={'PTR_ENTRY_META':str(unified/'full.meta')}
    c=study(cs,'fixed-0.5',entries)
    ratio=c['expanded']/u['0.5']['expanded'] if c['covered'] else None
    degree=c['mean_degree']/u['0.5']['mean_degree'] if c['covered'] else None
    close=bool(ratio is not None and .8<=ratio<=1.2)
    fair=bool(degree is not None and abs(degree-1)<=.05)
    gate=dict(passed=close and fair,expansion_ratio=ratio,mean_degree_ratio=degree,centre_within_20_percent=close,edge_budget_within_5_percent=fair,centre=c,note='Deviation is a review gate, not proof of an implementation error; no automatic parameter adjustment.')
    Q.save(O/'analysis/centre-preflight-gate.json',gate)
    if not gate['passed']:
        Q.timings(dict(uspec,**{'fixed-0.5':cs}));Q.status('stopped_at_centre_review_gate',gate=gate);return
    specs={};gains={};fixed={'0.5':(cs,c)}
    for a in ('0.1','0.9'):
        graph=build('fixed',a);s=Q.spec_laion(a,graph);s['env']={'PTR_ENTRY_META':str(unified/'full.meta')}
        fixed[a]=(s,study(s,'fixed-'+a,entries))
    # All graphs are built before the main interleaved measurement phase.
    for a in ('0.1','0.5','0.9'):
        specs['unified-'+a+'-off']=uspec['unified-'+a+'-off']
        s,result=fixed[a];specs['fixed-'+a]=s
        if result['covered']:
            gain=Q.bootstrap_gain(u[a],result);gain.update(covered=True,mean_degree_ratio=result['mean_degree']/u[a]['mean_degree'])
            gain['budget_matched']=abs(gain['mean_degree_ratio']-1)<=.05;gains[a]=gain
        else:gains[a]=dict(covered=False)
        s=Q.spec_laion(a,unified,True);study(s,'unified-'+a+'-on',entries);specs['unified-'+a+'-on']=s
    Q.timings(specs)
    proceed=all(gains[a].get('covered') and gains[a]['budget_matched'] and gains[a]['gain']>=.2 and gains[a]['conditional_paired_95_ci'][0]>0 for a in ('0.1','0.9'))
    Q.save(O/'analysis/specialization-gate.json',dict(passed=proceed,results=gains))
    Q.status('dram_complete_ssd_design_ready' if proceed else 'stopped_at_specialization_gate')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,Q.stop);signal.signal(signal.SIGINT,Q.stop)
    try:main()
    except Exception as exc:Q.stop();Q.status('failed',error=str(exc));raise
