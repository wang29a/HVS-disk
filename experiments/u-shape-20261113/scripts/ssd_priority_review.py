"""Conditional OpenImages SSD check using existing mixed-alpha GT buckets."""
import json,os,pathlib,statistics,struct,time
import numpy as np
GRID=[20,40,60,80,100,120,160,200,300,400,600,800]
DEPTHS=[10,20,40,80,160,320,None]
TARGETS=[90,92,94,95,96,97,98,99]
DATA=pathlib.Path('/mnt/nvme3/wz/dataset/Openimage_disk')
COORD=pathlib.Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
PQ='/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/'
def matrix(p,dtype=np.float32):
    with p.open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    if p.stat().st_size!=8+n*d*4:raise RuntimeError('matrix size '+str(p))
    return np.memmap(str(p),mode='r',dtype=dtype,offset=8,shape=(n,d))
def write(p,a):
    with p.open('wb') as f:f.write(struct.pack('<II',*a.shape));a.tofile(f)
def choose(points):
    """Training-only deterministic pages/compute representatives, no latency."""
    rules={}
    for name,key in [('pages',lambda p:(p['submitted_pages'],p['pq_space_evals'],p['L'],p['rerank_limit'])),('compute',lambda p:(p['pq_space_evals'],p['submitted_pages'],p['L'],p['rerank_limit']))]:
        selected=[]
        for target in TARGETS:
            candidates=[p for p in points if p['recall']>=target]
            if candidates:
                p=min(candidates,key=key);item=dict(L=p['L'],r=p['rerank_limit'],train_recall=p['recall'],train_pages=p['submitted_pages'],target=target)
                if not any((q['L'],q['r'])==(item['L'],item['r']) for q in selected):selected.append(item)
        rules[name]=selected
    return rules
def bracket(points,target=95.):
    # Preserve the training-selected order; do not choose a fast or cheap
    # held-out configuration when recall happens to fluctuate.
    for p in points:
        if p['recall']==target:return p,p,0.
    for a,b in zip(points,points[1:]):
        if a['recall']<target<b['recall']:return a,b,(target-a['recall'])/(b['recall']-a['recall'])
    return None
def run(Q,old,new,material):
    O=Q.O;root=O/'ssd';root.mkdir(exist_ok=True)
    Q.save(root/'status.json',dict(stage='preparing',material_graph_difference=material))
    tool=Q.E/'laion-six-cell-20261111/bin/full-layout'
    if not (new/'full.topo').exists():Q.execute([tool,'write',new/'float-graph.bin',new/'full.meta',new/'full.topo',root/'new-layout-write.json'],'materialize-openimage-topology',{},False,1200)
    if not (root/'new-layout-check.json').exists():Q.execute([tool,'check',new/'float-graph.bin',new/'full.meta',new/'full.topo',root/'new-layout-check.json'],'check-openimage-topology',{},False,1200)
    assert Q.load(root/'new-layout-check.json')['status']=='PASS'
    entries=old.joinpath('full.meta').read_bytes();ne=struct.unpack_from('<I',entries,28)[0];entrybytes=entries[32:32+4*ne]
    sources={}
    for version,g in [('old',old),('new',new)]:
        d=root/version;d.mkdir(exist_ok=True)
        metadata=bytearray((g/'full.meta').read_bytes());metadata[28:]=bytes(len(metadata)-28)
        struct.pack_into('<I',metadata,28,ne);metadata[32:32+4*ne]=entrybytes;(d/'query.meta').write_bytes(metadata)
        for name,target in [('group-1.bin',g/'full.topo'),('group-6.bin',COORD)]:
            p=d/name
            if not p.exists():p.symlink_to(target)
            if p.resolve()!=target.resolve():raise RuntimeError('unexpected layout target')
        sources[version]=dict(meta=d/'query.meta',topo=g/'full.topo',folder=d)
    raw={'qx':matrix(DATA/'query_img_emb.fvecs'),'qy':matrix(DATA/'query_text_emb.fvecs'),'alpha':matrix(DATA/'base_gt_round_00/query_alpha.fvecs'),'gt':matrix(DATA/'base_gt_round_00/top10_results.ivecs',np.uint32)}
    split=Q.load(Q.E/'metric-consistent-20261111/analysis/tuning-split-v2.json');workloads={}
    for name,lo,hi in [('side',.05,.15),('centre',.45,.55)]:
        w=root/name;w.mkdir(exist_ok=True);files={};identity={}
        for cohort in ('tune','report'):
            ids=[q for q in split['cohorts'][cohort]['query_ids'] if lo<=float(raw['alpha'][q,0])<hi]
            if len(ids)<20:raise RuntimeError('insufficient existing GT bucket '+name+' '+cohort)
            folder=w/cohort;folder.mkdir(exist_ok=True);files[cohort]={}
            for key,a in raw.items():
                p=folder/(key+'.bin');write(p,np.asarray(a[ids]));files[cohort][key]=p
            identity[cohort]=dict(query_ids=ids,count=len(ids))
        Q.save(w/'identity.json',dict(alpha_range=[lo,hi],fixed_alpha=False,cohorts=identity,source_split_sha256=Q.sha(Q.E/'metric-consistent-20261111/analysis/tuning-split-v2.json'),unchanged_query_weights=True))
        workloads[name]=files
    binary=Q.E/'metric-consistent-20261111/bin/metric-search'
    assert Q.sha(binary)==Q.load(Q.E/'metric-consistent-20261111/analysis/build.json')['binaries']['metric-search']
    def env(filtering,r,dram,diagnostic):
        e=dict(METRIC_RERANK=str(r)) if r is not None else {}
        if not filtering:e['LAYOUT_BYPASS_ALPHA']='1'
        if dram:e['LAYOUT_DRAM']='1'
        if diagnostic:e.update(LAYOUT_ONE_PASS='1',LAYOUT_GT_DIAGNOSTIC='1')
        return e
    def args(version,bucket,cohort,out,Ls):
        s=sources[version];f=workloads[bucket][cohort]
        return [binary,'query',s['meta'],s['topo'],COORD,s['folder'],'T_XY',out,PQ,f['qx'],f['qy'],f['alpha'],f['gt'],4,10]+list(Ls)
    def diagnostic(version,bucket,filtering,r,Ls,cohort,label):
        out=O/'results'/(label+'.json')
        if not out.exists():Q.execute(args(version,bucket,cohort,out,Ls),label,env(filtering,r,True,True),False,1200)
        return Q.load(out)
    arms={};plans={}
    historical=Q.load(Q.E/'metric-consistent-20261111/analysis/frozen-full-T_XY-pq.json') if not material else None
    for version in ('old','new'):
      for bucket in ('side','centre'):
       for filtering in (False,True):
        label='ssd-'+version+'-'+bucket+'-f'+str(int(filtering));arm=dict(version=version,bucket=bucket,filtering=filtering)
        frozen=root/(label+'-frozen.json')
        if frozen.exists():rules=Q.load(frozen)['rules']
        elif material:
            points=[]
            for r in DEPTHS:
                Ls=[L for L in GRID if r is None or r<=L]
                points+=diagnostic(version,bucket,filtering,r,Ls,'tune',label+'-tune-r'+str(r))
            rules=choose(points)
            Q.save(frozen,dict(rules=rules,selection='training-only minimum submitted pages / minimum PQ evaluations at prespecified recall bands; no latency selection',tune_points=points,created_unix=time.time()))
        else:
            candidates=[p for p in historical['selected'] if p['filtering']==filtering]
            candidates=sorted(candidates,key=lambda p:(abs(p['recall']-95),p['pages'],p['L'],p['r']))[:3]
            selected=[dict(L=p['L'],r=p['r'],train_recall=p['recall'],target=p['recall']) for p in sorted(candidates,key=lambda p:p['recall'])]
            rules={'historical-probe':selected};Q.save(frozen,dict(rules=rules,selection='paired old/new probes from historical training freeze, never held-out tuning'))
        unique=sorted(set((p['L'],p['r']) for ps in rules.values() for p in ps));reported={}
        for r in sorted(set(p[1] for p in unique)):
            Ls=[L for L,rr in unique if rr==r]
            for p in diagnostic(version,bucket,filtering,r,Ls,'report',label+'-report-r'+str(r)):reported[(p['L'],p['rerank_limit'])]=p
        arm['rules']=rules;arm['reported']=reported;arms[label]=arm;plans[label]={}
        for rule,ps in rules.items():
            ordered=[reported[(p['L'],p['r'])] for p in ps];b=bracket(ordered)
            plans[label][rule]=dict(covered=b is not None,configs=[dict(L=p['L'],r=p['rerank_limit'],recall=p['recall']) for p in ordered])
            if b:
                a,c,w=b;plans[label][rule].update(bracket=[(a['L'],a['rerank_limit']),(c['L'],c['rerank_limit'])],weight=w)
    Q.save(root/'report-brackets.json',plans)
    # The old engine has the same NUMA observation contract but a different
    # executable name and timing-field names from the pointer-only executor.
    original_sample=Q.sample
    def sample(pid):
        x=original_sample(pid);obs=Q.numa_guard.observe(pid)
        if obs:x['numa']=obs
        return x
    Q.sample=sample;pending=set(Q.load(O/'timings/pending.json')) if (O/'timings/pending.json').exists() else set()
    labels=list(arms)
    try:
      for rep in range(3):
       order=labels[rep:]+labels[:rep]
       if rep%2:order.reverse()
       for label in order:
        arm=arms[label];configs=set()
        for p in plans[label].values():
            if p['covered']:configs.update(tuple(c) for c in p['bracket'])
        for L,r in sorted(configs):
            name=label+'-L'+str(L)+'-r'+str(r)+'-rep'+str(rep+1);accepted=O/'timings'/(name+'-accepted.json')
            if accepted.exists():pending.discard(name);continue
            success=False
            for attempt in range(1,4):
                out=O/'results'/(name+'-attempt'+str(attempt)+'.json')
                try:
                    observations=Q.execute(args(arm['version'],arm['bucket'],'report',out,[L]),name+'-attempt'+str(attempt),env(arm['filtering'],r,False,False),True,240)
                except RuntimeError as exc:
                    if 'bounded timeout' not in str(exc):raise
                    Q.save(O/'timings'/(name+'-guard'+str(attempt)+'.json'),dict(accepted=False,reason='bounded measurement timeout'));continue
                point=Q.load(out)[0];normalized=dict(point,point_start_unix=point['start_unix'],measured_seconds=point['seconds'])
                guard=Q.validate_window(observations,normalized);Q.save(O/'timings'/(name+'-guard'+str(attempt)+'.json'),guard)
                if guard['accepted']:Q.save(accepted,dict(point=point,guard=guard,result=str(out)));success=True;break
                time.sleep(5)
            if success:pending.discard(name)
            else:pending.add(name)
    finally:Q.sample=original_sample
    Q.save(O/'timings/pending.json',sorted(pending))
    summary={}
    metrics=['latency_us','qps','submitted_pages','topology_submitted_pages','coordinate_submitted_pages','topology_submit_wait_us','coordinate_submit_wait_us','expanded_nodes','pq_space_evals']
    for label,rules in plans.items():
      summary[label]={}
      for rule,p in rules.items():
        if not p['covered']:summary[label][rule]=p;continue
        repeats=[]
        for rep in range(1,4):
            points=[]
            for L,r in p['bracket']:
                path=O/'timings'/(label+'-L'+str(L)+'-r'+str(r)+'-rep'+str(rep)+'-accepted.json')
                if not path.exists():break
                points.append(Q.load(path)['point'])
            if len(points)!=2:continue
            a,b=points;w=p['weight'];repeats.append({k:(1-w)*a[k]+w*b[k] for k in metrics})
        summary[label][rule]=dict(p,repeats=repeats,complete=len(repeats)==3,mean={k:statistics.mean(p[k] for p in repeats) for k in metrics} if repeats else {})
    Q.save(root/'matched95-summary.json',dict(scope='OpenImages existing mixed-alpha buckets; 4 threads NUMA3 SSD T|XY; no fixed-alpha GT invented; no p95 interpolation',results=summary))
    Q.save(root/'status.json',dict(stage='complete' if not pending else 'complete_with_pending',pending=sorted(pending),covered_cases=sum(p['covered'] for v in plans.values() for p in v.values())))
