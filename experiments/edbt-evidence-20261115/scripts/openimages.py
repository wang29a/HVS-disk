"""Second dataset: corrected unified vs equal dual vs nearest of three fixed graphs."""
import json,struct,sys
from pathlib import Path
import numpy as np
import common as c
import measure
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
from no_sidecar import matrix,jobs,byte_equal
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P
OUT=c.RUN/'openimages'
DATA=Path('/mnt/nvme3/wz/dataset/Openimage_disk')
CONTROL=c.U/'review-20261113/graphs/openimage'
PQ=Path('/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full')
METHODS=['DEG_off','M_equal','Or_nearest']
FILES={'qx':DATA/'query_img_emb.fvecs','qy':DATA/'query_text_emb.fvecs','alpha':DATA/'base_gt_round_00/query_alpha.fvecs','gt':DATA/'base_gt_round_00/top10_results.ivecs'}

def transform(text):
    start=text.index('  static unsigned route(float alpha) {');end=text.index('  Engine(',start)
    text=text[:start]+'''  static unsigned route(float alpha) {
    const float weights[] = {.1f,.5f,.9f};
    const unsigned indices[] = {2,4,6};
    unsigned best=0;
    for(unsigned j=1;j<3;++j)
      if(std::abs(alpha-weights[j])<std::abs(alpha-weights[best])) best=j;
    return indices[best];
  }
'''+text[end:]
    # Export per-query deterministic work for alpha/contribution bins. Latency
    # and QPS headlines remain the entire measured mixed workload.
    needle='out << "]}";\n      }\n      out << "]}";'
    if text.count(needle)!=1:raise RuntimeError('per-query export hook changed')
    extra='''out << "]";
        U qexpanded=0,qpq=0,qtopo=0;
        for(const auto &g:a.graph) {qexpanded+=g.expanded; qpq+=g.pq; qtopo+=g.submitted;}
        out << ",\\\"expanded\\\":" << qexpanded << ",\\\"pq_space_evals\\\":" << qpq
            << ",\\\"topology_pages\\\":" << qtopo << ",\\\"coordinate_pages\\\":" << a.coord_submitted
            << ",\\\"reranked\\\":" << a.reranked << "}";
      }
      out << "]}";'''
    # Python string escaping above deliberately generates JSON-escaped C++ literals.
    extra=extra.replace(chr(92)*3,chr(92))
    text=text.replace(needle,extra)
    return text

def compile_executor():
    for n in ('bin','build','validation'):(OUT/n).mkdir(parents=True,exist_ok=True)
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');lib=c.E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if c.sha(lib)!=cfg['library_sha256']:raise RuntimeError('library changed')
    src=c.MG/'scripts/multigraph.cpp';p=OUT/'build/multigraph.cpp';p.write_bytes(transform(src.read_bytes().decode('utf-8')).encode('utf-8'))
    c.run(['clang-format','-style=LLVM','-i',p],'format-openimages-executor',cpus='80,81',timeout=60)
    c.run(['g++-11']+cfg['flags']+[p,lib]+cfg['link']+['-o',OUT/'bin/multigraph'],'compile-openimages-executor',cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    c.save(OUT/'validation/build.json',dict(status='PASS',binary=c.sha(OUT/'bin/multigraph'),source=c.sha(p),parent_source=c.sha(src),changes='three-way nearest route, per-query work export; traversal unchanged'))

def inputs():
    out=OUT/'data';out.mkdir(exist_ok=True);stamp=out/'identity.json'
    sig=dict(files={k:c.sha(p) for k,p in FILES.items()},base={n:c.sha(DATA/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs')},split=c.sha(c.E/'metric-consistent-20261111/analysis/tuning-split-v2.json'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=sig:raise RuntimeError('OpenImages source changed')
        for n,h in old['outputs'].items():
            if c.sha(out/n)!=h:raise RuntimeError('OpenImages shared input changed')
        return
    c.capacity(5*2**30,'OpenImages shared coordinate pages and compact graphs')
    x,y=(matrix(DATA/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs'))
    if x.shape!=(456699,768) or y.shape!=x.shape:raise RuntimeError('dataset dimensions')
    with (out/'xy.pages').open('wb') as f:
        for first in range(0,len(x),1024):
            block=np.concatenate((x[first:first+1024],y[first:first+1024]),axis=1).astype('<f4')
            pages=np.zeros((len(block),2048),dtype='<f4');pages[:,:1536]=block;pages.tofile(f)
    for space in ('emb','loc'):
        for suffix in ('pq_compressed.bin','pq_pivots.bin'):
            src=PQ/(space+'_'+suffix);dst=out/src.name
            if dst.is_symlink():
                if dst.resolve()!=src.resolve():raise RuntimeError('PQ link changed')
            elif dst.exists():raise RuntimeError('unexpected PQ file')
            else:dst.symlink_to(src)
    split=c.load(c.E/'metric-consistent-20261111/analysis/tuning-split-v2.json')
    for cohort in ('tune','report'):
        ids=split['cohorts'][cohort]['query_ids']
        if len(ids)!=500:raise RuntimeError('query split not500')
        (out/(cohort+'-ids.txt')).write_text('\n'.join(map(str,ids))+'\n')
    (out/'gate-ids.txt').write_text('\n'.join((out/'report-ids.txt').read_text().splitlines()[:16])+'\n')
    # Native GT is reused. Actual contribution is defined from its top10.
    qx,qy=matrix(FILES['qx']),matrix(FILES['qy']);weights=matrix(FILES['alpha'])[:,0];truth=matrix(FILES['gt'],'<u4');contrib={}
    for qi in range(len(qx)):
        dx=np.linalg.norm(x[truth[qi]].astype(np.float64)-qx[qi].astype(np.float64),axis=1);dy=np.linalg.norm(y[truth[qi]].astype(np.float64)-qy[qi].astype(np.float64),axis=1)
        a=float(weights[qi]);v=a*dx+(1-a)*dy;good=v>0
        contrib[str(qi)]=dict(alpha=a,actual_image_contribution=float(np.mean(a*dx[good]/v[good])) if good.any() else None)
    c.save(out/'contribution.json',contrib)
    outputs={p.name:c.sha(p) for p in out.iterdir() if p.is_file() and p.name!='identity.json'}
    c.save(stamp,dict(identity=sig,outputs=outputs,shape=[456699,768,768],GT='existing mixed alpha, unchanged',coordinate_records_per_page=1))

def build(name):
    root=OUT/'graphs'/name;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')
    if name in ('x','y'):binary=c.MG_RUN/'bin/build-vamana';mode='single';a=None
    else:binary=c.U/'stage2-v2/bin/build-fixed';mode='dual';a={'a01':.1,'a05':.5,'a09':.9}[name]
    identity=dict(binary=c.sha(binary),inputs=c.sha(OUT/'data/identity.json'),alpha=a)
    if name not in ('x','y') and identity['binary']!=proof['binaries']['fixed']['builder_sha256']:raise RuntimeError('fixed builder changed')
    if name in ('x','y') and identity['binary']!=c.load(c.MG_RUN/'validation/build.json')['binaries']['build-vamana']:raise RuntimeError('Vamana builder changed')
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['graph_sha256']!=c.sha(root/'float-graph.bin'):raise RuntimeError('graph changed')
        return root
    c.capacity(3*2**30,'OpenImages graph '+name);start=c.time.time();first='base_text_emb.fvecs' if name=='y' else 'base_img_emb.fvecs'
    extra=dict(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8')
    if a is not None:extra['HVS_FIXED_ALPHA']=str(a)
    c.run([binary,mode,DATA/first,DATA/'base_text_emb.fvecs',456699,768,768,8,root],'build-openimages-'+name,cpus='72-79',timeout=4*3600,extra=extra)
    audit=c.load(root/'graph-audit.json')
    if audit['nodes']!=456699 or audit['quantized_lost_grid_points']!=0:raise RuntimeError('writer audit')
    c.save(stamp,dict(identity=identity,graph_sha256=c.sha(root/'float-graph.bin'),wall_seconds=c.time.time()-start,audit=audit));return root

def source_meta(source):
    """The native Vamana source-only writer emits compact.meta for M=0."""
    full=source/'full.meta'
    if full.exists():return full
    compact=source/'compact.meta'
    raw=compact.read_bytes()
    if len(raw)!=8192:raise RuntimeError('Vamana compact source metadata must be8192B')
    n,d1,d2,R,M=struct.unpack_from('<5I',raw);npp=struct.unpack_from('<Q',raw,20)[0]
    if not n or not d1 or not d2 or R!=40 or M!=0 or npp!=49:raise RuntimeError('unexpected Vamana source metadata')
    return compact

def pack(name,source):
    root=OUT/'packed'/name;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    identity=dict(source=c.sha(source/'float-graph.bin'),entry=c.sha(CONTROL/'full.meta'),packer=c.sha(c.MG_RUN/'bin/pack'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['sha256']!=c.sha(root/'compact.topo'):raise RuntimeError('packed graph changed')
        return
    log=c.run([c.MG_RUN/'bin/pack',source_meta(source),source/'float-graph.bin',root/'compact.meta',root/'compact.topo','-',CONTROL/'full.meta'],'pack-openimages-'+name,cpus='80,81',timeout=600)
    c.save(stamp,dict(identity=identity,stats=json.loads(log.read_text()),sha256=c.sha(root/'compact.topo'),topology_bytes=(root/'compact.topo').stat().st_size+8192))

def command(method,jp,out,cohort='report',medium='dram',seconds=0,trace='-',gate=False,audit=0):
    return [OUT/'bin/multigraph',OUT/'manifest.txt',FILES['qx'],FILES['qy'],FILES['alpha'],FILES['gt'],OUT/'data'/('gate-ids.txt' if gate else cohort+'-ids.txt'),out,medium,4,seconds,'pq',jp,trace,audit]

def work(p):
    t=sum(g['topology_pages'] for g in p['graphs']);d=p['coordinate_pages'];pq=sum(g['pq_space_evals'] for g in p['graphs'])
    return dict(pages=t+d,compute=pq*64+p['reranked']*1536,weighted_pages=24.8*t+5.2*d,topology_pages=t,coordinate_pages=d,pq=pq)

def tune_and_gate():
    curves={};oldwork=P.work;P.work=work
    try:
      for method in METHODS:
        root=OUT/'search'/method;root.mkdir(parents=True,exist_ok=True);alljobs=P.jobs(method,.5,.5);jp=root/'tune.jobs';jobs(jp,alljobs);out=root/'tune.json'
        if not out.exists():c.run(command(method,jp,out,'tune'),'openimages-tune-'+method,timeout=1800)
        frozen=dict(frontiers=P.frontiers(c.load(out)),jobs={v['label']:v for v in alljobs},tune_sha256=c.sha(out));stamp=root/'frozen.json'
        if stamp.exists() and c.load(stamp)!=frozen:raise RuntimeError('frozen OpenImages grid changed')
        c.save(stamp,frozen);labels=sorted(set(x for xs in frozen['frontiers'].values() for x in xs));jobs(root/'report.jobs',[frozen['jobs'][x] for x in labels]);out=root/'report.json'
        if not out.exists():c.run(command(method,root/'report.jobs',out),'openimages-report-'+method,timeout=1800)
        points={p['label']:p for p in c.load(out)};brackets={};chosen=set()
        for rule,order in frozen['frontiers'].items():
            for target in (.9,.95):
                pair=P.crossing([points[x] for x in order],target);key=rule+'-'+str(target)
                if pair is None:brackets[key]=dict(covered=False);continue
                a,b,w=pair;brackets[key]=dict(covered=True,labels=[a['label'],b['label']],target=target,weight=w);chosen.update([a['label'],b['label']])
        if not chosen:raise RuntimeError('no OpenImages recall target covered '+method)
        jobs(root/'time.jobs',[frozen['jobs'][x] for x in sorted(chosen)]);c.save(root/'brackets.json',brackets)
        gate=root/'gate.json'
        if not gate.exists():
            jobs(root/'gate.jobs',[alljobs[-1]]);prefixes={};errors=[]
            for medium in ('dram','ssd'):
                out=root/('gate-'+medium+'.json');prefix=root/('gate-'+medium)
                c.run(command(method,root/'gate.jobs',out,medium=medium,trace=prefix,gate=True,audit=1),'openimages-gate-'+method+'-'+medium,cpus='80-83',timeout=600)
                ps=c.load(out)
                if any(p['sidecar_bytes'] for p in ps):raise RuntimeError('range sidecar loaded')
                errors += [p['max_return_relative_error'] for p in ps];prefixes[medium]=prefix
            byte_equal(prefixes['dram'],prefixes['ssd'],[alljobs[-1]['label']]);c.save(gate,dict(status='PASS',max_return_relative_error=max(errors),trace_and_result_byte_equal=True))
        curves[method]=brackets
    finally:P.work=oldwork
    return curves

def main():
    for n in ('data','graphs','packed','search'):(OUT/n).mkdir(parents=True,exist_ok=True)
    compile_executor();inputs();sources={'u':CONTROL}
    for name in ('x','y','a01','a05','a09'):sources[name]=build(name)
    for name,src in sources.items():pack(name,src)
    lines=['HVS_MULTI_V1 456699 768 768 6',str(OUT/'data/xy.pages'),str(OUT/'data')+'/']
    for name in sources:
        p=OUT/'packed'/name;lines.append(' '.join(map(str,[name,p/'compact.meta',p/'compact.topo','-',0])))
    (OUT/'manifest.txt').write_text('\n'.join(lines)+'\n');curves=tune_and_gate();pending=[]
    for rep in range(1,4):
        order=METHODS[rep-1:]+METHODS[:rep-1]
        for method in order:
            root=OUT/'search'/method;saved=root/('rep'+str(rep)+'.json')
            if saved.exists():continue
            choice=measure.quiet();c.save(root/('affinity-rep'+str(rep)+'.json'),choice);out=root/('rep'+str(rep)+'-raw.json')
            data=measure.run(command(method,root/'time.jobs',out,medium='ssd',seconds=10),'openimages-time-'+method+'-rep'+str(rep),out,choice['choice']['cpus'])
            if data:c.save(saved,data)
            else:pending.append([method,rep])
    rows=[]
    for method,brackets in curves.items():
        for key,b in brackets.items():
            if not b['covered']:rows.append(dict(method=method,case=key,covered=False));continue
            reps=[]
            for rep in range(1,4):
                path=OUT/'search'/method/('rep'+str(rep)+'.json')
                if not path.exists():continue
                by={p['label']:p for p in c.load(path)['points']};r=P.matched([by[x] for x in b['labels']],b['target'])
                if not r['covered']:raise RuntimeError('recall changed')
                reps.append(r)
            row=dict(method=method,case=key,covered=len(reps)==3,repeats=reps)
            if reps:row['mean']={k:float(np.mean([r[k] for r in reps])) for k in ('latency_us','qps','expanded','pq_space_evals','reranked','topology_pages','coordinate_pages')}
            rows.append(row)
    # Binned observations are deterministic held-out work/recall, not separate
    # bin-timing or bin-optimal configurations. Preserve raw curves for review.
    contrib=c.load(OUT/'data/contribution.json');edges=[0,.05,.2,.4,.6,.8,.95,1];bins=[]
    for method in METHODS:
        for point in c.load(OUT/'search'/method/'report.json'):
            for axis in ('alpha','actual_image_contribution'):
                for lo,hi in zip(edges,edges[1:]):
                    qs=[q for q in point['queries'] if contrib[str(q['query_id'])][axis] is not None and lo<=contrib[str(q['query_id'])][axis] and (contrib[str(q['query_id'])][axis]<hi or hi==1 and contrib[str(q['query_id'])][axis]==1)]
                    if qs:bins.append(dict(method=method,label=point['label'],axis=axis,lo=lo,hi=hi,count=len(qs),recall=float(np.mean([q['hits']/10 for q in qs])),work={k:float(np.mean([q[k] for q in qs])) for k in ('expanded','pq_space_evals','topology_pages','coordinate_pages','reranked')}))
    c.save(OUT/'bins-work.json',dict(scope='global train-selected configs; held-out work and recall only, no bin latency claim',rows=bins))
    c.save(OUT/'summary.json',dict(results=rows,pending=pending,scope='OpenImages full, mixed-alpha, three-way nearest, no sidecar',storage={name:c.load(OUT/'packed'/name/'identity.json') for name in sources}))
    text='# OpenImages full multigraph comparison\n\nExisting mixed-alpha GT, corrected unified, equal dual Vamana, nearest of fixed {.1,.5,.9}; SSD four threads NUMA3, no shared cache.\n\n| method | rule/recall | latency ms | QPS | topology pages | coordinate pages |\n|---|---|---:|---:|---:|---:|\n'
    for r in rows:
        if not r['covered']:continue
        m=r['mean'];text+='| %s | %s | %.4f | %.1f | %.2f | %.2f |\n'%(r['method'],r['case'],m['latency_us']/1000,m['qps'],m['topology_pages'],m['coordinate_pages'])
    (OUT/'RESULTS.md').write_text(text+'\nBinned work/recall in bins-work.json. No ideal-oracle or native-HNSW claim.\n');return dict(complete=not pending,summary=str(OUT/'summary.json'))
