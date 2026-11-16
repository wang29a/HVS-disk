"""Four centre graphs including reused rho=1; matched-degree work, not latency tuning."""
import itertools,json,struct,sys
from pathlib import Path
import numpy as np
import common as c
import measure
from no_sidecar import matrix,jobs
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P

OUT=c.RUN/'degree'
GRID=[1,1.025,1.05,1.1,1.2,1.4,1.8,2]
SOURCES={'unified':c.U/'stage2-v2/graphs/unified','rho1':c.U/'stage2-v2/graphs/fixed-0.5'}
def choose(pilots):
    values=[(float(rho),row['degree']) for rho,row in pilots.items() if float(rho)>1]
    options=[]
    for group in itertools.combinations(values,3):
        ordered=sorted(group,key=lambda x:(x[1],x[0]));cost=sum(abs(d-t) for (_,d),t in zip(ordered,(24,30,35)))
        options.append((cost,tuple(r for r,d in ordered)))
    return list(min(options)[1])
def pilot_data():
    root=OUT/'pilot-data';root.mkdir(exist_ok=True);stamp=root/'identity.json'
    if stamp.exists():
        old=c.load(stamp)
        if old['source_subset_sha256']!=c.sha(c.D/'subset.json'):raise RuntimeError('pilot source subset changed')
        for name,h in old['files'].items():
            if c.sha(root/name)!=h:raise RuntimeError('pilot vectors changed '+name)
        return root
    rng=np.random.RandomState(20261114);ids=np.sort(rng.choice(2000000,100000,replace=False)).astype('<u4')
    from no_sidecar import matrix
    for name in ('base_img_emb.fvecs','base_text_emb.fvecs'):
        source=matrix(c.D/name);values=np.asarray(source[ids],dtype='<f4')
        with (root/name).open('wb') as f:f.write(struct.pack('<II',*values.shape));values.tofile(f)
    c.save(stamp,dict(seed=20261114,nodes=100000,source_subset_sha256=c.sha(c.D/'subset.json'),ids_sha256=__import__('hashlib').sha256(ids.tobytes()).hexdigest(),files={n:c.sha(root/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs')}))
    return root
def build(rho,pilot=False):
    name='rho'+str(rho);folder=OUT/('pilot' if pilot else 'graphs')/name;folder.mkdir(parents=True,exist_ok=True)
    data=pilot_data() if pilot else c.D;n=100000 if pilot else 2000000;binary=OUT/'bin/build-strength';stamp=folder/'identity.json'
    identity=dict(rho=rho,nodes=n,builder_sha256=c.sha(binary),inputs_sha256=c.sha(data/'identity.json' if pilot else c.D/'subset.json'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['graph_sha256']!=c.sha(folder/'float-graph.bin'):raise RuntimeError('degree build identity changed')
        return folder,old
    c.capacity(6*2**30,'centre degree graph '+name)
    start=c.time.time();c.run([binary,'dual',data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',n,512,512,8,folder],('pilot-' if pilot else 'full-')+name,cpus='72-79',timeout=8*3600,
        extra=dict(HVS_FIXED_ALPHA='0.5',HVS_PRUNE_RHO=str(rho),HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8'))
    audit=c.load(folder/'graph-audit.json')
    if audit['nodes']!=n or audit['quantized_lost_grid_points']!=0:raise RuntimeError('degree writer audit failed')
    result=dict(identity=identity,graph_sha256=c.sha(folder/'float-graph.bin'),degree=audit['edges']/n,edges=audit['edges'],wall_seconds=c.time.time()-start)
    c.save(stamp,result);return folder,result
def pack(name,source):
    root=OUT/'packed'/name;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    identity=dict(raw_sha256=c.sha(source/'float-graph.bin'),meta_sha256=c.sha(source/'full.meta'),entry_meta_sha256=c.sha(SOURCES['unified']/'full.meta'),pack_binary_sha256=c.sha(c.MG_RUN/'bin/pack'))
    if stamp.exists():
        saved=c.load(stamp)
        if saved['identity']!=identity:raise RuntimeError('degree pack changed')
        return saved
    log=c.run([c.MG_RUN/'bin/pack',source/'full.meta',source/'float-graph.bin',root/'compact.meta',root/'compact.topo','-',SOURCES['unified']/'full.meta'],'pack-degree-'+name,cpus='80,81',timeout=600)
    stats=json.loads(log.read_text());c.save(stamp,dict(identity=identity,stats=stats,topology_sha256=c.sha(root/'compact.topo')));return c.load(stamp)
def exact(name,source):
    root=OUT/'exact';root.mkdir(exist_ok=True);meta=OUT/'packed'/name/'compact.meta'
    entries=struct.unpack_from('<I',(SOURCES['unified']/'full.meta').read_bytes(),28)[0]
    grid=[x for x in [20,30,40,60,80,100,120,160,240,400,800,1600] if x>=entries];points={}
    for cohort in ('tune','report'):
        out=root/(name+'-'+cohort+'.json')
        if not out.exists():
            c.run([c.U/'stage2/bin/pointer-search',meta,source/'float-graph.bin',c.D/'base_img_emb.fvecs',c.D/'base_text_emb.fvecs',c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',c.D/'query-alpha-0.5.bin',c.D/'gt-alpha-0.5.bin',c.MG_RUN/'data'/(cohort+'-ids.txt'),out,4,0,0]+grid,
                'exact-degree-'+name+'-'+cohort,timeout=1200,extra=dict(PTR_ENTRY_META=str(SOURCES['unified']/'full.meta')))
        points[cohort]=c.load(out)
    pair=P.crossing(points['report'],.95)
    if pair is None:return dict(covered=False)
    a,b,w=pair
    return dict(covered=True,expanded=(1-w)*a['expanded']+w*b['expanded'],exact_space_evals=(1-w)*a['exact_space_evals']+w*b['exact_space_evals'],L_bracket=[a['L'],b['L']],recall_bracket=[a['recall'],b['recall']],weight=w)
def mg_command(manifest,jp,out,cohort,medium='dram',seconds=0):
    return [c.MG_RUN/'bin/multigraph',manifest,c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',c.D/'query-alpha-0.5.bin',c.D/'gt-alpha-0.5.bin',c.MG_RUN/'data'/(cohort+'-ids.txt'),out,medium,4,seconds,'pq',jp,'-',0]
def ssd(name,choice):
    root=OUT/'ssd'/name;root.mkdir(parents=True,exist_ok=True);packed=OUT/'packed'/name
    header=(c.MG_RUN/'manifest.txt').read_text().splitlines()[:3];header[0]='HVS_MULTI_V1 2000000 512 512 1';header.append(' '.join(map(str,['u',packed/'compact.meta',packed/'compact.topo','-',0])))
    manifest=root/'manifest.txt';manifest.write_text('\n'.join(header)+'\n');alljobs=P.jobs('DEG_off',.5,.5);jp=root/'tune.jobs';jobs(jp,alljobs);out=root/'tune.json'
    if not out.exists():c.run(mg_command(manifest,jp,out,'tune'),'degree-pq-tune-'+name,timeout=1200)
    training=c.load(out);labels=P.frontiers(training)['pages'];jobmap={x['label']:x for x in alljobs};selected=[jobmap[x] for x in labels];jobs(root/'frozen.jobs',selected)
    c.save(root/'frozen.json',dict(labels=labels,selection='training-only pages frontier',train_sha256=c.sha(out),jobs=selected))
    report=root/'report-work.json'
    if not report.exists():c.run(mg_command(manifest,root/'frozen.jobs',report,'report'),'degree-pq-report-'+name,timeout=1200)
    by={p['label']:p for p in c.load(report)};pair=P.crossing([by[x] for x in labels],.95)
    if pair is None:return dict(covered=False)
    a,b,w=pair;chosen=list(dict.fromkeys([a['label'],b['label']]));jobs(root/'time.jobs',[jobmap[x] for x in chosen]);repeated=[];pending=[]
    for rep in range(1,4):
        accepted=root/('rep'+str(rep)+'.json');timing=root/('rep'+str(rep)+'-raw.json')
        if accepted.exists():data=c.load(accepted)
        else:
            data=measure.run(mg_command(manifest,root/'time.jobs',timing,'report','ssd',10),'degree-time-'+name+'-rep'+str(rep),timing,choice['choice']['cpus'])
            if data:c.save(accepted,data)
        if not data:pending.append(rep);continue
        ps={x['label']:x for x in data['points']};r=P.matched([ps[a['label']],ps[b['label']]],.95)
        if not r['covered']:raise RuntimeError('deterministic recall changed while timing degree graph')
        repeated.append(r)
    result=dict(covered=not pending,repetitions=repeated,pending=pending,labels=[a['label'],b['label']],weight=w)
    if not pending:
        for field in ('latency_us','qps','expanded','topology_pages','coordinate_pages','reranked','pq_space_evals'):
            result[field]=float(np.mean([x[field] for x in repeated]))
    return result
def compare(rows):
    u=rows['unified'];special=sorted((v for k,v in rows.items() if k!='unified'),key=lambda v:v['degree'])
    for a,b in zip(special,special[1:]):
        if a['degree']<=u['degree']<=b['degree'] and a['degree']<b['degree']:
            if not all(v['exact']['covered'] and v['ssd']['covered'] for v in (u,a,b)):return dict(covered=False,reason='missing recall or timing coverage')
            w=(u['degree']-a['degree'])/(b['degree']-a['degree']);expanded=(1-w)*a['exact']['expanded']+w*b['exact']['expanded'];pages=(1-w)*a['ssd']['topology_pages']+w*b['ssd']['topology_pages']
            er=expanded/u['exact']['expanded'];pr=pages/u['ssd']['topology_pages']
            verdict='centre_graph_work_reducible' if er<=.9 and pr<=.9 else 'no_centre_advantage_detected' if abs(er-1)<=.1 and abs(pr-1)<=.1 else 'tradeoff_or_inconclusive'
            return dict(covered=True,degree_bracket=[a['degree'],b['degree']],weight=w,estimated_specialized_expanded=expanded,estimated_specialized_topology_pages=pages,expanded_ratio=er,topology_page_ratio=pr,verdict=verdict,note='interpolation estimate; not a directly built equal-degree graph, and not proof about both endpoints')
    return dict(covered=False,reason='full graph degree range does not bracket unified degree; no extrapolation')
def main():
    for n in ('pilot','graphs','packed','exact','ssd'):(OUT/n).mkdir(parents=True,exist_ok=True)
    import compile_degree;compile_degree.main();pilot_data();pilots={}
    for rho in GRID:
        _,row=build(rho,True);pilots[str(rho)]=row
    selected=choose(pilots);c.save(OUT/'selection.json',dict(pilot=pilots,selected_rho=selected,targets=[24,30,35],rule='pilot average degree only; no query or timing selection'))
    sources=dict(SOURCES)
    for rho in selected:
        source,_=build(rho);sources['rho'+str(rho)]=source
    choice=measure.quiet();c.save(OUT/'affinity.json',choice);rows={}
    for name,source in sources.items():
        stats=pack(name,source)['stats'];rows[name]=dict(degree=stats['edges']/stats['nodes'],edges=stats['edges'],source=str(source),exact=exact(name,source),ssd=ssd(name,choice))
        c.save(OUT/'progress.json',rows)
    result=dict(rows=rows,comparison=compare(rows),scope='centre alpha=.5 only; compact R40 layout; exact-pointer work and SSD PQ/rerank are separate curves')
    c.save(OUT/'summary.json',result)
    import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
    special=sorted((v for k,v in rows.items() if k!='unified'),key=lambda v:v['degree'])
    for field,family,ylabel in [('expanded','exact','Expanded nodes at 95% recall'),('topology_pages','ssd','Submitted topology pages at 95% recall')]:
        fig,ax=plt.subplots(figsize=(6,4));good=[r for r in special if r[family]['covered']];ax.plot([r['degree'] for r in good],[r[family][field] for r in good],'-o',label='specialized alpha=0.5')
        u=rows['unified']
        if u[family]['covered']:ax.scatter([u['degree']],[u[family][field]],marker='*',s=130,label='unified, filter off')
        ax.set_xlabel('Mean out-degree');ax.set_ylabel(ylabel);ax.grid(alpha=.2);ax.legend();fig.tight_layout()
        for ext in ('svg','pdf','png'):fig.savefig(str(OUT/(family+'-degree.'+ext)),dpi=180)
        plt.close(fig)
    text='# 中心权重度数扫描\n\n四张专用图含复用rho=1，采用相同入口与compact格式。精确导航的展开量与SSD PQ搜索的页数分开比较，配置只按训练工作量冻结。\n\n| 图 | 平均度数 | 精确导航展开数 | SSD拓扑页 | SSD坐标页 |\n|---|---:|---:|---:|---:|\n'
    for name,r in rows.items():text+='| %s | %.4f | %s | %s | %s |\n'%(name,r['degree'],'%.2f'%r['exact']['expanded'] if r['exact']['covered'] else '未覆盖','%.2f'%r['ssd']['topology_pages'] if r['ssd']['covered'] else '未覆盖','%.2f'%r['ssd']['coordinate_pages'] if r['ssd']['covered'] else '未覆盖')
    text+='\n事前规则的判读：`'+json.dumps(result['comparison'],ensure_ascii=False)+'`。按度数插值仅为受控预算附近的估计；不能将中心一组结果写成整条U形已被解释。\n'
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8');return dict(complete=all(r['ssd']['covered'] and r['exact']['covered'] for r in rows.values()),summary=str(OUT/'summary.json'))
