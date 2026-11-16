"""Five-weight SSD test of two relaxed unified graphs, no range sidecars."""
import json,struct,sys
from pathlib import Path
import numpy as np
import common as c
import measure
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
from no_sidecar import alpha,gt,jobs,byte_equal,distances
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P
OUT=c.RUN/'relaxed'
WEIGHTS=[.1,.3,.5,.7,.9]
def build(rho):
    root=OUT/'graphs'/('rho'+str(rho));root.mkdir(parents=True,exist_ok=True)
    binary=OUT/'bin/build-relaxed';stamp=root/'identity.json'
    identity=dict(rho=rho,binary=c.sha(binary),subset=c.sha(c.D/'subset.json'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['graph_sha256']!=c.sha(root/'float-graph.bin'):raise RuntimeError('relaxed build identity changed')
        return root
    c.capacity(8*2**30,'relaxed unified graph rho'+str(rho));start=c.time.time()
    c.run([binary,'dual',c.D/'base_img_emb.fvecs',c.D/'base_text_emb.fvecs',2000000,512,512,8,root], 'build-unified-rho'+str(rho),cpus='72-79',timeout=8*3600,extra=dict(HVS_UNIFIED_RHO=str(rho),HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8'))
    audit=c.load(root/'graph-audit.json')
    if audit['nodes']!=2000000 or audit['quantized_lost_grid_points']!=0:raise RuntimeError('writer failed')
    c.save(stamp,dict(identity=identity,graph_sha256=c.sha(root/'float-graph.bin'),wall_seconds=c.time.time()-start,audit=audit));return root

def pack(name,source):
    root=OUT/'packed'/name;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    control=c.U/'stage2-v2/graphs/unified'
    identity=dict(source=c.sha(source/'float-graph.bin'),entry=c.sha(control/'full.meta'),packer=c.sha(c.MG_RUN/'bin/pack'))
    if stamp.exists():
        old=c.load(stamp)
        if old['identity']!=identity or old['topology_sha256']!=c.sha(root/'compact.topo'):raise RuntimeError('packed graph changed')
        return old
    log=c.run([c.MG_RUN/'bin/pack',source/'full.meta',source/'float-graph.bin',root/'compact.meta',root/'compact.topo','-',control/'full.meta'],'pack-relaxed-'+name,cpus='80,81',timeout=900)
    stats=json.loads(log.read_text());n=stats['nodes'];pages=(n+48)//49
    if stats['record_bytes']!=164 or stats['nodes_per_page']!=49:raise RuntimeError('noncompact record')
    mm=np.memmap(str(root/'compact.topo'),dtype='u1',mode='r');degree=np.ndarray((pages,49),dtype='<u4',buffer=mm,strides=(8192,164)).reshape(-1)[:n]
    if degree.max()>40:raise RuntimeError('R exceeded')
    stats.update(mean_degree=float(degree.mean()),max_degree=int(degree.max()),full_slot_fraction=float(np.mean(degree==40)),topology_bytes=(root/'compact.topo').stat().st_size,meta_bytes=8192)
    c.save(stamp,dict(identity=identity,stats=stats,topology_sha256=c.sha(root/'compact.topo')));return c.load(stamp)

def command(manifest,a,jp,out,cohort='report',medium='dram',seconds=0,trace='-',ids=None):
    return [c.MG_RUN/'bin/multigraph',manifest,c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',alpha(a),gt(a),ids or c.MG_RUN/'data'/(cohort+'-ids.txt'),out,medium,4,seconds,'pq',jp,trace,0]

def prepare(name,source):
    stats=pack(name,source);root=OUT/'search'/name;root.mkdir(parents=True,exist_ok=True)
    header=(c.MG_RUN/'manifest.txt').read_text().splitlines()[:3];header[0]='HVS_MULTI_V1 2000000 512 512 1'
    p=OUT/'packed'/name;header.append(' '.join(map(str,['u',p/'compact.meta',p/'compact.topo','-',0])))
    manifest=root/'manifest.txt';manifest.write_text('\n'.join(header)+'\n')
    plan=dict(manifest=c.sha(manifest),binary=c.sha(c.MG_RUN/'bin/multigraph'),protocol=c.sha(c.ROOT/'README.md'),topology=stats['topology_sha256'])
    stamp=root/'identity.json'
    if stamp.exists() and c.load(stamp)!=plan:raise RuntimeError('search identity changed')
    c.save(stamp,plan)
    gate=root/'gates';gate.mkdir(exist_ok=True);idfile=gate/'ids.txt';idfile.write_text('\n'.join((c.MG_RUN/'data/report-ids.txt').read_text().splitlines()[:16])+'\n')
    curves={}
    for a in WEIGHTS:
        tag=str(a);folder=root/tag;folder.mkdir(exist_ok=True);alljobs=P.jobs('DEG_off',a,a);jp=folder/'tune.jobs';jobs(jp,alljobs)
        tune=folder/'tune.json'
        if not tune.exists():c.run(command(manifest,a,jp,tune,'tune'),'density-tune-'+name+'-'+tag,timeout=1800)
        frozen=dict(frontiers=P.frontiers(c.load(tune)),jobs={j['label']:j for j in alljobs},tune_sha256=c.sha(tune))
        path=folder/'frozen.json'
        if path.exists() and c.load(path)!=frozen:raise RuntimeError('training freeze changed')
        c.save(path,frozen)
        labels=sorted(set(x for xs in frozen['frontiers'].values() for x in xs));jp=folder/'report.jobs';jobs(jp,[frozen['jobs'][x] for x in labels])
        report=folder/'report.json'
        if not report.exists():c.run(command(manifest,a,jp,report),'density-report-work-'+name+'-'+tag,timeout=1800)
        by={p['label']:p for p in c.load(report)};brackets={};chosen=set()
        for rule,xs in frozen['frontiers'].items():
            for target in (.90,.95):
                b=P.crossing([by[x] for x in xs],target);key=rule+'-'+str(target)
                if b is None:brackets[key]=dict(covered=False);continue
                x,y,w=b;brackets[key]=dict(covered=True,labels=[x['label'],y['label']],weight=w,target=target);chosen.update([x['label'],y['label']])
        if not chosen:raise RuntimeError('no recall coverage')
        # Same frozen high-depth configuration, no extra parameter search.
        gj=gate/(tag+'.jobs');jobs(gj,[alljobs[-1]])
        check=gate/(tag+'-check.json')
        if not check.exists():
            pp={}
            for medium in ('dram','ssd'):
                out=gate/(tag+'-'+medium+'.json');prefix=gate/(tag+'-'+medium)
                c.run(command(manifest,a,gj,out,medium=medium,trace=prefix,ids=idfile),'density-gate-'+name+'-'+tag+'-'+medium,cpus='80-83',timeout=600)
                if any(p['sidecar_bytes'] for p in c.load(out)):raise RuntimeError('range sidecar loaded')
                pp[medium]=prefix
            byte_equal(pp['dram'],pp['ssd'],[alljobs[-1]['label']]);error=distances(c.load(gate/(tag+'-ssd.json')),a)
            c.save(check,dict(status='PASS',trajectory_results_equal=True,max_return_relative_error=error))
        jobs(folder/'time.jobs',[frozen['jobs'][x] for x in sorted(chosen)]);c.save(folder/'brackets.json',brackets)
        curves[tag]=dict(folder=str(folder),manifest=str(manifest),brackets=brackets)
    return curves

def main():
    for n in ('graphs','packed','search'):(OUT/n).mkdir(parents=True,exist_ok=True)
    sources={'control':c.U/'stage2-v2/graphs/unified'}
    for rho in (1.025,1.05):sources['rho'+str(rho)]=build(rho)
    curves={name:prepare(name,source) for name,source in sources.items()}
    tasks=[(n,a) for n in sources for a in WEIGHTS];pending=[]
    for rep in range(1,4):
        order=tasks[rep-1:]+tasks[:rep-1]
        if rep%2==0:order=list(reversed(order))
        for name,a in order:
            plan=curves[name][str(a)];folder=Path(plan['folder']);saved=folder/('rep'+str(rep)+'.json')
            if saved.exists():continue
            choice=measure.quiet();c.save(folder/('affinity-rep'+str(rep)+'.json'),choice)
            out=folder/('rep'+str(rep)+'-raw.json')
            data=measure.run(command(Path(plan['manifest']),a,folder/'time.jobs',out,medium='ssd',seconds=10),'density-time-'+name+'-'+str(a)+'-rep'+str(rep),out,choice['choice']['cpus'])
            if data:c.save(saved,data)
            else:pending.append([name,a,rep])
    results=[]
    for name in sources:
        for a in WEIGHTS:
            plan=curves[name][str(a)];folder=Path(plan['folder'])
            for key,bracket in plan['brackets'].items():
                if not bracket['covered']:results.append(dict(graph=name,alpha=a,case=key,covered=False));continue
                reps=[]
                for rep in range(1,4):
                    f=folder/('rep'+str(rep)+'.json')
                    if not f.exists():continue
                    by={p['label']:p for p in c.load(f)['points']};match=P.matched([by[x] for x in bracket['labels']],bracket['target'])
                    if not match['covered']:raise RuntimeError('timed recall changed')
                    reps.append(match)
                row=dict(graph=name,alpha=a,case=key,covered=len(reps)==3,repeats=reps)
                if reps:row['mean']={k:float(np.mean([r[k] for r in reps])) for k in ('latency_us','qps','expanded','pq_space_evals','reranked','topology_pages','coordinate_pages')}
                results.append(row)
    comparisons=[]
    for rho in ('rho1.025','rho1.05'):
        for a in WEIGHTS:
            group={r['graph']:r for r in results if r['alpha']==a and r['case']=='weighted_pages-0.95'}
            if not all(group[n]['covered'] for n in ('control',rho)):continue
            ratios=[r['latency_us']/u['latency_us'] for r,u in zip(group[rho]['repeats'],group['control']['repeats'])]
            comparisons.append(dict(graph=rho,alpha=a,latency_ratios=ratios,stable_ten_percent_gain=all(x<=.9 for x in ratios)))
    c.save(OUT/'summary.json',dict(results=results,pending=pending,primary_comparisons=comparisons,scope='Laion2M five weights; no sidecar; original node order'))
    text='# Relaxed unified graph\n\nFrozen weighted-page rule, 95% Recall@10, SSD four threads NUMA3; all strengths and weights retained.\n\n| graph | alpha | latency ms | topology pages | coordinate pages | expansions | PQ space evaluations | stable >=10% gain |\n|---|---:|---:|---:|---:|---:|---:|---|\n'
    for r in results:
        if r['case']!='weighted_pages-0.95' or not r['covered']:continue
        m=r['mean'];gain=next((v['stable_ten_percent_gain'] for v in comparisons if v['graph']==r['graph'] and v['alpha']==r['alpha']),None)
        text+='| %s | %.1f | %.4f | %.2f | %.2f | %.2f | %.1f | %s |\n'%(r['graph'],r['alpha'],m['latency_us']/1000,m['topology_pages'],m['coordinate_pages'],m['expanded'],m['pq_space_evals'],gain)
    (OUT/'RESULTS.md').write_text(text+'\nDegree/storage statistics: packed/*/identity.json. Missing points are pending, never extrapolated.\n')
    return dict(complete=not pending,summary=str(OUT/'summary.json'))
