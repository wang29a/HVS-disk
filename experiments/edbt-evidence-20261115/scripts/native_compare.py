"""Corrected native TXY vs compact no-range engine, equal total memory allowance."""
from pathlib import Path
import sys
import numpy as np
import common as c
import measure,compile_native,native_writer
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
from no_sidecar import jobs,alpha,gt,matrix
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P
import openimages as O
OUT=c.RUN/'native';THREADS=[1,2,4,8,16,32];BUDGET=2*2**30
LEVELS=[20,40,60,80,100,120,160,240,400,800,1600]

def cases():
    out=[dict(name='openimages',data=O.DATA,source=O.CONTROL,pq=O.PQ,files=O.FILES,ids=O.OUT/'data',manifest=O.OUT/'manifest.txt',engine=O.OUT/'bin/multigraph')]
    for a in (.1,.3,.5,.7,.9):out.append(dict(name='laion-'+str(a),a=a,data=c.D,source=c.U/'stage2-v2/graphs/unified',pq=c.MG_RUN/'data',files=dict(qx=c.D/'query_img_emb.fvecs',qy=c.D/'query_text_emb.fvecs',alpha=alpha(a),gt=gt(a)),ids=c.MG_RUN/'data',manifest=c.E/'paper-priorities-20261114/runtime/no-sidecar/manifest-no-sidecar.txt',engine=OUT/'bin/multigraph32'))
    return out

def native(spec,index,out,L,cohort='report',threads=4,seconds=0,plain=False,gate=False):
    return [OUT/'bin'/('native-plain' if plain else 'native-counted'),str(index)+'/',spec['files']['qx'],spec['files']['qy'],spec['files']['alpha'],spec['files']['gt'],spec['ids']/('gate-ids.txt' if gate else cohort+'-ids.txt'),out,'-',threads,seconds]+list(L)

def compact(spec,jp,out,cohort='report',threads=4,seconds=0):
    return [spec['engine'],spec['manifest'],spec['files']['qx'],spec['files']['qy'],spec['files']['alpha'],spec['files']['gt'],spec['ids']/(cohort+'-ids.txt'),out,'ssd',threads,seconds,'pq',jp,'-',0]

def explicit_work(p):
    pages=p['coupled_backend_pages'];pq=p['pq_space_evals'];exact=p['exact_space_evals'];dims=1536 if p.get('dataset')=='openimages' else 1024
    return dict(pages=pages,compute=pq*64+exact*(dims//2),weighted_pages=24.8*pages)

def native_frontier(points):
    result={}
    for rule in P.RULES:
        selected=[]
        for target in P.TARGETS:
            possible=[p for p in points if p['recall']>=target]
            if possible:
                p=min(possible,key=lambda p:(explicit_work(p)[rule],p['L']))
                if p['L'] not in [v['L'] for v in selected]:selected.append(p)
        result[rule]=[p['L'] for p in sorted(selected,key=lambda p:(p['recall'],p['L']))]
    return result

def train(spec,index,root):
    nout=root/'native-tune.json'
    if not nout.exists():c.run(native(spec,index,nout,LEVELS,'tune'),'native-tune-'+spec['name'],timeout=1800)
    ns=c.load(nout)
    for p in ns:p['dataset']=spec['name']
    if max(p['peak_rss_kib'] for p in ns)*1024>BUDGET:raise RuntimeError('native training exceeds frozen 2GiB allowance')
    nf=native_frontier(ns);Ls=sorted(set(x for xs in nf.values() for x in xs));nr=root/'native-report.json'
    if not nr.exists():c.run(native(spec,index,nr,Ls),'native-report-work-'+spec['name'],timeout=1800)
    nb={p['L']:p for p in c.load(nr)};native_brackets={};chosen=set()
    for rule,order in nf.items():
        for target in (.9,.95):
            pair=P.crossing([nb[x] for x in order],target);key=rule+'-'+str(target)
            if pair is None:native_brackets[key]=dict(covered=False);continue
            a,b,w=pair;native_brackets[key]=dict(covered=True,L=[a['L'],b['L']],weight=w,target=target);chosen.update([a['L'],b['L']])
    # Proposed engine configs reuse that dataset's already frozen train curves.
    if spec['name']=='openimages':
        cr=O.OUT/'search/DEG_off';freeze=c.load(cr/'frozen.json');cb=c.load(cr/'brackets.json')
    else:
        a=spec['a'];curve=c.load(c.MG_RUN/'frozen.json')['curves'][str(a)+'-DEG_off'];freeze=curve
        records=c.load(c.MG_RUN/'report/summary.json')['records'];cb={}
        for rule in P.RULES:
            for target in (.9,.95):
                r=next(x for x in records if x['medium']=='ssd' and x['alpha']==a and x['method']=='DEG_off' and x['rule']==rule and x['target']==target)
                cb[rule+'-'+str(target)]=dict(covered=r['covered'],labels=r['repeats'][0]['labels'] if r['covered'] else [],target=target)
    names=sorted(set(x for b in cb.values() if b['covered'] for x in b['labels']));jobs(root/'compact-time.jobs',[freeze['jobs'][x] for x in names])
    c.save(root/'frozen.json',dict(native=native_brackets,native_training_sha256=c.sha(nout),native_L=sorted(chosen),compact=cb,compact_jobs=[freeze['jobs'][x] for x in names],memory_allowance_bytes=BUDGET,shared_cache_bytes=0,selection='same three deterministic work rules, each on its own tune cohort; no report timing selection'))
    return c.load(root/'frozen.json')

def gates(spec,index,root):
    path=root/'gate.json'
    if path.exists():
        if c.load(path)['status']!='PASS':raise RuntimeError('old native gate not passed')
        return
    ids=spec['ids']/'gate-ids.txt'
    # Laion's old gate contains 16 tune/report IDs; OI gate holds 16 report IDs.
    plain=root/'gate-plain.json';counted=root/'gate-counted.json'
    for p,disabled in [(plain,True),(counted,False)]:c.run(native(spec,index,p,[80,160],plain=disabled,gate=True),'native-gate-'+spec['name']+('-plain' if disabled else '-counted'),cpus='80-83',timeout=600)
    pa,co=c.load(plain),c.load(counted);x,y=(matrix(spec['data']/n) for n in ('base_img_emb.fvecs','base_text_emb.fvecs'));qx,qy=matrix(spec['files']['qx']),matrix(spec['files']['qy']);aw=matrix(spec['files']['alpha'])[:,0];error=0.
    for p,q in zip(pa,co):
        if p['queries']!=q['queries'] or p['recall']!=q['recall']:raise RuntimeError('native counting changed outputs')
        if q['pq_candidates']<=0 or q['pq_space_evals']!=2*q['pq_candidates'] or q['exact_space_evals']!=2*q['expanded']:raise RuntimeError('explicit counts missing')
        for row in q['queries']:
            qi=row['query_id'];ids=row['ids'];dx=np.linalg.norm(x[ids].astype(np.float64)-qx[qi].astype(np.float64),axis=1);dy=np.linalg.norm(y[ids].astype(np.float64)-qy[qi].astype(np.float64),axis=1);a=float(aw[qi]);ref=a*dx+(1-a)*dy;got=np.asarray(row['distances']);valid=ref>0
            if np.any(got[~valid]!=0):raise RuntimeError('native zero distance')
            if valid.any():error=max(error,float(np.max(np.abs(got[valid]-ref[valid])/ref[valid])))
    if error>5e-6:raise RuntimeError('native independent returned distance error')
    for threads in (1,32):
        out=root/('gate-counted-T'+str(threads)+'.json')
        c.run(native(spec,index,out,[80,160],threads=threads,gate=True),'native-thread-gate-'+spec['name']+'-T'+str(threads),cpus=','.join(map(str,list(range(72,72+min(24,threads)))+list(range(168,168+max(0,threads-24))))),timeout=600,extra=dict(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads)))
        if [p['queries'] for p in c.load(out)]!=[p['queries'] for p in co]:raise RuntimeError('native thread-count changed results')
    # Cap-only engine variants must retain parent outputs at 1/4/32 threads.
    jp=root/'gate-compact.jobs';jobs(jp,[dict(label='gate',method='DEG_off',L=80,kx=40,ky=0,weight=spec.get('a',.5))])
    reference=None
    for threads in (1,4,32):
        out=root/('gate-compact-T'+str(threads)+'.json')
        args=compact(spec,jp,out,threads=threads)
        args[6]=spec['ids']/'gate-ids.txt'
        c.run(args,'compact-thread-gate-'+spec['name']+'-T'+str(threads),cpus=','.join(map(str,list(range(72,72+min(24,threads)))+list(range(168,168+max(0,threads-24))))),timeout=600,extra=dict(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads)))
        observed=[p['queries'] for p in c.load(out)]
        if reference is None:reference=observed
        elif observed!=reference:raise RuntimeError('compact thread-count changed results')
    c.save(path,dict(status='PASS',counter_disabled_outputs_equal=True,reader_boundary_reconciled=True,max_return_relative_error=error,queries=len(co[0]['queries']),L=[80,160]))

def affinity(threads):
    if threads<=4:return measure.quiet()['choice']['cpus'][:threads]
    return list(range(72,72+min(24,threads)))+list(range(168,168+max(0,threads-24)))

def match_native(points,bracket):
    by={p['L']:p for p in points};a,b=[by[x] for x in bracket['L']];w=bracket['weight'];out={k:(1-w)*a[k]+w*b[k] for k in ('latency_us','qps','pq_candidates','pq_space_evals','exact_space_evals','expanded','coupled_backend_pages','coupled_reader_us')};out['raw_p95_us']={str(p['L']):p['p95_us'] for p in (a,b)};return out

def main():
    compile_native.main()
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');src=c.MG/'scripts/multigraph.cpp';text=src.read_bytes().decode('utf-8')
    if text.count('threads > 8')!=1:raise RuntimeError('compact thread cap hook')
    p=OUT/'build/multigraph32.cpp';p.write_bytes(text.replace('threads > 8','threads > 32').encode('utf-8'))
    c.run(['g++-11']+cfg['flags']+[p,c.E/'metric-consistent-20261111/bin/libpipeann-metric.a']+cfg['link']+['-o',OUT/'bin/multigraph32'],'native-compile-compact32',cpus='80,81',timeout=600)
    # OpenImages three-route executor also needs the same independent cap-only change.
    spec_list=cases();oi=spec_list[0];text=(O.OUT/'build/multigraph.cpp').read_bytes().decode('utf-8');p=OUT/'build/openimages32.cpp';p.write_bytes(text.replace('threads > 8','threads > 32').encode('utf-8'))
    c.run(['g++-11']+cfg['flags']+[p,c.E/'metric-consistent-20261111/bin/libpipeann-metric.a']+cfg['link']+['-o',OUT/'bin/openimages32'],'native-compile-openimages32',cpus='80,81',timeout=600);oi['engine']=OUT/'bin/openimages32'
    plans=[]
    for spec in spec_list:
        index=native_writer.write('openimages' if spec['name']=='openimages' else 'laion',spec['source'],spec['data'],spec['pq'])
        root=OUT/'search'/spec['name'];root.mkdir(parents=True,exist_ok=True);gates(spec,index,root);plan=train(spec,index,root);plans.append((spec,index,root,plan))
    pending=[]
    for rep in range(1,4):
      for spec,index,root,plan in plans[rep-1:]+plans[:rep-1]:
       for threads in THREADS:
        for method in (['native','compact'] if rep%2 else ['compact','native']):
            tag=method+'-T'+str(threads)+'-rep'+str(rep);saved=root/(tag+'.json')
            if saved.exists():continue
            out=root/(tag+'-raw.json');cpus=affinity(threads);c.save(root/(tag+'-affinity.json'),dict(cpus=cpus,numa=3,total_memory_allowance=BUDGET,shared_cache=0))
            args=native(spec,index,out,plan['native_L'],threads=threads,seconds=10) if method=='native' else compact(spec,root/'compact-time.jobs',out,threads=threads,seconds=10)
            data=measure.run(args,'compare-'+spec['name']+'-'+tag,out,cpus,extra=dict(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads)))
            if data and all(p['peak_rss_kib']*1024<=BUDGET for p in data['points']):c.save(saved,data)
            else:pending.append([spec['name'],method,threads,rep,'noise_or_memory_budget'])
    rows=[]
    for spec,index,root,plan in plans:
      for threads in THREADS:
       for key,b in plan['native'].items():
        cb=plan['compact'][key]
        if not b['covered'] or not cb['covered']:rows.append(dict(dataset=spec['name'],threads=threads,case=key,covered=False));continue
        reps=[]
        for rep in range(1,4):
            a=root/('native-T'+str(threads)+'-rep'+str(rep)+'.json');d=root/('compact-T'+str(threads)+'-rep'+str(rep)+'.json')
            if not a.exists() or not d.exists():continue
            n=match_native(c.load(a)['points'],b);by={p['label']:p for p in c.load(d)['points']};v=P.matched([by[x] for x in cb['labels']],cb['target'])
            if not v['covered']:raise RuntimeError('compact timing recall changed')
            reps.append(dict(native=n,compact=v,speedup=n['latency_us']/v['latency_us']))
        rows.append(dict(dataset=spec['name'],threads=threads,case=key,covered=len(reps)==3,repeats=reps))
    c.save(OUT/'summary.json',dict(results=rows,pending=pending,memory='same 2GiB total allowance, no shared cache; actual RSS recorded; not equal resident-byte claim',scope='corrected objective native TXY, compact original-order T|XY; no Starling/cache package claim'))
    return dict(complete=not pending,summary=str(OUT/'summary.json'))
