"""Separate corrected-author statistical gate; then source-identical A/B work."""
import json,os,struct,sys,time
from pathlib import Path
import numpy as np
import common as c
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P

F=c.RUN/'function-v2'
S=c.RUN/'statistical-v2'
AB=c.RUN/'formal-ab-v2'
WEIGHTS=(.1,.3,.5,.7,.9)
LEVELS=[40,80,120,160,240,400,800,1600]
BUDGETS=[10,20,40,80,160,320]
SMALL_LEVELS=[10,20,40,60,80,100,120,160,240,400,800,1600]

def matrix(p,dtype='<f4'):
    with Path(p).open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    return np.memmap(str(p),dtype=dtype,mode='r',offset=8,shape=(n,d))
def alpha(a):return (c.MG_RUN/'data' if a in (.3,.7) else c.D)/('query-alpha-'+str(a)+'.bin')
def gt(a):return (c.MG_RUN/'data' if a in (.3,.7) else c.D)/('gt-alpha-'+str(a)+'.bin')
def jobs(path,values):
    Path(path).write_text(''.join('{label} {method} {L} {kx} {ky} {weight}\n'.format(**v) for v in values))
def command(manifest,a,jp,out,mode='dram',nav='pq',cohort='report',seconds=0,small=False,trace='-'):
    return [c.MG_RUN/'bin/multigraph',manifest,c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',alpha(a),S/('gt-'+str(a)+'.bin') if small else gt(a),c.MG_RUN/'data'/(cohort+'-ids.txt'),out,mode,4,seconds,nav,jp,trace,1 if trace!='-' else 0]
def run_once(args,label,out,timeout=3600,cpus='72-75',extra=None):
    sig=dict(args=list(map(str,args)),extra=extra or {})
    stamp=Path(str(out)+'.identity.json')
    if stamp.exists():
        old=c.load(stamp)
        if old['signature']!=sig or old['sha256']!=c.sha(out):raise RuntimeError('completed artifact identity changed '+str(out))
        return
    c.run(args,label,cpus=cpus,timeout=timeout,extra=extra)
    c.save(stamp,dict(signature=sig,sha256=c.sha(out)))
def audit(raw,output,packed=None):
    args=[F/'bin/graph-audit',raw]+([packed/'compact.meta',packed/'compact.topo',packed/'ranges.bin'] if packed else [])
    log=c.run(args,'graph-audit-'+output.parent.name+'-'+output.stem,cpus='80,81',timeout=3600)
    value=json.loads(log.read_text());c.save(output,value);return value
def pack(raw,meta,folder,entries):
    folder.mkdir(parents=True,exist_ok=True);stamp=folder/'pack-identity.json';sig=dict(raw=c.sha(raw),meta=c.sha(meta),entries=c.sha(entries),packer=c.sha(c.MG_RUN/'bin/pack'))
    if stamp.exists():
        old=c.load(stamp)
        if old['source']!=sig or any(c.sha(folder/n)!=h for n,h in old['files'].items()):raise RuntimeError('packed identity changed')
        return old
    log=c.run([c.MG_RUN/'bin/pack',meta,raw,folder/'compact.meta',folder/'compact.topo',folder/'ranges.bin',entries],'pack-ab-'+folder.name,cpus='80,81',timeout=3600)
    value=dict(source=sig,stats=json.loads(log.read_text()),files={n:c.sha(folder/n) for n in ('compact.meta','compact.topo','ranges.bin')})
    audit(raw,folder/'export-audit.json',folder);c.save(stamp,value);return value
def manifest(path,folder,coordinate,n,side=True):
    slots=(folder/'ranges.bin').stat().st_size//(n*80) if side else 0
    path.write_text('HVS_MULTI_V1 '+str(n)+' 512 512 1\n'+str(coordinate)+'\n'+str(c.MG_RUN/'data')+'/\nu '+str(folder/'compact.meta')+' '+str(folder/'compact.topo')+' '+(str(folder/'ranges.bin') if side else '-')+' '+str(slots)+'\n')
def prepare_small():
    S.mkdir(parents=True,exist_ok=True)
    source=c.RUN/'data-10k';x=matrix(source/'base_img_emb.fvecs').astype(np.float64);y=matrix(source/'base_text_emb.fvecs').astype(np.float64)
    coords=S/'xy.pages';packed=np.concatenate([x.astype('<f4'),y.astype('<f4')],axis=1);packed.tofile(str(coords))
    padding=(-coords.stat().st_size)%8192
    if padding:
        with coords.open('ab') as f:f.write(bytes(padding))
    qx=matrix(c.D/'query_img_emb.fvecs');qy=matrix(c.D/'query_text_emb.fvecs');truth={a:np.empty((len(qx),10),dtype='<u4') for a in WEIGHTS};ids=np.arange(len(x))
    for qi in range(len(qx)):
        dx=x-qx[qi].astype(np.float64);dy=y-qy[qi].astype(np.float64);dx=np.sqrt(np.einsum('ij,ij->i',dx,dx));dy=np.sqrt(np.einsum('ij,ij->i',dy,dy))
        for a in WEIGHTS:
            aa=float(np.float32(a));truth[a][qi]=np.lexsort((ids,aa*dx+(1-aa)*dy))[:10]
    for a,v in truth.items():
        with (S/('gt-'+str(a)+'.bin')).open('wb') as f:f.write(struct.pack('<II',len(qx),10));v.tofile(f)
    c.save(S/'data-identity.json',dict(source=c.sha(source/'identity.json'),coords=c.sha(coords),queries=[c.sha(c.D/n) for n in ('query_img_emb.fvecs','query_text_emb.fvecs')],gt={str(a):c.sha(S/('gt-'+str(a)+'.bin')) for a in WEIGHTS},objective='float64 alpha32*sqrt(sum(delta_x^2))+(1-alpha32)*sqrt(sum(delta_y^2)); full10k scan, stable id ties'))

def statistical_gate():
    S.mkdir(parents=True,exist_ok=True);proof=S/'gate.json'
    if proof.exists():
        value=c.load(proof)
        if value['status']!='PASS':raise RuntimeError('prior small statistical gate did not pass')
        return value
    data=c.RUN/'data-10k';author=S/'author-fixed';author.mkdir(exist_ok=True);raw=author/'float-graph.bin'
    run_once([F/'bin/build-author-fixed',data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',raw],'small-author-fixed',raw,cpus='80',timeout=3600,extra=dict(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1'))
    stats={}
    roots={'B':F/'capture-graph','author-fixed':author}
    entries=F/'capture-graph/full.meta'
    for name,root in roots.items():
        st=audit(root/'float-graph.bin',root/'structure-stats.json');stats[name]=st
        # Shared frozen B entries isolate graph quality from entry policy.
        meta=bytearray(entries.read_bytes());struct.pack_into('<I',meta,16,st['max_quantized_slots']);struct.pack_into('<Q',meta,20,8192//(4+40*(4+2*st['max_quantized_slots'])))
        m=root/'source-for-pack.meta';m.write_bytes(meta);out=S/'packed'/name;pack(root/'float-graph.bin',m,out,entries);manifest(S/('manifest-'+name+'.txt'),out,S/'xy.pages',10000)
    a,b=stats['B'],stats['author-fixed'];mean_diff=abs(b['mean_degree']-a['mean_degree'])/a['mean_degree'];degree_tvd=sum(abs(x-y) for x,y in zip(a['degree_histogram'],b['degree_histogram']))/20000
    active=[dict(alpha=w,B=x,author_fixed=y,absolute_percentage_points=100*abs(x-y),relative_difference=abs(x-y)/x if x else None) for w,x,y in zip(WEIGHTS,a['active_fractions'],b['active_fractions'])]
    c.save(S/'graph-comparison.json',dict(stats=stats,mean_degree_relative_difference=mean_diff,degree_distribution_TVD=degree_tvd,active=active,entry_policy='both use immutable B10k entries; author-native graph unchanged'))
    if not (S/'data-identity.json').exists():
        run_once([sys.executable,c.ROOT/'scripts/stage_checks.py','prepare-small'],'small-fullscan-GT',S/'data-identity.json',cpus='80,81',timeout=3600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    rows=[]
    for w in WEIGHTS:
      for name in roots:
        root=S/'search'/str(w)/name;root.mkdir(parents=True,exist_ok=True);jp=root/'jobs.txt';values=[dict(label='exact-L'+str(L),method='DEG_off',L=L,kx=10,ky=0,weight=w) for L in SMALL_LEVELS];jobs(jp,values)
        for cohort in ('tune','report'):
            out=root/(cohort+'.json');run_once(command(S/('manifest-'+name+'.txt'),w,jp,out,nav='exact',cohort=cohort,small=True),'small-exact-'+str(w)+'-'+name+'-'+cohort,out,timeout=3600)
        points=c.load(root/'report.json');match=P.matched(points,.95)
        rows.append(dict(alpha=w,graph=name,matched=match,curve=[dict(L=p['L'],recall=p['recall'],expanded=sum(g['expanded'] for g in p['graphs']),exact_space_evals=sum(g['exact_space_evals'] for g in p['graphs'])) for p in points]))
    diffs=[]
    for w in WEIGHTS:
        pair=[next(r for r in rows if r['alpha']==w and r['graph']==n)['matched'] for n in roots]
        if all(v['covered'] for v in pair):diffs.append(dict(alpha=w,covered=True,B=pair[0]['expanded'],author_fixed=pair[1]['expanded'],relative_difference=abs(pair[1]['expanded']-pair[0]['expanded'])/pair[0]['expanded']))
        else:diffs.append(dict(alpha=w,covered=False))
    passed=mean_diff<=.05 and all(r['covered'] and r['relative_difference']<=.05 for r in diffs) and all(r['absolute_percentage_points']<=5 for r in active)
    result=dict(status='PASS' if passed else 'STOP_AT_STATISTICAL_GATE',mean_degree_relative_difference=mean_diff,degree_distribution_TVD=degree_tvd,active=active,expansion_comparison=diffs,curves=rows,threshold=.05,source='independent corrected-author code, same input/shared B entries/exact pointer DRAM navigation')
    c.save(proof,result)
    if not passed:raise RuntimeError('small statistical gate failed; no2M builds')
    return result

def build_formal(mode,rep):
    root=AB/'graphs'/(mode+str(rep));root.mkdir(parents=True,exist_ok=True);binary=c.RUN/'bin'/('build-'+mode);out=root/'float-graph.bin';stamp=root/'build.json'
    if c.sha(binary)!=c.load(c.RUN/'validation/build.json')['binaries'][mode]['binary']:raise RuntimeError('formal A/B binary changed')
    sig=dict(binary=c.sha(binary),data=c.sha(c.D/'subset.json'),R=40,L=200,C=750,threads=8,numa=3)
    if stamp.exists():
        old=c.load(stamp)
        if old['signature']!=sig or old['graph_sha256']!=c.sha(out):raise RuntimeError('formal graph changed')
        return root
    c.capacity(5*2**30,'Laion2M A/B raw graph and validated compact topology/sidecar '+root.name)
    start=time.time();c.run([binary,'dual',c.D/'base_img_emb.fvecs',c.D/'base_text_emb.fvecs',2000000,512,512,8,root],'formal-build-'+root.name,cpus='72-79',timeout=12*3600,extra=dict(OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8',HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1'))
    c.save(stamp,dict(signature=sig,graph_sha256=c.sha(out),wall_seconds=time.time()-start,reverse_insert_stats=c.load(root/'reverse-insert-stats.json')))
    return root

def formal_dram():
    AB.mkdir(parents=True,exist_ok=True);coordinate=c.MG_RUN/'data/xy.pages';entries=c.U/'stage2-v2/graphs/unified/full.meta';all_graphs=[]
    for rep in (1,2):
      for mode in ('A','B'):
        root=build_formal(mode,rep);packed=AB/'packed'/root.name;pack(root/'float-graph.bin',root/'full.meta',packed,entries);st=audit(root/'float-graph.bin',root/'stats.json',packed)
        mf=AB/('manifest-'+root.name+'.txt');manifest(mf,packed,coordinate,2000000);all_graphs.append((mode,rep,root,mf))
    plans=[];missing=[]
    for mode,rep,root,mf in all_graphs:
      for w in WEIGHTS:
       for method in ('DEG_off','DEG_on'):
        dest=AB/'work'/root.name/str(w)/method;dest.mkdir(parents=True,exist_ok=True);jp=dest/'tune.jobs'
        values=[dict(label=method+'-L'+str(L)+'-r'+str(r),method=method,L=L,kx=r,ky=0,weight=w) for L in LEVELS for r in BUDGETS if r<=L];jobs(jp,values);out=dest/'tune.json'
        run_once(command(mf,w,jp,out,cohort='tune'),'AB-tune-'+root.name+'-'+str(w)+'-'+method,out,timeout=3600)
        frontiers=P.frontiers(c.load(out));names=sorted({v for front in frontiers.values() for v in front});by={v['label']:v for v in values};frozen=dict(frontiers=frontiers,jobs={n:by[n] for n in names},tune_sha256=c.sha(out),manifest_sha256=c.sha(mf),selection='three work rules, tune only; same loaded sidecar for on/off controlled test')
        c.save(dest/'frozen.json',frozen);report_jobs=dest/'report.jobs';jobs(report_jobs,[by[n] for n in names]);reports=[]
        for repeat in range(1,4):
            out=dest/('report-rep'+str(repeat)+'.json');run_once(command(mf,w,report_jobs,out),'AB-report-'+root.name+'-'+str(w)+'-'+method+'-'+str(repeat),out,timeout=3600);reports.append(c.load(out))
        rows=[]
        for rule,order in frontiers.items():
            matched=[]
            for points in reports:
                pp={v['label']:v for v in points};v=P.matched([pp[n] for n in order],.95)
                if v['covered']:
                    pair=P.crossing([pp[n] for n in order],.95);a,b,t=pair;v['L']=(1-t)*a['L']+t*b['L'];v['r']=(1-t)*a['kx']+t*b['kx']
                matched.append(v)
            covered=all(v['covered'] for v in matched)
            if not covered:missing.append([root.name,w,method,rule,'95% not bracketed; no extrapolation'])
            rows.append(dict(rule=rule,covered=covered,repeats=matched))
        c.save(dest/'matched.json',rows);plans.append(dict(mode=mode,graph_rep=rep,graph=root.name,alpha=w,method=method,path=str(dest),results=rows))
    variation=[]
    for w in WEIGHTS:
      for method in ('DEG_off','DEG_on'):
       for rule in P.RULES:
        row=dict(alpha=w,method=method,rule=rule,metrics={});by={}
        for mode in ('A','B'):
            values=[]
            for rep in (1,2):
                p=next(x for x in plans if x['mode']==mode and x['graph_rep']==rep and x['alpha']==w and x['method']==method);r=next(x for x in p['results'] if x['rule']==rule)
                if not r['covered']:break
                values.append({k:sum(v[k] for v in r['repeats'])/3 for k in ('expanded','pq_space_evals','L','reranked','topology_pages','coordinate_pages')})
            if len(values)==2:by[mode]=values
        if len(by)==2:
            for key in by['A'][0]:
                means={m:(by[m][0][key]+by[m][1][key])/2 for m in by};noise={m:abs(by[m][0][key]-by[m][1][key])/means[m] if means[m] else 0 for m in by};delta=(means['B']-means['A'])/means['A'] if means['A'] else 0;row['metrics'][key]=dict(A=means['A'],B=means['B'],relative_B_change=delta,within_version_variation=noise,meaningful=abs(delta)>=.1 and abs(delta)>2*max(noise.values()))
        variation.append(row)
    result=dict(status='COMPLETE' if not missing else 'COMPLETE_WITH_UNCOVERED_POINTS',records=plans,build_variation=variation,missing=missing,scope='DRAM work only; 2graphs/version,3query repeats are not independent builds')
    c.save(AB/'dram-summary.json',result)
    if missing:raise RuntimeError('95% work coverage incomplete; retain missing points and stop dependent SSD')
    return result

def ssd_stage():
    # Use the same measured-window guard as the completed native sweep.
    sys.path.insert(0,str(c.E/'edbt-evidence-20261115/scripts'));import measure
    rows=[];pending=[]
    for repeat in range(1,4):
     for w in (.1,.5):
      for mode in ('A','B') if repeat%2 else ('B','A'):
       name=mode+'1';mf=AB/('manifest-'+name+'.txt')
       for method in ('DEG_off','DEG_on') if repeat%2 else ('DEG_on','DEG_off'):
        dest=AB/'work'/name/str(w)/method;frozen=c.load(dest/'frozen.json');chosen=set();brackets={};report=c.load(dest/'report-rep1.json');by={p['label']:p for p in report}
        for rule,order in frozen['frontiers'].items():
            v=P.matched([by[n] for n in order],.95);brackets[rule]=v
            if v['covered']:chosen.update(v['labels'])
        folder=AB/'ssd'/name/str(w)/method;folder.mkdir(parents=True,exist_ok=True);jp=folder/'frozen.jobs';jobs(jp,[frozen['jobs'][n] for n in sorted(chosen)])
        saved=folder/('rep'+str(repeat)+'.json');out=folder/('raw-rep'+str(repeat)+'.json')
        # SSD/DRAM trace parity before any timed point for this input.
        gate=folder/'medium-gate.json'
        if not gate.exists():
            points={}
            for medium in ('dram','ssd'):
                proof=folder/('gate-'+medium+'.json');prefix=folder/('trace-'+medium);c.run(command(mf,w,jp,proof,mode=medium,trace=prefix),'AB-medium-gate-'+name+'-'+str(w)+'-'+method+'-'+medium,cpus='80-83',timeout=600);points[medium]=c.load(proof)
            good=[x['queries'] for x in points['dram']]==[x['queries'] for x in points['ssd']]
            for label in chosen:
                for suffix in ('','.ids','.dists','.legacy'):good=good and c.sha(folder/('trace-dram.'+label+suffix))==c.sha(folder/('trace-ssd.'+label+suffix))
            c.save(gate,dict(status='PASS' if good else 'FAIL',labels=sorted(chosen)))
            if not good:raise RuntimeError('AB medium gate failed')
        elif c.load(gate)['status']!='PASS':raise RuntimeError('prior AB medium gate failed')
        if saved.exists():data=c.load(saved)
        else:
            cpus=measure.quiet()['choice']['cpus'];data=measure.run(command(mf,w,jp,out,mode='ssd',seconds=10),'AB-ssd-'+name+'-'+str(w)+'-'+method+'-rep'+str(repeat),out,cpus)
            if data:c.save(saved,data)
            else:pending.append([name,w,method,repeat]);continue
        pp={x['label']:x for x in data['points']}
        for rule,v in brackets.items():
            matched=P.matched([pp[n] for n in v['labels']],.95) if v['covered'] else v
            if not matched['covered']:raise RuntimeError('deterministic SSD recall changed')
            rows.append(dict(version=mode,alpha=w,method=method,repeat=repeat,rule=rule,matched=matched))
    c.save(AB/'ssd-summary.json',dict(status='COMPLETE' if not pending else 'COMPLETE_WITH_PENDING',records=rows,pending=pending,source_graphs='A1 and B1 only; within-build variation from DRAM stage remains separately reported'))
    return not pending

if __name__=='__main__':
    if len(sys.argv)==2 and sys.argv[1]=='prepare-small':prepare_small()
    else:raise SystemExit('prepare-small only; main queue imports the gate/stage functions')
