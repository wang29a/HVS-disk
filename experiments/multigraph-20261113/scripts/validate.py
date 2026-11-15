"""Functional gates only: no result from here is a latency benchmark."""
import json
from pathlib import Path
import struct
import sys
import numpy as np
import common as c
import prepare_inputs as inp

def write_jobs(path,jobs):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(''.join('{label} {method} {L} {kx} {ky} {weight}\n'.format(**j) for j in jobs))
def command(manifest,alpha,gt,ids,out,mode,jobs,nav='pq',trace='-',seconds=0,threads=4,audit=0,qroot=None):
    qroot=qroot or c.D
    return [c.RUN/'bin/multigraph',manifest,qroot/'query_img_emb.fvecs',qroot/'query_text_emb.fvecs',alpha,gt,ids,out,mode,threads,seconds,nav,jobs,trace,audit]
def byte_equal(paths):
    for suffix in ('','.ids','.dists'):
        values=[Path(str(p)+suffix).read_bytes() for p in paths]
        if any(v!=values[0] for v in values[1:]):raise RuntimeError('trajectory/result byte gate failed '+str(paths)+' suffix='+suffix)
def distances(output,base,queries,alpha):
    x,y=inp.p.matrix(base/'base_img_emb.fvecs'),inp.p.matrix(base/'base_text_emb.fvecs')
    qx,qy=inp.p.matrix(queries/'query_img_emb.fvecs'),inp.p.matrix(queries/'query_text_emb.fvecs');a=inp.p.matrix(alpha)[:,0]
    worst=0.
    for point in c.load(output):
        for row in point['queries']:
            i=row['query_id'];ids=np.array(row['ids'],dtype=np.uint32)
            dx=np.linalg.norm(np.asarray(x[ids],dtype=np.float64)-np.asarray(qx[i],dtype=np.float64),axis=1)
            dy=np.linalg.norm(np.asarray(y[ids],dtype=np.float64)-np.asarray(qy[i],dtype=np.float64),axis=1)
            af=float(a[i]);ref=af*dx+(1-af)*dy;got=np.asarray(row['distances'],dtype=np.float64)
            if np.any(got[ref==0]!=0):raise RuntimeError('independent zero-distance gate')
            if np.any(ref>0):worst=max(worst,float(np.max(np.abs(got[ref>0]-ref[ref>0])/ref[ref>0])))
    if worst>5e-6:raise RuntimeError('independent float64 max relative error '+str(worst))
    return worst

def centre(manifest,root=None,source=None,ids=None):
    root=root or c.D;source=source or c.U/'stage2-v2/graphs/fixed-0.5';ids=ids or c.RUN/'data/gate-ids.txt'
    out=c.RUN/'validation/centre';out.mkdir(exist_ok=True)
    alpha=root/'query-alpha-0.5.bin';gt=root/'gt-alpha-0.5.bin'
    jobs=[dict(label='centre-L'+str(L),method='Or_ideal',L=L,kx=10,ky=0,weight=.5) for L in (40,80,160)]
    jobpath=out/'jobs.txt';write_jobs(jobpath,jobs)
    threads=4 if root==c.D else 2;cpus='72-75' if root==c.D else '80,81'
    c.run(command(manifest,alpha,gt,ids,out/'new.json','dram',jobpath,nav='exact',trace=out/'new',qroot=root,threads=threads),'gate-centre-new',cpus=cpus,timeout=600)
    old=[c.U/'stage2/bin/pointer-search',source/'full.meta',source/'float-graph.bin',root/'base_img_emb.fvecs',root/'base_text_emb.fvecs',root/'query_img_emb.fvecs',root/'query_text_emb.fvecs',alpha,gt,ids,out/'old.json',threads,0,0,40,80,160]
    # Both use the existing experiment's common unified entry set.
    meta=c.U/'stage2-v2/graphs/unified/full.meta' if source==c.U/'stage2-v2/graphs/fixed-0.5' else source/'full.meta'
    c.run(old,'gate-centre-old',cpus=cpus,timeout=600,extra=dict(PTR_TRACE=str(out/'old'),PTR_ENTRY_META=str(meta)))
    for j in jobs:
        L=j['L']
        for suffix,newsuffix in [('', '.legacy'),('.ids','.ids'),('.dists','.dists')]:
            a=Path(str(out/'old')+'.L'+str(L)+suffix);b=Path(str(out/'new')+'.'+j['label']+newsuffix)
            if a.read_bytes()!=b.read_bytes():raise RuntimeError('centre old/new byte parity L='+str(L)+' '+suffix)
    return dict(status='PASS',L=[40,80,160],queries=len(Path(ids).read_text().split()),mode='exact pointer navigation; same existing graph and entries',source_graph_sha256=c.sha(source/'float-graph.bin'))

def coverage():
    """Amended gate: only PQ-exhaustive B versus graph-search C can stop us."""
    path=c.RUN/'three-layer-20261114/summary.json'
    if not path.exists():raise RuntimeError('run the frozen three-layer diagnostic first')
    result=c.load(path)
    if result['status']!='PASS':raise RuntimeError('three-layer B/C gate failed; see '+str(path))
    identity=result['identity']
    if identity['original_binary_sha256']!=c.sha(c.RUN/'bin/multigraph'):raise RuntimeError('three-layer executor identity changed')
    for name,h in identity['packed_graphs'].items():
        if c.sha(c.RUN/'packed'/name/'identity.json')!=h:raise RuntimeError('three-layer graph identity changed '+name)
    return result['groups']

def real():
    manifest=c.RUN/'manifest.txt';folder=c.RUN/'validation/real';folder.mkdir(exist_ok=True)
    result=dict(status='RUNNING',centre=centre(manifest),cases=[],maximum_relative_error=0.)
    for a in inp.WEIGHTS:
        for method in ('M_equal','M_weighted','Or_ideal','F','DEG_on','DEG_off'):
            w=c.load(c.RUN/'data/contribution.json')[str(a)]['tune']['mean']
            from protocol import allocations
            x,y=allocations(20,w) if method=='M_weighted' else (20,20) if method=='M_equal' else (20,0)
            js=[dict(label=method+'-L'+str(L),method=method,L=L,kx=x,ky=y,weight=a) for L in (80,160)]
            label='a'+str(a)+'-'+method;jp=folder/(label+'.jobs');write_jobs(jp,js);tr=[]
            for medium in ('dram','ssd','topo'):
                prefix=folder/(label+'-'+medium);out=Path(str(prefix)+'.json')
                c.run(command(manifest,inp.alphapath(a),inp.gtpath(a),c.RUN/'data/gate-ids.txt',out,medium,jp,trace=prefix,audit=0),label+'-'+medium,timeout=600)
                result['maximum_relative_error']=max(result['maximum_relative_error'],distances(out,c.D,c.D,inp.alphapath(a)));tr.append(prefix)
            for j in js:byte_equal([Path(str(p)+'.'+j['label']) for p in tr])
            result['cases'].append(dict(alpha=a,method=method,L=[80,160],media=['dram','ssd','topo'],byte_equal=True))
            c.save(folder/'progress.json',result)
    result.update(status='PASS',coverage=coverage(),binary_sha256=c.sha(c.RUN/'bin/multigraph'),manifest_sha256=c.sha(manifest))
    c.save(c.RUN/'validation/real-gates.json',result);return result

def tiny():
    t=c.RUN/'tiny';t.mkdir(exist_ok=True);n=256;q=12
    for name in ('base_img_emb.fvecs','base_text_emb.fvecs'):
        inp.p.write_matrix(t/name,inp.p.matrix(c.D/name)[:n],'<f4')
    # Queries are original vectors plus tiny perturbations; include exact ties.
    for base,query in [('base_img_emb.fvecs','query_img_emb.fvecs'),('base_text_emb.fvecs','query_text_emb.fvecs')]:
        data=np.array(inp.p.matrix(t/base)[np.arange(q)*7],dtype=np.float32);data[1:]+=np.float32(.001);inp.p.write_matrix(t/query,data,'<f4')
    x,y,qx,qy=(inp.p.matrix(t/name) for name in inp.p.NAMES)
    weights=[0.,.1,.5,.9,1.];gt,_=inp.p.exact_batch(x,y,qx.astype(np.float64),qy.astype(np.float64),[float(np.float32(a)) for a in weights],10,32768)
    for k,a in enumerate(weights):inp.p.write_matrix(t/('gt-alpha-'+str(a)+'.bin'),gt[k],'<u4');inp.p.write_matrix(t/('query-alpha-'+str(a)+'.bin'),np.full((q,1),np.float32(a)),'<f4')
    (t/'ids.txt').write_text('\n'.join(map(str,range(q)))+'\n')
    combined=np.concatenate((x,y),axis=1).astype('<f4');combined.tofile(str(t/'xy.pages'))
    # Reuse full-dataset trained books and original-ID codes for these rows.
    srcids=inp.p.matrix(c.D/'source_ids.bin','<u4')[:n,0]
    for space in ('emb','loc'):
        source=inp.PQ/(space+'_pq_compressed.bin')
        with source.open('rb') as f:rows,d=struct.unpack('<II',f.read(8))
        codes=np.memmap(str(source),dtype='u1',mode='r',offset=8,shape=(rows,d))
        with (t/source.name).open('wb') as f:f.write(struct.pack('<II',n,d));np.asarray(codes[srcids]).tofile(f)
        pivot=t/(space+'_pq_pivots.bin')
        if not pivot.exists():pivot.symlink_to(inp.PQ/pivot.name)
    for name,mode in [('fixed','fixed'),('unified','unified'),('x','single'),('y','single')]:
        folder=t/name;folder.mkdir(exist_ok=True)
        binary=c.U/'stage2-v2/bin'/('build-'+mode) if mode!='single' else c.RUN/'bin/build-vamana'
        base=t/('base_text_emb.fvecs' if name=='y' else 'base_img_emb.fvecs')
        c.run([binary,'dual' if mode!='single' else 'single',base,t/'base_text_emb.fvecs',n,512,512,2,folder],'tiny-build-'+name,cpus='80,81',timeout=180,extra=dict(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',HVS_FIXED_ALPHA='0.5',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
        if mode=='single' and not (folder/'full.meta').exists():
            (folder/'full.meta').symlink_to('compact.meta')
        # The fixture retains native centre entries for comparison with old.
        side=folder/'ranges.bin' if name=='unified' else '-'
        c.run([c.RUN/'bin/pack',folder/'full.meta',folder/'float-graph.bin',folder/'compact.meta',folder/'compact.topo',side,folder/'full.meta'],'tiny-pack-'+name,cpus='80,81',timeout=60)
    lines=['HVS_MULTI_V1 %d 512 512 8'%n,str(t/'xy.pages'),str(t)+'/']
    for name in ('x','y','a01','a03','a05','a07','a09','u'):
        folder=t/('unified' if name=='u' else name if name in ('x','y') else 'fixed')
        slots=struct.unpack('<I',(folder/'full.meta').read_bytes()[16:20])[0] if name=='u' else 0
        lines.append(' '.join(map(str,[name,folder/'compact.meta',folder/'compact.topo',folder/'ranges.bin' if name=='u' else '-',slots])))
    manifest=t/'manifest.txt';manifest.write_text('\n'.join(lines)+'\n');proof=dict(status='RUNNING',cases=[])
    for a in weights:
        methods=['M_equal','M_weighted','M_endpoint','Or_nearest','F','DEG_on','DEG_off']+(['Or_ideal'] if a in (.1,.5,.9) else [])
        for method in methods:
            js=[dict(label=method,method=method,L=160,kx=40,ky=40 if method.startswith('M_') else 0,weight=a)];jp=t/'jobs.txt';write_jobs(jp,js);prefix=[]
            for medium in ('dram','ssd','topo'):
                label='tiny-'+str(a)+'-'+method+'-'+medium;tr=t/label;out=t/(label+'.json')
                c.run(command(manifest,t/('query-alpha-'+str(a)+'.bin'),t/('gt-alpha-'+str(a)+'.bin'),t/'ids.txt',out,medium,jp,trace=tr,threads=2,qroot=t),'check-'+label,cpus='80,81',timeout=120)
                distances(out,t,t,t/('query-alpha-'+str(a)+'.bin'));prefix.append(Path(str(tr)+'.'+method))
            byte_equal(prefix);proof['cases'].append(dict(alpha=a,method=method,byte_equal=True))
    proof['centre']=centre(manifest,root=t,source=t/'fixed',ids=t/'ids.txt')
    proof.update(status='PASS',binary_sha256=c.sha(c.RUN/'bin/multigraph'),fixture_nodes=n,queries=q)
    c.save(c.RUN/'validation/tiny-gates.json',proof)
if __name__=='__main__':
    try:tiny() if sys.argv[1]=='tiny' else real()
    except Exception as e:c.save(c.RUN/'validation/failure.json',dict(status='FAIL',error=str(e)));raise
