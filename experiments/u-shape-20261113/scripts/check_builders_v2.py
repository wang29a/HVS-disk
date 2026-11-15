#!/usr/bin/env python3
"""Actual C++ member tests and full tiny-build/exhaustive-query integration."""
import hashlib,json,os,pathlib,struct,subprocess
import numpy as np
R=pathlib.Path(__file__).resolve().parent.parent;O=R/'stage2-v2';OLD=R/'stage2'
def main():
    proof=json.loads((O/'validation/semantic-proof.json').read_text())
    env=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    with (O/'logs/integration.log').open('w') as log:
        def run(args,extra=None):
            e=dict(env);e.update(extra or {});subprocess.check_call(list(map(str,args)),env=e,stdout=log,stderr=subprocess.STDOUT,timeout=180)
        for cmd in proof['commands']:
            if any(str(a).endswith('/test_actual_pruner.cpp') for a in cmd):run(cmd)
        for mode in ('unified','fixed'):
            for a in (['0','0.1','0.3','0.5','0.7','0.9','1'] if mode=='fixed' else ['0.5']):
                run(['numactl','--all','--physcpubind=80,81','--membind=3',O/'bin'/('test-actual-'+mode),mode,a],{'HVS_FIXED_ALPHA':a})
        root=O/'validation/build-fixtures';root.mkdir(exist_ok=True)
        source=OLD/'validation/fixed-fixture'
        def matrix(path):
            with path.open('rb') as f:
                n,d=struct.unpack('<II',f.read(8));return np.fromfile(f,dtype='<f4').reshape(n,d)
        def save(path,data):
            with path.open('wb') as f:f.write(struct.pack('<II',*data.shape));data.tofile(f)
        x=matrix(source/'x.bin');y=matrix(source/'y.bin');n,dim=x.shape;q=32
        save(root/'qx.bin',x[:q]);save(root/'qy.bin',y[:q])
        results=[]
        for mode,a in [('unified','0.5'),('fixed','0'),('fixed','0.1'),('fixed','0.5'),('fixed','0.9'),('fixed','1')]:
            folder=root/(mode+'-'+a);folder.mkdir(exist_ok=True)
            run(['numactl','--all','--physcpubind=80,81','--membind=3',O/'bin'/('build-'+mode),'dual',source/'x.bin',source/'y.bin',n,dim,dim,2,folder],dict(HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1',HVS_FIXED_ALPHA=a))
            audit=json.loads((folder/'graph-audit.json').read_text());assert audit['nodes']==n and audit['quantized_lost_grid_points']==0
            score=float(np.float32(a))*np.sqrt(((x[:q,None].astype(np.float64)-x[None].astype(np.float64))**2).sum(2))+(1-float(np.float32(a)))*np.sqrt(((y[:q,None].astype(np.float64)-y[None].astype(np.float64))**2).sum(2))
            gt=np.argsort(score,axis=1,kind='mergesort')[:,:10].astype('<u4');save(folder/'gt.bin',gt)
            save(folder/'alpha.bin',np.full((q,1),float(a),dtype='<f4'))
            run(['numactl','--all','--physcpubind=80,81','--membind=3',OLD/'bin/pointer-search',folder/'full.meta',folder/'float-graph.bin',source/'x.bin',source/'y.bin',root/'qx.bin',root/'qy.bin',folder/'alpha.bin',folder/'gt.bin','-',folder/'search.json',2,0,0,n])
            result=json.loads((folder/'search.json').read_text())[0]
            assert result['recall']==1.,(mode,a,result['recall'])
            assert result['reachable']==n,(mode,a,'reachability',result['reachable'])
            counters=json.loads((folder/'pruning-counters.json').read_text())
            if mode=='fixed':
                assert counters['fixed_calls']>0 and counters['fixed_candidates']>counters['fixed_kept']
            results.append(dict(mode=mode,alpha=float(a),nodes=n,query_count=q,exhaustive_recall=result['recall'],reachable=result['reachable'],mean_degree=result['mean_degree'],pruning_counters=counters))
    final=dict(status='PASS',semantic_proof_sha256=hashlib.sha256((O/'validation/semantic-proof.json').read_bytes()).hexdigest(),actual_fixture_source_sha256=hashlib.sha256((R/'scripts/test_actual_pruner.cpp').read_bytes()).hexdigest(),actual_member_cases=8000,includes_multidimensional_integer_geometry=True,tiny_builds=results,note='Final backlinks are not all guaranteed to have undergone pruning; the non-coverage invariant is checked on actual prune outputs, not assumed for every final adjacency list.')
    (O/'validation/integration-proof.json').write_text(json.dumps(final,indent=2)+'\n')
    print('PASS actual member reference cases + six full tiny builds with exhaustive top10 query agreement')
if __name__=='__main__':main()
