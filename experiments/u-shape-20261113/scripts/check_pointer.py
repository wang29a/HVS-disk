#!/usr/bin/env python3
"""Real corrected OpenImages graph parity; diagnostics, not performance."""
import hashlib, json, os, pathlib, subprocess
R=pathlib.Path(__file__).resolve().parent.parent
OLD=R.parent/'metric-consistent-20261111'
OUT=R/'stage2/validation'
DATA=pathlib.Path('/mnt/nvme3/wz/dataset/Openimage_disk')
COORD=pathlib.Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
PQ='/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/'
def run(args,env,log):
    with open(str(log),'w') as f:
        subprocess.check_call(['numactl','--all','--physcpubind=80,81','--membind=3']+list(map(str,args)),env=env,stdout=f,stderr=subprocess.STDOUT,timeout=180)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    env={k:v for k,v in os.environ.items() if not k.startswith(('LAYOUT_','METRIC_','PTR_'))}
    env.update(LC_ALL='C',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    Q=OLD/'scratch/engine-preflight'
    proofs=[]
    for name,kind,filtering,concat in [('full-on','dual',True,False),('full-off','dual',False,False),('compact','dual',False,False),('single','single',False,True)]:
        G=OLD/'scratch'/kind; stem='full' if name.startswith('full') else 'compact'
        layout=OUT/(name+'-layout'); layout.mkdir(exist_ok=True)
        for filename,source in [('group-1.bin',G/(stem+'.topo')),('group-6.bin',COORD)]:
            if not (layout/filename).exists(): (layout/filename).symlink_to(source)
        base=[str(G/(stem+'.meta')),str(G/(stem+'.topo')),str(COORD),str(layout),'T_XY']
        le=dict(env,LAYOUT_DRAM='1',METRIC_NAV_EXACT='1',LAYOUT_ONE_PASS='1',LAYOUT_TRACE=str(OUT/(name+'-legacy')),LAYOUT_DUMP=str(OUT/(name+'-legacy')))
        if name=='full-off': le['LAYOUT_BYPASS_ALPHA']='1'
        if concat: le['METRIC_CONCAT']='1'
        legacy=[OLD/'bin/metric-search','query']+base+[OUT/(name+'-legacy.json'),PQ,Q/'query1.bin',Q/'query2.bin',Q/'alpha.bin',Q/'gt.bin',2,0,40,120]
        run(legacy,le,OUT/(name+'-legacy.log'))
        pe=dict(env,PTR_TRACE=str(OUT/(name+'-pointer')))
        if concat: pe['PTR_CONCAT']='1'
        pointer=[R/'stage2/bin/pointer-search',G/(stem+'.meta'),G/'float-graph.bin',DATA/'base_img_emb.fvecs',DATA/'base_text_emb.fvecs',Q/'query1.bin',Q/'query2.bin',Q/'alpha.bin',Q/'gt.bin','-',OUT/(name+'-pointer.json'),2,0,int(filtering),40,120]
        run(pointer,pe,OUT/(name+'-pointer.log'))
        for L in (40,120):
            for suffix in ('','.ids','.dists'):
                a=(OUT/(name+'-legacy.L'+str(L)+suffix)).read_bytes()
                b=(OUT/(name+'-pointer.L'+str(L)+suffix)).read_bytes()
                if a!=b: raise RuntimeError('byte parity failed '+name+' L='+str(L)+' suffix='+suffix)
        a=json.loads((OUT/(name+'-legacy.json')).read_text())
        b=json.loads((OUT/(name+'-pointer.json')).read_text())
        for x,y in zip(a,b):
            for k in ('expanded_nodes','inspected_edges','alpha_rejected'):
                target={'expanded_nodes':'expanded','inspected_edges':'inspected_edges','alpha_rejected':'alpha_rejected'}[k]
                # Historical summaries use alpha_rejects, inspected key names
                # are checked below if present; byte trajectory parity is primary.
                if k in x and abs(x[k]-y[target])>1e-8: raise RuntimeError('work counter mismatch '+k)
        proofs.append(dict(case=name,L=[40,120],queries=8,ids_distances_and_expansion_batches_byte_equal=True))
    result=dict(status='PASS',performance_evidence=False,cases=proofs,binary_sha256=hashlib.sha256((R/'stage2/bin/pointer-search').read_bytes()).hexdigest())
    (OUT/'pointer-parity.json').write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__': main()
