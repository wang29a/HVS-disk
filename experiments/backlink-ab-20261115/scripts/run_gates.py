"""Protocol gates only; no SSD or formal2M construction before success."""
import signal,struct,sys
import numpy as np
import common as c
import compile_ab,compile_author
from graph_check import read,compare

def subset():
    root=c.RUN/'data-10k';root.mkdir(exist_ok=True);stamp=root/'identity.json';sig=dict(source=c.sha(c.D/'subset.json'),seed=20261115,n=10000,method='RandomState choice without replacement, IDs sorted')
    if stamp.exists():
        old=c.load(stamp)
        if old['signature']!=sig:raise RuntimeError('gate data identity changed')
        for n,h in old['files'].items():
            if c.sha(root/n)!=h:raise RuntimeError('gate data changed')
        return root
    rng=np.random.RandomState(20261115);ids=np.sort(rng.choice(2000000,10000,replace=False)).astype('<u4');ids.tofile(str(root/'subset_ids.u32'))
    for name in ('base_img_emb.fvecs','base_text_emb.fvecs'):
        with (c.D/name).open('rb') as f:n,d=struct.unpack('<II',f.read(8))
        if (n,d)!=(2000000,512):raise RuntimeError('gate input shape')
        data=np.memmap(str(c.D/name),dtype='<f4',mode='r',offset=8,shape=(n,d))
        with (root/name).open('wb') as f:f.write(struct.pack('<II',10000,512));np.asarray(data[ids],dtype='<f4').tofile(f)
    c.save(stamp,dict(signature=sig,files={p.name:c.sha(p) for p in root.iterdir() if p.is_file()}));return root

def build(mode,data):
    root=c.RUN/'gate-graphs'/mode;root.mkdir(parents=True,exist_ok=True);stamp=root/'identity.json'
    binary=c.U/'stage2-v2/bin/build-unified' if mode=='formal-A' else c.RUN/'bin'/('build-'+mode)
    signature=dict(binary=c.sha(binary),input=c.sha(data/'identity.json'),threads=1)
    if stamp.exists():
        old=c.load(stamp)
        if old['signature']!=signature or old['graph']!=c.sha(root/'float-graph.bin'):raise RuntimeError('gate graph changed')
        return root
    start=c.time.time()
    if mode=='author':args=[binary,data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',root/'float-graph.bin']
    else:args=[binary,'dual',data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',10000,512,512,1,root]
    c.run(args,'gate-build-'+mode,cpus='80',timeout=3600,extra=dict(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1'))
    c.save(stamp,dict(signature=signature,graph=c.sha(root/'float-graph.bin'),wall_seconds=c.time.time()-start));return root

def main():
    for name in ('logs','commands','bin','build','validation','gate-graphs'):(c.RUN/name).mkdir(parents=True,exist_ok=True)
    frozen=c.load(c.ROOT/'FREEZE.json');import hashlib
    if hashlib.sha256((c.ROOT/'README.md').read_bytes()[:frozen['bytes']]).hexdigest()!=frozen['sha256']:raise RuntimeError('protocol prefix changed')
    lock=c.lock();c.capacity(4*2**30,'AB independent gate compilation, source copy and10k graphs')
    compile_ab.main();compile_author.main();data=subset();graphs={};stats={};dirs={}
    for mode in ('formal-A','A','B','author'):
        dirs[mode]=build(mode,data);graphs[mode],stats[mode]=read(dirs[mode]/'float-graph.bin');c.save(c.RUN/'validation'/('structure-'+mode+'.json'),stats[mode])
    regression=dict(status='PASS' if c.sha(dirs['A']/'float-graph.bin')==c.sha(dirs['formal-A']/'float-graph.bin') else 'FAIL',A=c.sha(dirs['A']/'float-graph.bin'),formal_A=c.sha(dirs['formal-A']/'float-graph.bin'),criterion='whole raw graph byte-equal including saved order and float ranges')
    relation=compare(graphs['B'],graphs['author'],c.RUN/'validation/B-author-node-differences.jsonl.gz')
    reference=dict(status='PASS' if relation['neighbor_sets_equal_fraction']>=.99 else 'REQUIRES_EXPLANATION_STOP',comparison=relation)
    result=dict(reference=reference,structure={m:stats[m] for m in ('A','B')},A_regression=regression,author_structure=stats['author'],formal_graphs='not_started',SSD='not_authorised')
    passed=reference['status']=='PASS' and regression['status']=='PASS' and all(stats[m]['status']=='PASS' for m in ('A','B'))
    result['status']='ALL_GATES_PASS' if passed else 'STOP_AT_CORRECTNESS_GATE'
    c.save(c.RUN/'gates.json',result);c.status('gates_passed_stage1_ready' if passed else 'stopped_at_correctness_gate',result=str(c.RUN/'gates.json'),reason='No unexplained <99% author match may start formal builds')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.status('failed',error=str(e));raise
