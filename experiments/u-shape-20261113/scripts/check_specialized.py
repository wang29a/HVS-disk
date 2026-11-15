#!/usr/bin/env python3
"""Small deterministic integration fixture for the fixed-target builder."""
import hashlib,json,os,pathlib,struct,subprocess
import numpy as np
R=pathlib.Path(__file__).resolve().parent.parent; O=R/'stage2'
def take(f,fmt):
    n=struct.calcsize(fmt); b=f.read(n)
    if len(b)!=n: raise RuntimeError('truncated fixture graph')
    return struct.unpack(fmt,b)
def main():
    out=O/'validation/fixed-fixture'; out.mkdir(parents=True,exist_ok=True)
    rng=np.random.RandomState(20261113); n=256; dims=32
    for name in ('x','y'):
        a=rng.normal(size=(n,dims)).astype('<f4')
        with (out/(name+'.bin')).open('wb') as f: f.write(struct.pack('<II',n,dims)); a.tofile(f)
    env=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    results=[]
    for a in ('0.1','0.9'):
        dest=out/a; dest.mkdir(exist_ok=True)
        cmd=['numactl','--all','--physcpubind=80,81','--membind=3',str(O/'bin/build-fixed'),'dual',str(out/'x.bin'),str(out/'y.bin'),str(n),str(dims),str(dims),'2',str(dest)]
        with (dest/'build.log').open('w') as log: subprocess.check_call(cmd,env=dict(env,HVS_FIXED_ALPHA=a),stdout=log,stderr=subprocess.STDOUT,timeout=120)
        edges=0; ranges=0
        with (dest/'float-graph.bin').open('rb') as f:
            assert take(f,'<QII')==(n,dims,dims)
            for node in range(n):
                degree,=take(f,'<I'); assert degree<=40
                for _ in range(degree):
                    nb,nr=take(f,'<II'); assert nb<n and nb!=node
                    intervals=[take(f,'<ff') for _ in range(nr)]
                    # Equal internal channels remove preference dependence:
                    # selected edges must be available over the whole interval.
                    if intervals!=[(0.,1.)]: raise RuntimeError('specialized edge unexpectedly has restricted preference range '+str(intervals))
                    edges+=1; ranges+=nr
            assert not f.read(1)
        assert edges>0
        results.append(dict(alpha=float(a),edges=edges,full_intervals=ranges,graph_sha256=hashlib.sha256((dest/'float-graph.bin').read_bytes()).hexdigest()))
    if results[0]['graph_sha256']==results[1]['graph_sha256']: raise RuntimeError('fixed alpha did not change synthetic topology')
    b=json.loads((O/'specialized-build.json').read_text())
    proof=dict(status='PASS',builder_sha256=b['builder_sha256'],distance_sites=b['distance_sites'],fixtures=results,scope='integration/format/fixed-objective degeneration; not graph quality or performance evidence')
    (O/'specialized-validation.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__': main()
