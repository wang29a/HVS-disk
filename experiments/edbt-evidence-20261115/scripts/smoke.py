"""Small real-file runtime gates, independent of performance experiments."""
from pathlib import Path
import sys
import common as c
import native_writer,openimages as O
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
from no_sidecar import byte_equal,jobs
sys.path.insert(0,str(c.MG/'scripts'))
from validate import distances

def main():
    root=c.RUN/'smoke';root.mkdir(exist_ok=True);tiny=c.MG_RUN/'tiny';nroot=c.RUN/'native'
    c.status=lambda stage,**kw:c.save(root/'status.json',dict(stage=stage,**kw))
    index=native_writer.write('tiny',tiny/'unified',tiny,tiny)
    cases=[];maxerror=0.
    for a in (0.,.1,.5,.9,1.):
        name=str(a);outputs=[]
        for variant in ('plain','counted'):
            out=root/('native-'+variant+'-'+name+'.json')
            args=[nroot/'bin'/('native-'+variant),str(index)+'/',tiny/'query_img_emb.fvecs',tiny/'query_text_emb.fvecs',tiny/('query-alpha-'+name+'.bin'),tiny/('gt-alpha-'+name+'.bin'),tiny/'ids.txt',out,'-',2,0,40,80]
            c.run(args,'smoke-native-'+variant+'-'+name,cpus='80,81',timeout=120,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
            values=c.load(out);outputs.append(values);maxerror=max(maxerror,distances(out,tiny,tiny,tiny/('query-alpha-'+name+'.bin')))
        if [p['queries'] for p in outputs[0]]!=[p['queries'] for p in outputs[1]]:raise RuntimeError('counted native changed outputs')
        if any(p['pq_space_evals']!=2*p['pq_candidates'] or p['expanded']*2!=p['exact_space_evals'] or p['pq_candidates']<=0 for p in outputs[1]):raise RuntimeError('native counters')
        jp=root/'multi.jobs';jobs(jp,[dict(label=m,method=m,L=80,kx=20,ky=20 if m=='M_equal' else 0,weight=a) for m in ('DEG_off','M_equal','Or_nearest')])
        prefixes=[]
        for medium in ('dram','ssd'):
            out=root/('multi-'+name+'-'+medium+'.json');prefix=root/('multi-'+name+'-'+medium)
            c.run([O.OUT/'bin/multigraph',tiny/'manifest.txt',tiny/'query_img_emb.fvecs',tiny/'query_text_emb.fvecs',tiny/('query-alpha-'+name+'.bin'),tiny/('gt-alpha-'+name+'.bin'),tiny/'ids.txt',out,medium,2,0,'pq',jp,prefix,1],'smoke-route-'+name+'-'+medium,cpus='80,81',timeout=120,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
            values=c.load(out);maxerror=max(maxerror,max(p['max_return_relative_error'] for p in values))
            if any('expanded' not in q for p in values for q in p['queries']):raise RuntimeError('query work JSON missing')
            prefixes.append(prefix)
        byte_equal(prefixes[0],prefixes[1],['DEG_off','M_equal','Or_nearest']);cases.append(dict(alpha=a,corrected_native_counts=True,three_method_trace_parity=True))
    c.save(root/'proof.json',dict(status='PASS',nodes=256,queries=12,cases=cases,max_return_relative_error=maxerror,scope='real files and real native/three-route executors; not performance or full-dataset gate'))
    c.status('complete',proof=str(root/'proof.json'))
    if (c.RUN/'smoke-failure.json').exists():c.os.replace(str(c.RUN/'smoke-failure.json'),str(root/'previous-failure-before-exit-fix.json'))
if __name__=='__main__':
    try:main()
    except Exception as e:c.save(c.RUN/'smoke-failure.json',dict(error=str(e)));raise
