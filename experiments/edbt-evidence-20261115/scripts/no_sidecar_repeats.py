"""Keep the identity-backed first pass and add two paired repetitions."""
from pathlib import Path
import sys
import numpy as np
import common as c
import measure
sys.path.insert(0,str(c.E/'paper-priorities-20261114/scripts'))
import no_sidecar as N
OUT=c.RUN/'no-sidecar-three';OLD=c.E/'paper-priorities-20261114/runtime/no-sidecar'
def main():
    OUT.mkdir(exist_ok=True)
    oldplan=c.load(OLD/'plan.json');oldsummary=c.load(OLD/'summary.json');gate=c.load(OLD/'correctness.json')
    if oldsummary['pending'] or gate['status']!='PASS' or c.sha(c.MG_RUN/'bin/multigraph')!=oldplan['binary_sha256']:raise RuntimeError('first pass not reusable')
    manifest=OLD/'manifest-no-sidecar.txt'
    if c.sha(manifest)!=oldplan['derived_manifest_sha256']:raise RuntimeError('manifest changed')
    rows=[dict(r,repeat=1,source=str(OLD/'summary.json')) for r in oldsummary['results']];pending=[]
    for rep in (2,3):
      for i,a in enumerate((.1,.3,.5,.7,.9)):
        cases=['loaded','no-sidecar','specialized'];cases=cases[(i+rep)%3:]+cases[:(i+rep)%3]
        for case in cases:
            info=oldplan['plans'][str(a)][case];tag=str(a)+'-'+case+'-rep'+str(rep);saved=OUT/(tag+'.json');jp=OUT/(tag+'.jobs');N.jobs(jp,info['jobs']);out=OUT/(tag+'-raw.json')
            if saved.exists():data=c.load(saved)
            else:
                choice=measure.quiet();c.save(OUT/(tag+'-affinity.json'),choice)
                args=N.command(manifest if case=='no-sidecar' else c.MG_RUN/'manifest.txt',a,jp,out,seconds=10)
                data=measure.run(args,'no-sidecar-three-'+tag,out,choice['choice']['cpus'])
                if data:c.save(saved,data)
            if not data:pending.append(tag);continue
            by={p['label']:p for p in data['points']};x,y=[by[n] for n in info['labels']];w=info['weight']
            rows.append(dict(alpha=a,case=case,repeat=rep,latency_us=(1-w)*x['latency_us']+w*y['latency_us'],qps=(1-w)*x['qps']+w*y['qps'],peak_rss_kib=max(p['peak_rss_kib'] for p in data['points']),raw_p95_us={p['label']:p['p95_us'] for p in data['points']}))
    means=[]
    for a in (.1,.3,.5,.7,.9):
        for case in ('loaded','no-sidecar','specialized'):
            rs=[r for r in rows if r['alpha']==a and r['case']==case]
            means.append(dict(alpha=a,case=case,covered=len(rs)==3,repeats=rs,mean_latency_us=float(np.mean([r['latency_us'] for r in rs])),max_peak_rss_kib=max(r['peak_rss_kib'] for r in rs)))
    c.save(OUT/'summary.json',dict(results=means,pending=pending,first_pass_sha256=c.sha(OLD/'summary.json'),first_pass_plan_sha256=c.sha(OLD/'plan.json'),selection='unchanged original work-frozen configurations; first accepted pass plus two new paired repeats'))
    return dict(complete=not pending,summary=str(OUT/'summary.json'))
