"""Run the approved A/B/C check now, then resume only if B/C passes."""
import fcntl
from pathlib import Path
import signal
import sys
import time
import numpy as np
import common as c
import prepare_inputs as inp
import run_queue as queue
import validate as v
import layer_analysis as la

OUT=c.RUN/'three-layer-20261114'
def state(stage,**extra):
    c.save(OUT/'status.json',dict(stage=stage,pid=c.os.getpid(),updated_unix=time.time(),**extra))
    c.status(stage,**extra)

def proof_identity():
    build=c.load(OUT/'validation/build.json')
    if c.sha(OUT/'bin/pq-layers')!=build['binary_sha256']:raise RuntimeError('diagnostic binary changed')
    if c.sha(c.ROOT/'scripts/pq_layers.cpp')!=build['source_sha256']:raise RuntimeError('diagnostic source changed')
    if c.sha(c.RUN/'bin/multigraph')!=build['original_binary_sha256']:raise RuntimeError('original executor changed')
    if c.sha(c.ROOT/'scripts/multigraph.cpp')!=build['included_executor_sha256']:raise RuntimeError('included engine changed')
    return dict(original_binary_sha256=build['original_binary_sha256'],diagnostic_binary_sha256=build['binary_sha256'],
        packed_graphs={name:c.sha(c.RUN/'packed'/name/'identity.json') for name in ('x','y','u')},
        subset_sha256=c.sha(c.D/'subset.json'),shared_inputs_sha256=c.sha(c.RUN/'data/shared.json'),
        amendment_sha256=c.sha(c.ROOT/'AMENDMENT-20261114.json'),readme_sha256=c.sha(c.ROOT/'README.md'),
        gt_sha256={str(a):c.sha(inp.gtpath(a)) for a in inp.WEIGHTS},L=20000,k=320)

def main():
    own=(c.RUN/'coordinator.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    queue.verify_sources('three-layer-20261114');global_lock=c.lock()
    identity=proof_identity();path=OUT/'identity.json'
    if path.exists() and c.load(path)!=identity:raise RuntimeError('diagnostic identity changed; old evidence must stay separate')
    c.save(path,identity)
    thresholds=c.load(c.ROOT/'AMENDMENT-20261114.json')
    query_ids=c.load(c.RUN/'data/oracle.json')['query_ids']
    if list(map(int,(c.RUN/'data/gate-ids.txt').read_text().split()))!=query_ids:raise RuntimeError('preselected queries changed')
    (OUT/'manifest.txt').write_text((c.RUN/'manifest.txt').read_text())
    state('running_three_layer_BC',L=20000,k=320,queries=len(query_ids),threads=2)
    c.run([OUT/'bin/pq-layers',OUT/'manifest.txt',c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',c.RUN/'data/gate-ids.txt',OUT/'candidates',20000,320,2],
        'three-layer-candidates',cpus='80,81',timeout=1800,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    state('running_three_layer_A',queries=len(query_ids),threads=2)
    exact=la.exact_A(OUT,c.D,c.D,query_ids,inp.p,inp.WEIGHTS,identity)
    x,y,qx,qy=(inp.p.matrix(c.D/name) for name in inp.p.NAMES)
    truths={str(a):inp.p.matrix(inp.gtpath(a),'<u4') for a in inp.WEIGHTS}
    raw={qi:c.load(OUT/'candidates'/('q-'+str(qi)+'.json')) for qi in query_ids}
    old={}
    for a in (.1,.3):
        file=c.RUN/'validation/real'/('oracle-a'+str(a)+'.json')
        old[str(a)]={r['query_id']:r for r in c.load(file)[0]['queries']}
    result=la.analyze(raw,exact,query_ids,x,y,qx,qy,truths,inp.WEIGHTS,thresholds,old_points=old)
    result.update(identity=identity,pq_score_audits={str(qi):r['pq_score_audit'] for qi,r in raw.items()})
    # Exact A must reproduce the previously measured two-index upper bound.
    oracle=c.load(c.RUN/'data/oracle.json')
    for group in result['groups']:
        if group['method']=='M' and abs(group['recall']['A']-oracle['weights'][str(group['alpha'])]['recall'])>1e-12:raise RuntimeError('recomputed exact A differs from saved oracle')
    c.save(OUT/'summary.json',result);(OUT/'RESULTS.md').write_text(la.markdown(result),encoding='utf-8')
    if result['status']!='PASS':
        state('stopped_at_BC_gate',failed=[dict(alpha=r['alpha'],method=r['method'],recall=r['recall'],BC_candidates=r['BC_candidates']) for r in result['groups'] if not r['passed']]);return
    state('three_layer_passed_resuming_original_queue')
    # Reuse existing inputs and graphs by content identity. No GT or graph is
    # rebuilt, and the original plan/failure logs are not overwritten.
    v.coverage()
    for name in ('a01','a03','a07','a09'):queue.pack(name,queue.build(name))
    queue.manifest(['x','y','a01','a03','a05','a07','a09','u']);v.real()
    curves=queue.tune();queue.measure(curves)
    c.run([sys.executable,c.ROOT/'scripts/report.py'],'report-laion',cpus='72,73',timeout=300)
    state('awaiting_user_review',next='Laion-only completed; stop before OpenImages')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as error:c.stop();state('stopped',error=str(error),action='no parameter adjustment');raise
