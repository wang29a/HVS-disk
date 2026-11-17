"""Resume only the sampling-coverage stop; immutable artifacts are revalidated."""
import fcntl
import json
import time
import numpy as np
import run_g0 as c
import run_g2 as g
from run_g1 import matrix


def main():
    previous=c.ROOT/'runtime/g2'
    old=json.loads((previous/'status.json').read_text())
    assert old['stage']=='STOPPED_AT_G2' and old['reason']=='95% not bracketed full-0.5-tune-0'
    c.RUN=c.ROOT/'runtime/g2-attempt2';c.RUN.mkdir(exist_ok=True)
    if (c.RUN/'resume-identity.json').exists():raise RuntimeError('attempt2 already used')
    g.DEPTHS=[10,12,14]+g.DEPTHS
    lock=(g.E/'budget-reorder-20261103/queue.lock').open('a+')
    g.status('waiting_global_lock');fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        proof=json.loads((previous/'protocol.json').read_text());assert c.sha(g.OUT/'source-ids.u32')==proof['source_ids_sha256']
        for path,digest in proof['source_inputs'].items():assert c.sha(path)==digest
        ids=np.fromfile(str(g.OUT/'source-ids.u32'),dtype='<u4')
        for s in ('img','text'):
            actual=matrix(g.OUT/('base-'+s+'.bin'));source=matrix(g.DATA/('base_'+s+'_emb.fvecs'))
            assert np.array_equal(actual,np.asarray(source[ids]))
            assert np.array_equal(matrix(g.OUT/('initial-'+s+'.bin')),actual[:90000])
        assert not (g.OUT/'pq').exists() and not (g.OUT/'exact').exists(),'mutable fixtures unexpectedly exist'
        identity=json.loads((previous/'build-identity.json').read_text())
        for path,digest in identity['sources'].items():assert c.sha(path)==digest
        assert c.sha(previous/'g2_insert')==identity['binary']
        builder=json.loads((previous/'builder-identity.json').read_text())
        assert c.sha(g.MG/'bin/multigraph')==builder['executor']
        retained=[p for p in g.OUT.rglob('*') if p.is_file()]
        c.save('resume-identity.json',dict(reason='sampling grid started above target; no quality-gate failure observed',
            previous_status=old,previous_runtime=str(previous),depths=g.DEPTHS,
            retained_artifacts={str(p):c.sha(p) for p in retained},
            executor_build=identity,static_executor=builder['executor']))
        full=g.evaluate('full',g.OUT/'full/packed/compact.meta',g.OUT/'full/packed/compact.topo');c.save('full-matched.json',full)
        comparisons=[]
        for mode in ('pq','exact'):
            root=g.fixture(mode);g.run([previous/'g2_insert',root,mode],'insert-'+mode,timeout=14400)
            rows=g.evaluate(mode,root/'eval.meta',root/'eval.topo');c.save(mode+'-matched.json',rows)
            for row,base in zip(rows,full):
                increase=row['matched']['expanded']/base['matched']['expanded']-1
                comparisons.append(dict(mode=mode,alpha=row['alpha'],cohort=row['cohort'],repeat=row['repeat'],
                    baseline=base['matched'],incremental=row['matched'],increase=increase,passed=increase<=.15))
            c.save('comparisons.json',comparisons)
            if any(not r['passed'] for r in comparisons if r['cohort']=='report'):
                raise RuntimeError('G2 expansion increase exceeds 15%; see comparisons.json')
        g.status('G2_PASSED_BOTH_DISTANCES',G4='PASSED_G1_AND_G2_BOTH',G3='NOT_RUN')
    except Exception as e:
        g.status('STOPPED_AT_G2',reason=str(e),dependent_experiments='not started');raise


if __name__=='__main__':main()
