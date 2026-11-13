"""Serial four-core mechanism follow-up. Reuse graph, GT and accepted timings.

Batch diagnostic L values in one existing search process per r/filter/cohort;
formal timings keep the validated one-point monitor and three rotated repeats.
"""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'laion-six-cell-20261111'
sys.path.insert(0,str(OLD/'scripts'))
import common as c
import run_queue as q
import phase_runtime as p
import coverage_extension as coverage
from summarize import summarize
sys.path.insert(0,str(ROOT/'scripts'))
import existing_analysis
c.ROOT,c.ACTIVE=ROOT,ROOT

GRID=[(L,r) for L in (60,70,80,100,120,140,160) for r in (60,80) if r<=L]
COUNTERS=('recall','expanded_nodes','pq_space_evals','topology_submitted_pages',
          'coordinate_submitted_pages','inspected_edges','alpha_rejected_edges')


def link_results(original,root):
    for src in (original/'results').iterdir():
        if src.is_file():
            dst=root/'results'/src.name
            if not dst.exists():dst.symlink_to(src)
    for src in (original/'analysis').glob('report-time-*-accepted.json'):
        dst=root/'analysis'/src.name
        if not dst.exists():dst.symlink_to(src)


def diagnostic_batch(cohort,filtering,r,levels,force=False):
    def label(L):
        return ('tune-full-T_XY-pq' if cohort=='tune' else 'report-work-full-T_XY-pq')+'-f{}-L{}-r{}'.format(int(filtering),L,r)
    todo=[L for L in levels if force or not (p.ROOT/'results'/(label(L)+'.json')).exists()]
    if not todo:return
    stem='dense-{}-f{}-r{}'.format(cohort,int(filtering),r)
    marker=p.ROOT/'analysis'/(stem+'-batch.json')
    if marker.exists():return
    meta,topo,folder=q.layout('full','T_XY')
    files={k:v['path'] for k,v in p.SPLIT['cohorts'][cohort]['files'].items()}
    result=p.ROOT/'results'/(stem+'.json')
    env=dict(p.EXTRA,LAYOUT_DRAM='1',LAYOUT_ONE_PASS='1',LAYOUT_GT_DIAGNOSTIC='1',METRIC_RERANK=str(r))
    if not filtering:env['LAYOUT_BYPASS_ALPHA']='1'
    if cohort=='report':
        env.update(LAYOUT_TRACE=str(p.ROOT/'analysis'/stem),LAYOUT_DUMP=str(p.ROOT/'analysis'/stem))
    args=[p.ROOT/'bin/metric-search','query',meta,topo,c.COORD,folder,'T_XY',result,q.PQ.as_posix()+'/',
          files['query1'],files['query2'],files['alpha'],files['gt'],4,10]+todo
    c.run(args,stem,c.CPUS,env,timeout=900)
    points=q.read(result)
    if [x['L'] for x in points]!=todo:raise RuntimeError('batch L identity')
    for point in points:
        dest=p.ROOT/'results'/(label(point['L'])+'.json')
        if dest.exists():
            previous=q.read(dest)[0]
            if any(abs(previous[k]-point[k])>1e-6 for k in COUNTERS):
                raise RuntimeError('batched diagnostics changed recall or deterministic work')
        else:c.save(dest,[point])
    c.save(marker,dict(status='PASS',cohort=cohort,filtering=filtering,r=r,levels=todo,
        deterministic_counter_parity_with_existing=True,one_resident_load_for_multiple_L=True,
        timing_use='diagnostics only; one-pass latency never used for comparison'))


def new_trace_summary():
    rows=[]
    files=p.SPLIT['cohorts']['report']['files']
    for L in (80,120,160):
        prefix=p.ROOT/'analysis'/('dense-report-f0-r80.L'+str(L))
        output=p.ROOT/'analysis'/('parents-off-L{}-r80.json'.format(L))
        if not output.exists():
            c.run([ROOT/'bin/edge-audit','trace',q.GRAPH/'full.meta',q.GRAPH/'full.topo',prefix,
                files['alpha']['path'],str(prefix)+'.ids',files['gt']['path'],output],
                'parents-off-L{}-r80'.format(L),[72])
        records=q.read(output)
        sums={k:sum(x[k] for x in records) for k in records[0] if k not in ('query_id','alpha')}
        rows.append(dict(L=L,r=80,queries=len(records),recall=100*sums['true_hits']/(10*len(records)),sums=sums,
            inactive_candidate_parent_fraction=sums['candidate_inactive_parent']/sums['candidate_nonentry'],
            inactive_returned_parent_fraction=sums['returned_inactive_parent']/sums['returned_nonentry'],
            inactive_true_returned_parent_fraction=sums['true_inactive_parent']/sums['true_nonentry'] if sums['true_nonentry'] else None))
    c.save(p.ROOT/'analysis/first-discovery-summary.json',dict(records=rows,
        cohort='held-out report queries at declared fixed L/r',
        interpretation='First-discovery provenance, not indispensable-edge or counterfactual reachability evidence.'))


def run():
    c.init()
    if not (ROOT/'analysis/existing-analysis-status.json').exists():existing_analysis.main()
    inventory=q.read(OLD/'analysis/inventory.json')
    for workload in inventory['workloads']:
        original=OLD/'workloads'/('alpha-'+str(workload['fixed_alpha']))
        root,split=q.prepare_workload(workload)
        original_split=q.read(original/'analysis/tuning-split-v2.json')
        for cohort in ('tune','report'):
            if split['cohorts'][cohort]['query_ids']!=original_split['cohorts'][cohort]['query_ids']:
                raise RuntimeError('query cohort changed')
            for key,value in split['cohorts'][cohort]['files'].items():
                if value['sha256']!=original_split['cohorts'][cohort]['files'][key]['sha256']:
                    raise RuntimeError('workload contents changed')
        c.ACTIVE=root
        p.ROOT,p.B,p.SPLIT,p.COORD,p.PQ=root,c,split,c.COORD,str(q.PQ)+'/'
        p.layout=q.layout
        link_results(original,root)
        if c.digest(ROOT/'bin/metric-search')!=q.read(OLD/'analysis/tool-validation.json')['search_sha256']:
            raise RuntimeError('search binary changed')
        q.await_memory(80<<30,'dense-DRAM')
        identity=root/'analysis/dense-input-identity.json'
        current=dict(graph_validation_sha256=c.digest(OLD/'analysis/full-graph-identity.json'),
            split_sha256=c.digest(root/'analysis/tuning-split-v2.json'),
            old_frozen_sha256=c.digest(original/'analysis/frozen-full-T_XY-pq-coverage-v2.json'),
            binary_sha256=c.digest(ROOT/'bin/metric-search'),driver_sha256=c.digest(Path(__file__)),grid=[list(x) for x in GRID],
            carried_over_accepted_timing_points=len(list((root/'analysis').glob('report-time-*-accepted.json'))))
        if not identity.exists():c.save(identity,current)
        elif any(q.read(identity)[key]!=current[key] for key in current if key!='carried_over_accepted_timing_points'):
            raise RuntimeError('dense experiment identity changed')
        frozen_path=root/'analysis/frozen-full-T_XY-pq-dense-v3.json'
        if not frozen_path.exists():
            rows={(x['filtering'],x['L'],x['r']):x for x in q.read(original/'analysis/frozen-full-T_XY-pq-coverage-v2.json')['all_training_points']}
            for filtering in (False,True):
                for r in (60,80):
                    levels=[L for L,rr in GRID if rr==r]
                    diagnostic_batch('tune',filtering,r,levels)
                    for L in levels:
                        label='tune-full-T_XY-pq-f{}-L{}-r{}'.format(int(filtering),L,r)
                        point=q.read(root/'results'/(label+'.json'))[0]
                        rows[(filtering,L,r)]=dict(filtering=filtering,L=L,r=r,recall=point['recall'],
                            pages=point['submitted_pages'],pq=point['pq_space_evals'],
                            exact=point['navigation_exact_space_evals'],rerank_evals=point['exact_space_evals'],
                            source=str(root/'results'/(label+'.json')))
            frozen=coverage.freeze(list(rows.values()),False)
            frozen.update(protocol='dense-v3',declared_grid=GRID,old_frozen_sha256=current['old_frozen_sha256'])
            c.save(frozen_path,frozen)
        frozen=q.read(frozen_path)
        # Preserve all declared-grid report work/trace points regardless of
        # their reporting recall or latency. Only training chooses timings.
        for filtering in (False,True):
            for r in (60,80):diagnostic_batch('report',filtering,r,[L for L,rr in GRID if rr==r],force=True)
        new_trace_summary()
        exact=q.read(original/'analysis/frozen-full-T_XY-exact-coverage-v2.json')
        c.save(root/'analysis/frozen-full-T_XY-exact-dense-v3.json',exact)
        p.measure(frozen,['DRAM','SSD'])
        summarize(root,'-dense-v3')
        c.save(root/'analysis/done.json',dict(pending=p.pending_points(),completed_unix=time.time()))
    c.ACTIVE=ROOT
    pending={w.name:q.read(w/'analysis/pending-measurements.json') for w in (ROOT/'workloads').iterdir()
             if (w/'analysis/pending-measurements.json').exists() and q.read(w/'analysis/pending-measurements.json')}
    c.status('complete_with_pending' if pending else 'complete',pending=pending,
        scope='Laion local L/r densification; static activity; OI fixed-parameter buckets; discovery-edge provenance')


if __name__=='__main__':
    c.init()
    c.status('queued')
    lock=open(str(ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:run()
    except Exception as error:
        c.status('failed',error=str(error));raise
