"""Stage 5: five fixed-alpha workloads; DRAM tuning, frozen SSD timings only."""
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
SCAN=ROOT/'weight-scan'
OLD=ROOT.parent/'laion-six-cell-20261111'
sys.path.insert(0,str(ROOT/'scripts'))
import run_phase_one as phase
c,q,p=phase.c,phase.q,phase.p
coverage=phase.coverage
from summarize import source,interpolate
c.ROOT,c.ACTIVE=SCAN,SCAN
ALPHAS=[.1,.3,.5,.7,.9]
LEVELS=[40,60,70,80,100,120,140,160,200,320]
DEPTHS=[20,40,60,80,120,160]


def curve(frozen,filtering,family):
    candidates=[x for x in frozen['selected'] if x['filtering']==filtering]
    key=(lambda x:(x['pages'],x['pq']+x['exact'],x['L'],x['r'])) if family=='pages' else (
        lambda x:(x['pq']+x['exact'],x['pages'],x['L'],x['r']))
    rows=[]
    for target in frozen['targets']:
        covering=[x for x in candidates if x['recall']>=target]
        if not covering:continue
        point=min(covering,key=key)
        if not rows or (point['L'],point['r'])!=(rows[-1]['L'],rows[-1]['r']):rows.append(point)
    return rows


def choose_root(workload):
    root,split=q.prepare_workload(workload)
    c.ACTIVE=root
    p.ROOT,p.B,p.SPLIT,p.COORD,p.PQ=root,c,split,c.COORD,str(q.PQ)+'/'
    p.layout=q.layout
    previous=ROOT/'workloads'/('alpha-'+str(workload['fixed_alpha']))
    if previous.exists():
        old_split=q.read(previous/'analysis/tuning-split-v2.json')
        for cohort in ('tune','report'):
            if split['cohorts'][cohort]['query_ids']!=old_split['cohorts'][cohort]['query_ids']:
                raise RuntimeError('query IDs changed')
            if any(v['sha256']!=old_split['cohorts'][cohort]['files'][k]['sha256'] for k,v in split['cohorts'][cohort]['files'].items()):
                raise RuntimeError('query/GT contents changed')
        phase.link_results(previous,root)
    return root


def grid_for(levels):return sorted(set((L,r) for L in levels for r in DEPTHS+[L] if r<=L))


def diagnostics(cohort,grid):
    for filtering in (False,True):
        for r in sorted(set(r for L,r in grid)):
            levels=[L for L,rr in grid if rr==r]
            prefix='tune-full-T_XY-pq' if cohort=='tune' else 'report-work-full-T_XY-pq'
            missing=[L for L in levels if not (p.ROOT/'results'/('{}-f{}-L{}-r{}.json'.format(prefix,int(filtering),L,r))).exists()]
            marker=p.ROOT/'analysis'/('dense-{}-f{}-r{}-batch.json'.format(cohort,int(filtering),r))
            if missing and marker.exists():
                # An adaptive high-L extension is a new batch. Keep its prior
                # proof, then allow the unchanged routine to fill only gaps.
                archived=marker.with_name(marker.stem+'-before-extension-'+str(max(missing))+'.json')
                marker.replace(archived)
            # Use a unique batch namespace. No report trajectories are needed
            # in this scan, but the reused batch routine stores them; counters
            # and GT candidate diagnostics stay outside timed measurements.
            phase.diagnostic_batch(cohort,filtering,r,levels)


def rows_from_grid(grid):
    rows=[]
    for filtering in (False,True):
        for L,r in grid:
            label='tune-full-T_XY-pq-f{}-L{}-r{}'.format(int(filtering),L,r)
            point=q.read(p.ROOT/'results'/(label+'.json'))[0]
            rows.append(dict(filtering=filtering,L=L,r=r,recall=point['recall'],pages=point['submitted_pages'],
                pq=point['pq_space_evals'],exact=point['navigation_exact_space_evals'],
                rerank_evals=point['exact_space_evals'],source=str(p.ROOT/'results'/(label+'.json'))))
    return rows


def summary(root,frozen,alpha):
    rounds=[r for r in (1,2,3) if all(source(root,'pq',m,x,r) for m in ('DRAM','SSD') for x in frozen['selected'])]
    rows=[]
    for family in ('pages','compute'):
        for medium in ('DRAM','SSD'):
            for filtering in (False,True):
                configs=curve(frozen,filtering,family)
                observations=[interpolate([source(root,'pq',medium,x,r) for x in configs],95) for r in rounds]
                row=dict(family=family,medium=medium,filtering=filtering)
                if len(rounds)!=3 or any(x is None for x in observations):row.update(status='UNCOVERED_OR_INCOMPLETE',rounds=rounds)
                else:
                    keys=observations[0]['values'].keys()
                    row.update(status='COVERED',rounds=rounds,mean={k:sum(x['values'][k] for x in observations)/3 for k in keys},observations=observations)
                rows.append(row)
    result=dict(alpha=alpha,target_recall=95,threads=4,numa=3,records=rows,complete_rounds=rounds,
        note='DRAM-only tuning; SSD uses frozen configs. Work rules do not claim global minimum latency.')
    c.save(root/'analysis/weight-summary.json',result)
    return result


def main():
    c.init()
    started=time.monotonic()
    # Wait without taking the experiment lock. Never delay the running stage 4.
    gates=[ROOT/'analysis/offline-first-three-done.json']
    while not all(x.exists() for x in gates):
        c.status('waiting_for_offline_prerequisites',missing=[str(x) for x in gates if not x.exists()])
        if time.monotonic()-started>21600:raise RuntimeError('offline prerequisites not available after six hours')
        time.sleep(60)
    if q.read(gates[0])['status']!='complete':raise RuntimeError('first three analyses failed')
    lock=open(str(ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    if q.read(ROOT/'status.json')['stage'] not in ('complete','complete_with_pending'):
        raise RuntimeError('stage 4 has not completed; refusing to overlap')
    inventory=q.read(OLD/'analysis/inventory.json')
    original=q.read(ROOT.parent/'concurrency-alpha/analysis/manifest.json')['configurations']
    inventory['workloads']=sorted([x for x in original if x['dataset']=='laion' and x['fixed_alpha'] in ALPHAS],key=lambda x:x['fixed_alpha'])
    if [x['fixed_alpha'] for x in inventory['workloads']]!=ALPHAS:raise RuntimeError('five GTs unavailable')
    for w in inventory['workloads']:
        for path,expected in w['input_sha256'].items():
            if c.digest(path)!=expected:raise RuntimeError('existing GT/query hash changed')
    c.save(SCAN/'analysis/inventory.json',inventory)
    # Validate, do not regenerate, the five existing GTs. The exhaustive
    # sample scan belongs to stage 5 and is bound to four cores in DRAM.
    gt_proof=SCAN/'analysis/gt-objective-audit.json'
    if not gt_proof.exists():
        c.ACTIVE=SCAN
        c.run([sys.executable,ROOT/'scripts/check_scan_gt.py',SCAN],'validate-five-existing-GTs',c.CPUS,
              dict(OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',OMP_NUM_THREADS='4'),timeout=3600)
    if q.read(gt_proof)['status']!='PASS':raise RuntimeError('GT objective audit failed')
    results=[]
    q.TARGETS=[92,94,95,96,97]
    for workload in inventory['workloads']:
        root=choose_root(workload)
        q.await_memory(80<<30,'weight-scan-DRAM')
        frozen_path=root/'analysis/frozen-weight-scan.json'
        if not frozen_path.exists():
            levels=list(LEVELS);grid=grid_for(levels)
            diagnostics('tune',grid)
            rows=rows_from_grid(grid)
            for L in (600,1000,1600):
                if all(max(x['recall'] for x in rows if x['filtering']==f)>=97 for f in (False,True)):break
                levels.append(L);grid=grid_for(levels)
                diagnostics('tune',grid);rows=rows_from_grid(grid)
            frozen=coverage.freeze(rows,False)
            frozen.update(protocol='five weights; common initial L/r grid, bounded training-only extension',
                levels=levels,depths=DEPTHS,grid=grid,search_sha256=c.digest(SCAN/'bin/metric-search'),
                driver_sha256=c.digest(Path(__file__)))
            c.save(frozen_path,frozen)
        frozen=q.read(frozen_path)
        diagnostics('report',[tuple(x) for x in frozen['grid']])
        p.measure(frozen,['DRAM','SSD'])
        results.append(summary(root,frozen,workload['fixed_alpha']))
        c.save(root/'analysis/done.json',dict(completed_unix=time.time(),pending=p.pending_points()))
    c.ACTIVE=SCAN
    c.save(SCAN/'analysis/five-weight-results.json',dict(reports=results,
        distance_scale=q.read(ROOT/'analysis/distance-scale.json'),
        x_axis='Nominal alpha and matched-GT image contribution from local 256-query scale analysis'))
    incomplete=[x['alpha'] for x in results if any(r['status']!='COVERED' for r in x['records'])]
    c.status('complete_with_gaps' if incomplete else 'complete',incomplete_alphas=incomplete)


if __name__=='__main__':
    try:main()
    except Exception as error:
        c.status('failed',error=str(error));raise
