"""Frozen training curves, complete repeats only; no extrapolation or tail interpolation."""
import json
from pathlib import Path
import statistics
import time
import common as c

METRICS=('latency_us','qps','submitted_pages','topology_submitted_pages','coordinate_submitted_pages',
    'pq_space_evals','navigation_exact_space_evals','exact_space_evals','expanded_nodes',
    'reranked_nodes','topology_submit_wait_us','coordinate_submit_wait_us','pq_lookup_us','exact_us',
    'inspected_edges','alpha_rejected_edges')


def read(path): return json.loads(Path(path).read_text())


def source(root,nav,medium,config,repeat):
    label='report-time-full-T_XY-{}-{}-f{}-L{}-r{}-repeat{}'.format(
        nav,medium,int(config['filtering']),config['L'],config['r'],repeat)
    if not (root/'analysis'/(label+'-accepted.json')).exists(): return None
    return label,read(root/'results'/(label+'.json'))[0]


def curve(frozen,filtering,family):
    candidates=[x for x in frozen['tuning_frontier'] if x['filtering']==filtering]
    key=(lambda x:(x['pages'],x['pq']+x['exact'],x['L'],x['r'])) if family=='pages' else (
        lambda x:(x['pq']+x['exact'],x['pages'],x['L'],x['r']))
    rows=[]
    for target in frozen['targets']:
        covering=[x for x in candidates if x['recall']>=target]
        if not covering: continue
        point=min(covering,key=key)
        if point not in frozen['selected']: raise RuntimeError('unfrozen configuration selected')
        if not rows or (point['L'],point['r'])!=(rows[-1]['L'],rows[-1]['r']): rows.append(point)
    return rows


def interpolate(rows,target):
    # First crossing along the predeclared training family. Never pick a
    # faster bracket using held-out measurements. Quantiles stay raw only.
    if len(rows)==1 and abs(rows[0][1]['recall']-target)<1e-10:
        a=b=rows[0]; w=0.
    else:
        bracket=next(((a,b) for a,b in zip(rows,rows[1:]) if a[1]['recall']<=target<=b[1]['recall']),None)
        if bracket is None: return None
        a,b=bracket
        w=(target-a[1]['recall'])/(b[1]['recall']-a[1]['recall']) if b[1]['recall']!=a[1]['recall'] else 0.
    return dict(values={k:a[1][k]+w*(b[1][k]-a[1][k]) for k in METRICS},
        endpoints=[dict(label=label,L=data['L'],r=data['rerank_limit'],recall=data['recall'],
                        p95_us=data['p95_us']) for label,data in (a,b)])


def summarize(root, manifest_tag=''):
    root=Path(root)
    records=[]
    complete={}
    for nav,media in [('exact',['DRAM']),('pq',['DRAM','SSD'])]:
        frozen=read(root/'analysis'/('frozen-full-T_XY-'+nav+manifest_tag+'.json'))
        rounds=[r for r in (1,2,3) if all(source(root,nav,m,p,r) is not None for m in media for p in frozen['selected'])]
        complete[nav]=rounds
        if not rounds: continue
        for family in ('pages','compute'):
            for filtering in (False,True):
                configs=curve(frozen,filtering,family)
                for medium in media:
                    observations=[]
                    for repeat in rounds:
                        rows=[source(root,nav,medium,x,repeat) for x in configs]
                        observations.append(interpolate(rows,95))
                    entry=dict(nav=nav,medium=medium,family=family,filtering=filtering,target_recall=95)
                    if any(x is None for x in observations):
                        entry['status']='UNCOVERED'
                    else:
                        entry.update(status='COVERED',rounds=rounds,observations=observations,
                            mean={k:statistics.mean(x['values'][k] for x in observations) for k in METRICS},
                            median={k:statistics.median(x['values'][k] for x in observations) for k in METRICS})
                    records.append(entry)
    contrasts=[]
    for off in records:
        if off['filtering'] or off['status']!='COVERED': continue
        on=next((x for x in records if x['filtering'] and x['status']=='COVERED' and
                 all(x[k]==off[k] for k in ('nav','medium','family'))),None)
        if not on: continue
        a,b=off['mean'],on['mean']
        evaluations='navigation_exact_space_evals' if off['nav']=='exact' else 'pq_space_evals'
        added=b['expanded_nodes']-a['expanded_nodes']
        contrasts.append(dict(nav=off['nav'],medium=off['medium'],family=off['family'],
            off_latency_over_on=a['latency_us']/b['latency_us'],
            absolute_latency_saved_us=a['latency_us']-b['latency_us'],
            extra_expansions_with_filter=added,saved_navigation_space_evals=a[evaluations]-b[evaluations],
            saved_space_evals_per_extra_expansion=(a[evaluations]-b[evaluations])/added if added>0 else None,
            cells=[dict(filtering=f,
                alpha_reject_fraction=d['alpha_rejected_edges']/d['inspected_edges'] if d['inspected_edges'] else None,
                amortized_topology_service_us_per_page=d['topology_submit_wait_us']/d['topology_submitted_pages'] if d['topology_submitted_pages'] else None,
                amortized_coordinate_service_us_per_page=d['coordinate_submit_wait_us']/d['coordinate_submitted_pages'] if d['coordinate_submitted_pages'] else None,
                coordinate_pages_per_reranked_node=d['coordinate_submitted_pages']/d['reranked_nodes'] if d['reranked_nodes'] else None)
                for f,d in [(False,a),(True,b)]]))
    c.save(root/'analysis'/('six-cell-summary'+manifest_tag+'.json'),dict(dataset='Laion',alpha=read(root/'analysis/tuning-split-v2.json')['alpha'],
        generated_unix=time.time(),target_recall=95,threads=4,numa=3,complete_rounds=complete,
        records=records,contrasts=contrasts,
        notes=['Full original ID order; corrected weighted-Euclidean graph; no shared cache or Starling.',
               'Full curves and L/r endpoints retained; unavailable crossings are not extrapolated.',
               'Tail latency is available at raw points only, never interpolated.',
               'DRAM submitted pages mean page-service requests, not physical SSD I/O.',
               'Service time/page is amortized submit-and-wait batch cost, not single-request device latency.',
               'Historical 37.48% rejection is not assumed for the rebuilt graph.']))


if __name__=='__main__':
    import sys
    summarize(sys.argv[1])
