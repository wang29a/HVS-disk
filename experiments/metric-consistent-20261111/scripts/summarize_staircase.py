"""Summarize accepted, frozen configurations without selecting on report latency.

Reconstruct the two deterministic tuning families from the frozen training
frontier. Use only complete rounds for each search type, and retain the
measured points bracketing each interpolated recall target. No tail-quantile
interpolation and no inference from incomplete or rejected timing records.
"""
import json
from pathlib import Path
import statistics
import time

ROOT = Path(__file__).resolve().parent.parent
METRICS = ('latency_us', 'qps', 'submitted_pages', 'topology_submitted_pages',
           'coordinate_submitted_pages', 'pq_space_evals', 'navigation_exact_space_evals',
           'topology_submit_wait_us', 'coordinate_submit_wait_us', 'pq_lookup_us', 'exact_us',
           'expanded_nodes', 'pq_comparisons', 'exact_space_evals', 'reranked_nodes')


def read(path):
    return json.loads(path.read_text())


def label(nav, medium, config, repeat):
    return 'report-time-full-T_XY-{}-{}-f{}-L{}-r{}-repeat{}'.format(
        nav, medium, int(config['filtering']), config['L'], config['r'], repeat)


def point(nav, medium, config, repeat):
    name = label(nav, medium, config, repeat)
    if not (ROOT/'analysis'/(name+'-accepted.json')).exists():
        return None
    data = read(ROOT/'results'/(name+'.json'))[0]
    return name, data


def family_curve(frozen, filtering, family):
    candidates = [p for p in frozen['tuning_frontier'] if p['filtering'] == filtering]
    key = (lambda p:(p['pages'], p['pq']+p['exact'], p['L'], p['r'])) if family == 'pages' else (
        lambda p:(p['pq']+p['exact'], p['pages'], p['L'], p['r']))
    curve = []
    for target in (80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99):
        covering = [p for p in candidates if p['recall'] >= target]
        if not covering:
            continue
        config = min(covering, key=key)
        if not any((p['filtering'], p['L'], p['r']) == (config['filtering'], config['L'], config['r'])
                   for p in frozen['selected']):
            raise RuntimeError('training representative was not frozen for timing')
        if not curve or (config['L'], config['r']) != (curve[-1]['L'], curve[-1]['r']):
            curve.append(config)
    return curve


def matched(nav, medium, curve, repeat, target):
    rows = [point(nav, medium, config, repeat) for config in curve]
    if any(p is None for p in rows):
        raise RuntimeError('round completeness gate failed')
    brackets = [(a, b) for a, b in zip(rows, rows[1:]) if a[1]['recall'] <= target <= b[1]['recall']]
    if not brackets:
        return None
    # First crossing along the predeclared training curve; never choose a
    # bracket or configuration by the reporting set's measured latency.
    a, b = brackets[0]
    weight = (target-a[1]['recall'])/(b[1]['recall']-a[1]['recall']) if b[1]['recall'] != a[1]['recall'] else 0
    return dict(values={k:a[1][k]+weight*(b[1][k]-a[1][k]) for k in METRICS},
                source_points=[dict(label=x[0], recall=x[1]['recall'], L=x[1]['L'], r=x[1]['rerank_limit']) for x in (a, b)],
                crossing_count=len(brackets))


def main():
    records, completed = [], {}
    for nav, media in [('exact', ['DRAM']), ('pq', ['DRAM', 'SSD'])]:
        frozen = read(ROOT/'analysis'/('frozen-full-T_XY-'+nav+'.json'))
        rounds = [r for r in (1, 2, 3) if all(point(nav, m, p, r) is not None
                  for m in media for p in frozen['selected'])]
        completed[nav] = rounds
        if not rounds:
            continue
        for family in (['pages'] if nav == 'exact' else ['pages', 'compute']):
            for filtering in (False, True):
                curve = family_curve(frozen, filtering, family)
                for medium in media:
                    for target in (90, 95):
                        observations = [matched(nav, medium, curve, repeat, target) for repeat in rounds]
                        if any(p is None for p in observations):
                            records.append(dict(nav=nav, medium=medium, family=family,
                                                filtering=filtering, recall=target, status='UNCOVERED'))
                            continue
                        records.append(dict(nav=nav, medium=medium, family=family, filtering=filtering,
                            recall=target, status='COVERED', rounds=rounds,
                            mean={k:statistics.mean(p['values'][k] for p in observations) for k in METRICS},
                            median={k:statistics.median(p['values'][k] for p in observations) for k in METRICS},
                            observations=observations))
    report = dict(generated_unix=time.time(), dataset='OpenImages', layout='Full T_XY, original ID order',
                  queries='500 held-out report queries; training and frozen configuration selection separate',
                  threads=4, numa=3, complete_rounds=completed,
                  aggregation='within-round linear recall interpolation, then across-round mean/median',
                  caveat='DRAM page services are not physical SSD reads; family ranges are not error bars',
                  records=records)
    path = ROOT/'analysis/staircase-summary.json'
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(report, indent=2)+'\n')
    tmp.replace(path)
    print(json.dumps(dict(complete_rounds=completed,
        recall95=[{k:v for k,v in row.items() if k not in ('observations',)}
                  for row in records if row['recall'] == 95]), indent=2))


if __name__ == '__main__':
    main()
