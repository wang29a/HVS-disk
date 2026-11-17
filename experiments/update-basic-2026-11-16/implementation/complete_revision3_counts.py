"""Reuse completed diagnostics; extend exact radius checks to unreachable IDs."""
import fcntl
import json
from pathlib import Path
import run_g0 as c
import run_revision3_diagnostics as d


def main():
    root=c.ROOT/'runtime/revision3-diagnostics'
    lock=(c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    result=json.loads((root/'diagnostic-results.json').read_text())
    existing={v['id']:v for v in map(json.loads,(root/'low-degree-duplicate-evidence.jsonl').read_text().splitlines())}
    all_unreachable=set()
    by_graph={}
    for row in result['Laion2M']:
        label=row['graph']+'-'+row['entry_policy']
        ids={int(x) for x in (root/(label+'-unreachable.txt')).read_text().split()}
        by_graph[label]=ids
        all_unreachable.update(ids)
    missing=all_unreachable-set(existing)
    extra=root/'extra-unreachable'
    extra.mkdir(exist_ok=True)
    c.RUN=extra
    if missing:
        x=d.matrix(d.DATA/'base_img_emb.fvecs');y=d.matrix(d.DATA/'base_text_emb.fvecs')
        existing.update(d.radius_matches(x,y,missing))
    for row in result['Laion2M']:
        ids=by_graph[row['graph']+'-'+row['entry_policy']]
        row['unreachable_exact_zero_both']=sum(existing[v]['exact_zero_both'] for v in ids)
        row['unreachable_lt_1e_6_both']=sum(existing[v]['lt_1e_6_both'] for v in ids)
    result['distance_inputs']={str(p):c.sha(p) for p in (d.DATA/'base_img_emb.fvecs',d.DATA/'base_text_emb.fvecs',d.DATA/'subset.json')}
    result['radius_coverage']=dict(original_low_degree_targets=len(existing)-len(missing),
        extra_unreachable_targets=len(missing),total_unique_targets=len(existing),
        definition='both low-degree points and all unreachable points; count of points with at least one other qualifying point',
        zero_is_subset_of_lt_1e_6=True)
    (root/'diagnostic-results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(extra_targets=len(missing),total_targets=len(existing))))


if __name__=='__main__':main()
