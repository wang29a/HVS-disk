"""Read only prior accepted aggregate points; no experiment execution."""
import json
from pathlib import Path
import sys
ROOT=Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/metric-consistent-20261111')
METRICS=('L','rerank_limit','reranked_nodes','candidate_count','recall','latency_us','qps','submitted_pages',
         'topology_submitted_pages','coordinate_submitted_pages','topology_page_hits','coordinate_page_hits',
         'expanded_nodes','pq_space_evals','exact_space_evals','topology_submit_wait_us',
         'coordinate_submit_wait_us','exact_us')
records=[]
for arm in ('single','compact','full'):
    for kind in ('TXY','T_XY'):
        frozen=json.loads((ROOT/'analysis'/('frozen-{}-{}-pq.json'.format(arm,kind))).read_text())
        result={}
        for config in frozen['selected']:
            for repeat in (1,2,3):
                label='report-time-{}-{}-pq-SSD-f{}-L{}-r{}-repeat{}'.format(arm,kind,int(config['filtering']),config['L'],config['r'],repeat)
                if not (ROOT/'analysis'/(label+'-accepted.json')).exists():continue
                point=json.loads((ROOT/'results'/(label+'.json')).read_text())[0]
                result[label]={k:point[k] for k in METRICS}
        records.append(dict(arm=arm,layout=kind,frozen=frozen,points=result))
Path(sys.argv[1]).write_text(json.dumps(dict(source=str(ROOT),records=records)))
