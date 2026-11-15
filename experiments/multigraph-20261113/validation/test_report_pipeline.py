"""Synthetic report smoke test. Never reads or creates real benchmark results."""
import json
from pathlib import Path
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import common as c
import protocol as p
import report

def main():
    with tempfile.TemporaryDirectory(prefix='hvs-multigraph-report-smoke-') as temp:
        c.RUN=Path(temp)
        (c.RUN/'timings').mkdir()
        curves={};contribution={}
        for a in (.1,.3,.5,.7,.9):
            contribution[str(a)]={'report':{'mean':a}}
            for method in p.METHODS:
                key=str(a)+'-'+method;curves[key]=dict(alpha=a,method=method,frontiers={r:['low','high'] for r in p.RULES})
                for mode in ('ssd','topo'):
                    for rep in range(1,4):
                        for label,recall in [('low',.85),('high',.99)]:
                            pt=dict(label=label,recall=recall,latency_us=1000+rep,qps=3000+rep,reranked=20,coordinate_pages=10,coordinate_submitted_pages=10,coordinate_wait_us=20,candidates_before_union=40,candidates_after_union=30,p95_us=2000,peak_rss_kib=1024,resident_page_bytes=0,
                                graphs=[dict(expanded=30,pq_space_evals=300,topology_pages=15,topology_submitted_pages=15,topology_wait_us=50)])
                            c.save(c.RUN/'timings'/(key+'-'+label+'-'+mode+'-rep'+str(rep)+'.json'),{'point':pt})
        c.save(c.RUN/'frozen.json',dict(curves=curves));c.save(c.RUN/'data/contribution.json',contribution)
        for graph in ('x','y','a01','a03','a05','a07','a09','u'):
            c.save(c.RUN/'packed'/graph/'identity.json',dict(topology_bytes=8192,sidecar_bytes=128 if graph=='u' else 0,build_seconds=1,stats=dict(nodes=256,edges=1024)))
        report.main()
        summary=c.load(c.RUN/'report/summary.json')
        assert len(summary['records'])==360
        assert all(r['covered'] for r in summary['records'])
        assert all('p95_us' not in r for r in summary['records'])
        assert len(list((c.RUN/'report').glob('*.svg')))==12
        assert len(list((c.RUN/'report').glob('*.pdf')))==12
        assert len(summary['decision_checks'])==6
        print(json.dumps(dict(status='PASS',scope='synthetic fixture only; no performance evidence',matched_rows=360,svg_figures=12,pdf_figures=12,decision_groups=6)))
if __name__=='__main__':main()
