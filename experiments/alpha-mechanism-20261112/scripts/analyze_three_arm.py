"""Local matched-recall layout comparison from already completed runs."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
TARGETS=(80,85,88,90,92,94,95,96,97,98,99)


def analyze(source):
    result=[]
    for record in json.loads(Path(source).read_text())['records']:
        arm,kind=record['arm'],record['layout'];frozen=record['frozen']
        filtering=arm=='full'
        for family in ('pages','compute'):
            # These are the representatives actually frozen by train-only
            # work rules. Equal-work, different-r points can disappear from
            # the Pareto frontier while remaining valid frozen selections.
            candidates=[x for x in frozen['selected'] if x['filtering']==filtering]
            key=(lambda x:(x['pages'],x['pq']+x['exact'],x['L'],x['r'])) if family=='pages' else (
                 lambda x:(x['pq']+x['exact'],x['pages'],x['L'],x['r']))
            curve=[]
            for target in TARGETS:
                covering=[x for x in candidates if x['recall']>=target]
                if covering:
                    c=min(covering,key=key)
                    if not curve or (c['L'],c['r'])!=(curve[-1]['L'],curve[-1]['r']):curve.append(c)
            for target in (90,95):
                observations=[];endpoints=[]
                for repeat in (1,2,3):
                    labels=['report-time-{}-{}-pq-SSD-f{}-L{}-r{}-repeat{}'.format(arm,kind,int(filtering),x['L'],x['r'],repeat) for x in curve]
                    if not all(x in record['points'] for x in labels):break
                    rows=[record['points'][x] for x in labels]
                    crossing=next((i for i in range(len(rows)-1) if rows[i]['recall']<=target<=rows[i+1]['recall']),None)
                    if crossing is None:break
                    a,b=rows[crossing],rows[crossing+1]
                    w=(target-a['recall'])/(b['recall']-a['recall']) if b['recall']!=a['recall'] else 0
                    observations.append({k:a[k]+w*(b[k]-a[k]) for k in a if k!='recall'})
                    endpoints.append(dict(repeat=repeat,weight=w,points=[dict(label=labels[crossing+j],
                        **{k:p[k] for k in ('L','rerank_limit','recall','reranked_nodes','coordinate_submitted_pages','coordinate_page_hits')}) for j,p in enumerate((a,b))]))
                row=dict(arm=arm,layout=kind,family=family,recall=target,status='COVERED' if len(observations)==3 else 'INCOMPLETE_OR_UNCOVERED')
                if len(observations)==3:row.update(mean={k:sum(x[k] for x in observations)/3 for k in observations[0]},observations=observations,interpolation_endpoints=endpoints)
                result.append(row)
    contrasts=[]
    for x in result:
        if x['layout']!='TXY' or x['status']!='COVERED':continue
        y=next((z for z in result if z['layout']=='T_XY' and z['status']=='COVERED' and all(z[k]==x[k] for k in ('arm','family','recall'))),None)
        if y:contrasts.append(dict(arm=x['arm'],family=x['family'],recall=x['recall'],
            coupled=x['mean'],separated=y['mean'],latency_change_pct=(y['mean']['latency_us']/x['mean']['latency_us']-1)*100,
            qps_change_pct=(y['mean']['qps']/x['mean']['qps']-1)*100))
    output=dict(dataset='OpenImages',threads=4,numa=3,repeats=3,records=result,contrasts=contrasts,
        input=source,scope='Completed equal-total-dimension single graph vs dual no-alpha vs dual Full; each compares TXY to T_XY under its own objective/GT. No rerun.')
    dest=ROOT/'analysis/three-arm-layout-summary.json';dest.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':analyze(sys.argv[1])
