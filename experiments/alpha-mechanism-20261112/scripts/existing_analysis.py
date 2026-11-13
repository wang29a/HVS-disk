"""Read-only existing results: graph activity, OI fixed-parameter bins, trace provenance."""
import json
from pathlib import Path
import re
import struct
import sys

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'laion-six-cell-20261111'
OI=ROOT.parent/'metric-consistent-20261111'
sys.path.insert(0,str(OLD/'scripts'))
import common as c
c.ROOT=ROOT
c.ACTIVE=ROOT


def read(path):return json.loads(Path(path).read_text())


def bin_index(alpha):return min(4,int(float(alpha)*5))


def openimages_bins():
    pattern=re.compile(r'report-work-full-T_XY-pq-f([01])-L(\d+)-r(\d+)\.json\.L\d+\.queries\.json$')
    grouped={}
    for path in (OI/'results').glob('report-work-full-T_XY-pq-*.queries.json'):
        match=pattern.match(path.name)
        if match:
            f,L,r=map(int,match.groups());grouped.setdefault((L,r),{})[f]=(path,read(path))
    with (OI/'scratch/dual/full.meta').open('rb') as source:
        source.seek(28);entries=struct.unpack('<I',source.read(4))[0]
    reports=[]
    for (L,r),pair in sorted(grouped.items()):
        if set(pair)!={0,1}:continue
        a,b=pair[0][1],pair[1][1]
        if len(a)!=len(b) or any(x['query_id']!=y['query_id'] or x['alpha']!=y['alpha'] for x,y in zip(a,b)):
            raise RuntimeError('unpaired OI queries')
        bins=[]
        for k in range(5):
            rows=[(x,y) for x,y in zip(a,b) if bin_index(x['alpha'])==k]
            if not rows:continue
            columns=[]
            for side in (0,1):
                values=[v[side] for v in rows];n=len(values)
                columns.append(dict(expanded_nodes=sum(x['expanded_nodes'] for x in values)/n,
                    pq_space_evals=sum(2*(x['pq_comparisons']+entries) for x in values)/n,
                    inspected_edges=sum(x['inspected_edges'] for x in values)/n,
                    alpha_rejected_edges=sum(x['alpha_rejected_edges'] for x in values)/n,
                    submitted_pages=sum(x['submitted_pages'] for x in values)/n))
            off,on=columns;added=on['expanded_nodes']-off['expanded_nodes']
            bins.append(dict(alpha_bin=[k/5.,(k+1)/5.],queries=len(rows),off=off,on=on,
                saved_space_evals_per_extra_expansion=(off['pq_space_evals']-on['pq_space_evals'])/added if added>0 else None))
        reports.append(dict(L=L,r=r,sources=[str(pair[i][0]) for i in (0,1)],bins=bins))
    c.save(ROOT/'analysis/openimages-alpha-bins.json',dict(dataset='OpenImages',paired_fixed_parameter_reports=reports,
        queries='existing 500 held-out mixed-alpha queries',bin_convention='[lo,hi), except last includes 1',
        entry_count=entries,pq_evaluation_formula='2*(per-query unique neighbor comparisons + entry count)',
        limits='Per-query files do not contain hits/recall. These are paired fixed-L/r work counters, NOT matched-recall bucket comparisons. One-pass latency is not reported.'))


def existing_traces():
    for alpha in (.5,.1):
        w=OLD/'workloads'/('alpha-'+str(alpha))
        split=read(w/'analysis/tuning-split-v2.json')['cohorts']['tune']['files']
        prefix=w/'analysis/parity-f0-DRAM.L80'
        output=ROOT/'analysis'/('existing-off-alpha-{}-L80-r40-parents.json'.format(alpha))
        c.run([ROOT/'bin/edge-audit','trace',OLD/'scratch/full/full.meta',OLD/'scratch/full/full.topo',
               prefix,split['alpha']['path'],str(prefix)+'.ids',split['gt']['path'],output],
              'existing-parents-alpha-'+str(alpha),[72])


def main():
    (ROOT/'analysis').mkdir(parents=True,exist_ok=True)
    c.run([ROOT/'bin/edge-audit','static',OLD/'scratch/full/full.meta',OLD/'scratch/full/full.topo',
           ROOT/'analysis/laion-edge-activity.json'],'laion-static-activity',[72])
    openimages_bins()
    existing_traces()
    rows=[]
    for path in sorted((ROOT/'analysis').glob('existing-off-alpha-*-parents.json')):
        a=read(path)
        sums={key:sum(x[key] for x in a) for key in a[0] if key not in ('alpha','query_id')}
        fractions={label:sums[num]/sums[den] if sums[den] else None for label,num,den in (
            ('inactive_inspected_fraction','inactive_inspected_edges','inspected_edges'),
            ('inactive_discovery_fraction','inactive_first_discovery_edges','first_discovery_edges'),
            ('inactive_candidate_parent_fraction','candidate_inactive_parent','candidate_nonentry'),
            ('inactive_returned_parent_fraction','returned_inactive_parent','returned_nonentry'),
            ('inactive_true_returned_parent_fraction','true_inactive_parent','true_nonentry'))}
        rows.append(dict(alpha=a[0]['alpha'],queries=len(a),L=80,r=40,cohort='tuning parity trace; NOT matched recall',
            source=str(path),sums=sums,pooled_fractions=fractions))
    c.save(ROOT/'analysis/existing-trace-summary.json',dict(records=rows,
        definition='First-discovery edge for eventual rerank candidates/returned IDs; seeds excluded.',
        limit='Observational provenance only. An inactive first-discovery edge is not proof that the candidate becomes unreachable under filtering; alternative paths exist.'))
    c.save(ROOT/'analysis/existing-analysis-status.json',dict(status='complete'))


if __name__=='__main__':main()
