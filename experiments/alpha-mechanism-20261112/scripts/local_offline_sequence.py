"""Local stages 1→2→3, then prepare independent two-index oracle coverage.

node3 only serves existing files. Poll for stage-4-derived counters every three
minutes (six-hour bound); do not launch remote search or statistical work.
"""
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import numpy as np
import analyze_distance_scale as scale

ROOT=Path(__file__).resolve().parent.parent
REMOTE='/mnt/nvme3/wz/hvs-disk-build-20261027/experiments/alpha-mechanism-20261112'
EXPERIMENTS='/mnt/nvme3/wz/hvs-disk-build-20261027/experiments'
SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=10','node3']


def save(name,data):
    dest=ROOT/'analysis'/name;tmp=dest.with_suffix(dest.suffix+'.partial')
    tmp.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n');tmp.replace(dest)


def status(stage,**extra):save('offline-status.json',dict(stage=stage,updated_unix=time.time(),**extra))


def copy_from(source,dest):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10','node3:'+source,str(dest)],check=True,timeout=120)


def upload(local,remote):
    subprocess.run(['scp','-q','-o','BatchMode=yes','-o','ConnectTimeout=10',str(local),'node3:'+remote],check=True,timeout=120)


def local_selectivity():
    start=time.monotonic()
    needed=[REMOTE+'/workloads/alpha-'+str(a)+'/analysis/first-discovery-summary.json' for a in (.1,.5)]
    while True:
        command=' && '.join('test -f '+shlex.quote(x) for x in needed)
        ready=subprocess.run(SSH+[command],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=30)
        if ready.returncode==0:break
        status('step_2_waiting_for_existing_report_traces',elapsed_seconds=time.monotonic()-start)
        if time.monotonic()-start>21600:raise RuntimeError('stage-4 trace counters unavailable after six hours')
        time.sleep(180)
    samples=ROOT/'snapshots';summaries=[]
    for a,src in zip((.1,.5),needed):
        dest=samples/('discovery-summary-'+str(a)+'.json');copy_from(src,dest)
        summaries.append(json.loads(dest.read_text()))
    candidates=[]
    for a in summaries[0]['records']:
        for b in summaries[1]['records']:
            if a['r']==b['r'] and abs(a['recall']-95)<=2 and abs(b['recall']-95)<=2:
                candidates.append((abs(a['recall']-95)+abs(b['recall']-95)+abs(a['recall']-b['recall']),a,b))
    if not candidates:raise RuntimeError('no comparable existing report recall configurations')
    _,chosen_a,chosen_b=min(candidates,key=lambda x:x[0])
    records=[];arrays=[]
    for alpha,chosen in zip((.1,.5),(chosen_a,chosen_b)):
        source=REMOTE+'/workloads/alpha-'+str(alpha)+'/analysis/parents-off-L{}-r{}.json'.format(chosen['L'],chosen['r'])
        dest=samples/('selected-parent-counters-'+str(alpha)+'.json');copy_from(source,dest)
        data=json.loads(dest.read_text());sums={k:sum(x[k] for x in data) for k in data[0] if k not in ('query_id','alpha')}
        def ratio(num,den):return sums[num]/sums[den] if sums[den] else None
        rates=dict(rejected_inspected=ratio('inactive_inspected_edges','inspected_edges'),
            rejected_unique_evaluated=ratio('inactive_first_discovery_edges','first_discovery_edges'),
            rejected_candidate_parents=ratio('candidate_inactive_parent','candidate_nonentry'),
            rejected_returned_true_parents=ratio('true_inactive_parent','true_nonentry'))
        rates['selectivity_gap']=rates['rejected_inspected']-rates['rejected_candidate_parents']
        rates['retained_candidate_enrichment']=(1-rates['rejected_candidate_parents'])/(1-rates['rejected_inspected'])
        records.append(dict(alpha=alpha,L=chosen['L'],r=chosen['r'],recall=chosen['recall'],queries=len(data),rates=rates,counts=sums))
        arrays.append(np.asarray([[x[k] for k in ('inactive_inspected_edges','inspected_edges','candidate_inactive_parent','candidate_nonentry')] for x in data],dtype=np.float64))
    if abs(records[0]['recall']-records[1]['recall'])>1:raise RuntimeError('selected recall mismatch exceeds 1 percentage point')
    rng=np.random.RandomState(20261112);ids=rng.randint(0,len(arrays[0]),size=(2000,len(arrays[0])))
    gaps=[]
    for x in arrays:
        totals=x[ids].sum(axis=1);gaps.append(totals[:,0]/totals[:,1]-totals[:,2]/totals[:,3])
    result=dict(records=records,paired_query_bootstrap_repeats=2000,
        difference_in_selectivity_gap_alpha01_minus_alpha05=dict(mean=records[0]['rates']['selectivity_gap']-records[1]['rates']['selectivity_gap'],
            ci95=list(map(float,np.percentile(gaps[0]-gaps[1],[2.5,97.5])))),
        compute='local aggregation of existing stage-4 trace-derived counters',
        definitions='Inspected adjacency edges include duplicates; unique evaluated edges are all first discoveries. Candidate-parent rates exclude seeds.',
        limit='Descriptive protection of final candidates; first-discovery edges are not necessarily indispensable paths.')
    save('filter-selectivity.json',result)


def stage_three():
    data=json.loads((ROOT/'analysis/preliminary-aggregates.json').read_text())['activity']
    if len(data['activity'])!=101 or any(not 0<=x['active_edges']<=x['total_edges'] for x in data['activity']):raise RuntimeError('static activity curve invalid')
    save('activity-step-three.json',dict(source='previous completed full scan of corrected Full topology',
        edges=data['edges'],points=[dict(alpha=x['alpha100']/100,active_fraction=x['active_edges']/x['total_edges']) for x in data['activity']],
        compute='local read of existing counts; no new index scan on experiment host'))


def prepare_oracle_base():
    target=ROOT/'snapshots/openimage-full-concat.bin'
    expected=8+456699*1536*4
    if not target.exists():
        temporary=target.with_suffix('.partial')
        source='/mnt/nvme2/wz/hvs-metric-consistent-20261111/concat-base.bin'
        command='numactl --all --physcpubind=76 --membind=3 nice -n 19 ionice -c 2 -n 7 cat '+shlex.quote(source)
        # This is an independent read-only nvme2 transfer, not a benchmark.
        # Do not acquire or block the search queue's nvme3 experiment lock.
        with temporary.open('wb') as out,(ROOT/'analysis/coverage-transfer.log').open('w') as log:
            subprocess.run(SSH+[command],stdout=out,stderr=log,check=True,timeout=21600)
        if temporary.stat().st_size!=expected:raise RuntimeError('coverage base transfer size mismatch')
        temporary.replace(target)
    if target.stat().st_size!=expected:raise RuntimeError('coverage base size')
    save('coverage-source-status.json',dict(status='ready',bytes=expected,local_path=str(target),
        preparation='independent read-only nvme2 transfer, one low-priority core; coverage compute local'))
    upload(ROOT/'analysis/coverage-source-status.json',REMOTE+'/analysis/coverage-source-status.json')
    return target


def main():
    (ROOT/'analysis').mkdir(exist_ok=True)
    status('step_1_local_distance_scale')
    scale.main(ROOT/'snapshots')
    status('step_2_local_selectivity')
    local_selectivity()
    status('step_3_local_activity_counts')
    stage_three()
    save('offline-first-three-done.json',dict(status='complete',completed_unix=time.time(),
        outputs=['distance-scale.json','filter-selectivity.json','activity-step-three.json'],compute_host='local workstation'))
    for name in ('distance-scale.json','filter-selectivity.json','activity-step-three.json','offline-first-three-done.json'):
        upload(ROOT/'analysis'/name,REMOTE+'/analysis/'+name)
    status('preparing_independent_two_index_oracle_source')
    try:base=prepare_oracle_base()
    except Exception as error:
        save('coverage-source-status.json',dict(status='failed',error=str(error)))
        upload(ROOT/'analysis/coverage-source-status.json',REMOTE+'/analysis/coverage-source-status.json')
        status('first_three_complete_oracle_source_pending',error=str(error));return
    status('independent_two_index_oracle_running_locally')
    subprocess.run([sys.executable,ROOT/'scripts/dual_index_oracle.py',base],check=True,timeout=21600)
    status('complete')


if __name__=='__main__':
    try:main()
    except Exception as error:
        status('failed',error=str(error));raise
