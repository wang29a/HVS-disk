"""After the five-weight scan: three arms × two layouts, topology-only DRAM.

Coordinates stay on SSD during timings. Shared frozen L/r within each arm.
"""
import fcntl
import json
from pathlib import Path
import re
import shutil
import struct
import sys
import time

ROOT=Path(__file__).resolve().parent.parent
OLD=ROOT.parent/'metric-consistent-20261111'
SCAN=ROOT.parent/'alpha-mechanism-20261112/weight-scan'
sys.path.insert(0,str(ROOT.parent/'laion-six-cell-20261111/scripts'))
import common as c
import phase_runtime as p
from summarize import interpolate
c.ROOT,c.ACTIVE=ROOT,ROOT
COORD=Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
PQ='/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/'
TARGETS=[88,90,92,94,95,96,97]
BASE_QUERY=p.query


def read(path):return json.loads(Path(path).read_text())


def graph(arm):
    folder=OLD/'scratch'/('single' if arm=='single' else 'dual')
    stem='full' if arm=='full' else 'compact'
    return folder/(stem+'.meta'),folder/(stem+'.topo')


def layout(arm,kind):
    meta,topo=graph(arm)
    folder=p.ROOT/'scratch'/(arm+'-'+kind);folder.mkdir(exist_ok=True)
    if kind=='T_XY':
        for name,target in [('group-1.bin',topo),('group-6.bin',COORD)]:
            link=folder/name
            if not link.exists():link.symlink_to(target)
            if link.resolve()!=target.resolve():raise RuntimeError('layout source link changed')
    elif kind=='TXY':
        identity=folder/'identity.json';file=folder/'group-7.bin'
        if not identity.exists() or not file.exists():
            # Only this queue's incomplete generated image may be replaced.
            if file.exists() and not identity.exists():file.unlink()
            n,d1,d2,R,M,npp=struct.unpack('<5IQ',meta.read_bytes()[:28])
            width=4+R*(4+2*M)+(d1+d2)*4
            expected=((n+(8192//width)-1)//(8192//width))*8192
            if shutil.disk_usage(str(ROOT)).free<expected+(512<<20):
                raise RuntimeError('one coupled image plus 512 MiB reserve does not fit; sources retained')
            c.run([ROOT/'bin/metric-search','build',meta,topo,COORD,folder,kind,identity],
                  'build-'+arm+'-coupled-image',c.CPUS,timeout=1800)
            if read(identity)['status']!='PASS' or file.stat().st_size!=expected:
                raise RuntimeError('coupled layout validation failed')
    else:raise RuntimeError('layout out of scope')
    return meta,topo,folder


def checked_query(*args,**kwargs):
    point=BASE_QUERY(*args,**kwargs)
    arm,kind,medium=args[:3]
    if point['topology_medium']!='DRAM' or point['topology_resident_bytes']!=graph(arm)[1].stat().st_size:
        raise RuntimeError('topology not fully resident')
    if point['topology_submitted_pages'] or abs(point['topology_memory_node_reads']-point['expanded_nodes'])>1e-6:
        raise RuntimeError('traversal submitted topology I/O')
    if point['coordinate_medium']!=medium:
        raise RuntimeError('wrong coordinate supply medium')
    if abs(point['coordinate_submitted_pages']-point['reranked_nodes'])>1e-6 or point['coordinate_page_hits']:
        raise RuntimeError('unexpected coordinate reuse/preload or page-density model')
    return point


def freeze(arm):
    output=p.ROOT/'analysis/frozen-common.json'
    if output.exists():return read(output)
    rows=[];filtering=arm=='full'
    pattern=re.compile(r'tune-'+arm+r'-T_XY-pq-f([01])-L(\d+)-r(\d+)\.json$')
    meta=graph(arm)[0].read_bytes();n,d1,d2=struct.unpack('<III',meta[:12])
    if 8192//((d1+d2)*4)!=1:raise RuntimeError('coordinate page-cost model requires one record/page')
    entries=struct.unpack('<I',meta[28:32])[0]
    for source in sorted((OLD/'results').glob('tune-'+arm+'-T_XY-pq-*.json')):
        match=pattern.match(source.name)
        if not match:continue
        f,L,r=map(int,match.groups())
        if bool(f)!=filtering or L<entries:continue
        point=read(source)[0]
        if point['queries']!=500:raise RuntimeError('not the original training split')
        rows.append(dict(filtering=filtering,L=L,r=r,recall=point['recall'],
            pages=point['reranked_nodes'],pq=point['pq_space_evals'],exact=0,
            expanded=point['expanded_nodes'],source=str(source)))
    if not rows:raise RuntimeError('no original train-only tuning records')
    selected=[]
    for target in TARGETS:
        covering=[x for x in rows if x['recall']>=target]
        if not covering:continue
        for key in (lambda x:(x['pages'],x['pq'],x['L'],x['r']),lambda x:(x['pq'],x['pages'],x['L'],x['r'])):
            point=min(covering,key=key)
            if point not in selected:selected.append(point)
    if not selected:raise RuntimeError('training recall uncovered')
    result=dict(arm=arm,targets=TARGETS,selected=selected,all_training_points=rows,
        source='existing independent 500-query training diagnostics; no held-out latency selection',
        page_cost='topology backend pages=0; unique coordinate pages=actual reranked nodes, one record/page',
        layout_control='same selected L/r for TXY and T_XY',frozen_unix=time.time())
    c.save(output,result)
    return result


def verify(frozen):
    arm=frozen['arm'];filtering=arm=='full'
    proof=p.ROOT/'analysis/real-input-parity.json'
    if proof.exists():return
    for kind in ('TXY','T_XY'):
        checked_query(arm,kind,'SSD',False,filtering,80,40,'report','parity-'+kind,trace=True)
    for suffix in ('','.ids','.dists'):
        a=p.ROOT/'analysis'/('parity-TXY.L80'+suffix)
        b=p.ROOT/'analysis'/('parity-T_XY.L80'+suffix)
        if a.read_bytes()!=b.read_bytes():raise RuntimeError('resident topology changed layout trajectory/score/IDs')
    for point in frozen['selected']:
        label='verify-train-L{}-r{}'.format(point['L'],point['r'])
        actual=checked_query(arm,'T_XY','DRAM',False,filtering,point['L'],point['r'],'tune',label)
        for k,expected in [('recall',point['recall']),('pq_space_evals',point['pq']),
                           ('expanded_nodes',point['expanded']),('coordinate_submitted_pages',point['pages'])]:
            if abs(actual[k]-expected)>1e-6:raise RuntimeError('predicted training work mismatch: '+k)
    c.save(proof,dict(status='PASS',bitwise_equal_layout_traces_ids_distances=True,
        trained_recall_and_work_reused_exactly=True,topology_backend_pages=0,
        timed_coordinate_medium='SSD',same_frozen_Lr=True))


def measure(frozen):
    arm=frozen['arm'];selected=frozen['selected'];deferred=[]
    for point in selected:
        for kind in ('TXY','T_XY'):
            label='report-work-{}-{}-pq-f{}-L{}-r{}'.format(arm,kind,int(point['filtering']),point['L'],point['r'])
            checked_query(arm,kind,'DRAM',False,point['filtering'],point['L'],point['r'],'report',label)
    for repeat in range(3):
        order=selected[repeat%len(selected):]+selected[:repeat%len(selected)]
        if repeat%2:order=list(reversed(order))
        for point in order:
            for kind in (('TXY','T_XY') if repeat%2==0 else ('T_XY','TXY')):
                label='report-time-{}-{}-pq-SSD-f{}-L{}-r{}-repeat{}'.format(arm,kind,int(point['filtering']),point['L'],point['r'],repeat+1)
                args=(arm,kind,'SSD',False,point['filtering'],point['L'],point['r'],'report',label)
                if not p.timed_or_defer(args):deferred.append(args)
    for args in deferred:p.timed_or_defer(args)


def summarize(frozen):
    arm=frozen['arm'];records=[]
    for family in ('pages','compute'):
        key=(lambda x:(x['pages'],x['pq'],x['L'],x['r'])) if family=='pages' else (lambda x:(x['pq'],x['pages'],x['L'],x['r']))
        curve=[]
        for target in TARGETS:
            covering=[x for x in frozen['selected'] if x['recall']>=target]
            if covering:
                x=min(covering,key=key)
                if not curve or (x['L'],x['r'])!=(curve[-1]['L'],curve[-1]['r']):curve.append(x)
        for kind in ('TXY','T_XY'):
            for target in (90,95):
                observations=[]
                for repeat in (1,2,3):
                    rows=[]
                    for x in curve:
                        label='report-time-{}-{}-pq-SSD-f{}-L{}-r{}-repeat{}'.format(arm,kind,int(x['filtering']),x['L'],x['r'],repeat)
                        if not (p.ROOT/'analysis'/(label+'-accepted.json')).exists():break
                        rows.append((label,read(p.ROOT/'results'/(label+'.json'))[0]))
                    if len(rows)!=len(curve):break
                    value=interpolate(rows,target)
                    if value is None:break
                    # Include matched-r and exact-distance work explicitly.
                    ends=value['endpoints'];raw=[x[1] for x in rows if x[0] in [z['label'] for z in ends]]
                    if len(raw)==1:raw=raw*2
                    weight=(target-raw[0]['recall'])/(raw[1]['recall']-raw[0]['recall']) if raw[1]['recall']!=raw[0]['recall'] else 0
                    value['values']['rerank_limit']=raw[0]['rerank_limit']+weight*(raw[1]['rerank_limit']-raw[0]['rerank_limit'])
                    observations.append(value)
                entry=dict(arm=arm,layout=kind,family=family,recall=target,status='COVERED' if len(observations)==3 else 'INCOMPLETE_OR_UNCOVERED')
                if len(observations)==3:entry.update(observations=observations,mean={k:sum(x['values'][k] for x in observations)/3 for k in observations[0]['values']})
                records.append(entry)
    c.save(p.ROOT/'analysis/resident-summary.json',dict(records=records,topology='canonical pure topology resident',
        coordinate='SSD native TXY/T_XY pages; no traversal-induced coordinate reuse',pending=p.pending_points()))


def run():
    c.init();start=time.monotonic()
    while True:
        upstream=read(SCAN/'status.json')
        if upstream['stage'] in ('complete','complete_with_gaps','failed'):break
        c.status('waiting_for_weight_scan',upstream_stage=upstream['stage'])
        if time.monotonic()-start>86400:raise RuntimeError('weight-scan wait exceeded 24 hours')
        time.sleep(60)
    lock=open(str(ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    proof=read(ROOT/'analysis/preflight.json')
    if proof['status']!='PASS' or proof['binary_sha256']!=c.digest(ROOT/'bin/metric-search'):raise RuntimeError('preflight/binary gate')
    c.save(ROOT/'analysis/upstream-finished.json',upstream)
    split=read(OLD/'analysis/tuning-split-v2.json')
    for cohort in split['cohorts'].values():
        for v in cohort['files'].values():
            if c.digest(v['path'])!=v['sha256']:raise RuntimeError('query/GT identity changed')
    all_summaries=[]
    for arm in ('single','compact','full'):
        work=ROOT/'arms'/arm
        for name in ('analysis','logs','results','scratch','bin'):(work/name).mkdir(parents=True,exist_ok=True)
        link=work/'bin/metric-search'
        if not link.exists():link.symlink_to(ROOT/'bin/metric-search')
        for cohort in ('tune','report'):
            dst=work/'scratch'/('single-'+cohort+'-gt.bin')
            if not dst.exists():dst.symlink_to(OLD/'scratch'/('single-'+cohort+'-gt.bin'))
        c.ACTIVE=work
        p.ROOT,p.B,p.SPLIT,p.COORD,p.PQ=work,c,split,COORD,PQ
        p.EXTRA=dict(p.EXTRA,LAYOUT_TOPO_DRAM='1')
        p.layout=layout;p.query=checked_query
        if (work/'analysis/done.json').exists() and not read(work/'analysis/done.json')['pending']:
            all_summaries.append(read(work/'analysis/resident-summary.json'));continue
        layout(arm,'TXY');layout(arm,'T_XY')
        frozen=freeze(arm);verify(frozen);measure(frozen);summarize(frozen)
        all_summaries.append(read(work/'analysis/resident-summary.json'))
        c.save(work/'analysis/done.json',dict(completed_unix=time.time(),pending=p.pending_points()))
        file=work/'scratch'/(arm+'-TXY')/'group-7.bin'
        c.save(work/'analysis/coupled-image-retired.json',dict(path=str(file),sha256=c.digest(file),bytes=file.stat().st_size,
            reproducible_from='original canonical graph and coordinates; layout identity retained'))
        file.unlink()
    c.ACTIVE=ROOT
    c.save(ROOT/'analysis/three-arm-resident-results.json',dict(arms=all_summaries,threads=4,numa=3,
        caveat='Topology-only DRAM control; intentionally removes traversal-driven coordinate prefetch, not a deployment-speedup claim.'))
    pending=any(x['pending'] for x in all_summaries)
    c.status('complete_with_pending' if pending else 'complete')


if __name__=='__main__':
    try:run()
    except Exception as error:
        c.status('failed',error=str(error));raise
