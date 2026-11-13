"""Append a declared lower-L grid; keep original frozen manifests and accepted runs.

Motivation: the coarse training grid already starts above 95% for exact/off.
Run both filtering settings and both weights; never select by held-out latency.
"""
import fcntl
import os
from pathlib import Path
import struct
import time
import common as c
import run_queue as q
import phase_runtime as p
from summarize import summarize


def freeze(rows,exact):
    selected,frontier=[],[]
    for filtering in (False,True):
        candidates=[x for x in rows if x['filtering']==filtering]
        for x in candidates:
            if not any(y['recall']>=x['recall'] and y['pages']<=x['pages'] and
                y['pq']+y['exact']<=x['pq']+x['exact'] and
                (y['recall']>x['recall'] or y['pages']<x['pages'] or y['pq']+y['exact']<x['pq']+x['exact'])
                for y in candidates): frontier.append(x)
        for target in q.TARGETS:
            covering=[x for x in candidates if x['recall']>=target]
            if not covering: continue
            for key in (lambda x:(x['pages'],x['pq']+x['exact'],x['L'],x['r']),
                        lambda x:(x['pq']+x['exact'],x['pages'],x['L'],x['r'])):
                x=min(covering,key=key)
                if x not in selected: selected.append(x)
        if not any(x['filtering']==filtering for x in selected):
            selected.append(max(candidates,key=lambda x:x['recall']))
    return dict(arm='full',layout='T_XY',exact=exact,selected=selected,tuning_frontier=frontier,
        all_training_points=rows,targets=q.TARGETS,tune_queries=500,report_queries=500,
        selection='same two deterministic training-only work rules; added L24/L32 for both filters and weights',
        frozen_unix=time.time(),split_sha256=c.digest(p.ROOT/'analysis/tuning-split-v2.json'))


def run():
    missing=[]
    for workload in q.read(c.ROOT/'analysis/inventory.json')['workloads']:
        root,split=q.prepare_workload(workload)
        c.ACTIVE=root
        p.ROOT,p.B,p.SPLIT,p.COORD,p.PQ=root,c,split,c.COORD,str(q.PQ)+'/'
        p.layout=q.layout
        q.await_memory(80<<30,'lower-L-coverage')
        manifests=[]
        for exact,media in [(True,['DRAM']),(False,['DRAM','SSD'])]:
            nav='exact' if exact else 'pq'
            original=root/'analysis'/('frozen-full-T_XY-'+nav+'.json')
            if not original.exists():
                missing.append(str(original));continue
            output=root/'analysis'/('frozen-full-T_XY-'+nav+'-coverage-v2.json')
            if output.exists():
                manifest=q.read(output)
            else:
                rows=list(q.read(original)['all_training_points'])
                with (q.GRAPH/'full.meta').open('rb') as source:
                    source.seek(28); entries=struct.unpack('<I',source.read(4))[0]
                if entries>24: raise RuntimeError('lower-L protocol requires reviewing entry count')
                for L in (24,32):
                    for r in ([L] if exact else [20,L]):
                        for filtering in (False,True):
                            label='coverage-tune-{}-f{}-L{}-r{}'.format(nav,int(filtering),L,r)
                            point=p.query('full','T_XY','DRAM',exact,filtering,L,r,'tune',label)
                            rows.append(dict(filtering=filtering,L=L,r=r,recall=point['recall'],
                                pages=point['submitted_pages'],pq=point['pq_space_evals'],
                                exact=point['navigation_exact_space_evals'],rerank_evals=point['exact_space_evals'],
                                source=str(root/'results'/(label+'.json'))))
                manifest=freeze(rows,exact)
                manifest.update(original_frozen_sha256=c.digest(original),extension_script_sha256=c.digest(Path(__file__)),
                    protocol_note='Prospective lower-L amendment applied symmetrically; prior frozen manifests retained.')
                c.save(output,manifest)
            p.measure(manifest,media)
            manifests.append(nav)
        if len(manifests)==2: summarize(root,'-coverage-v2')
        c.save(root/'analysis/coverage-extension-status.json',dict(completed_unix=time.time(),
            completed_arms=manifests,pending=p.pending_points(),missing_original_manifests=missing))
    c.ACTIVE=c.ROOT
    c.save(c.ROOT/'coverage-queue-status.json',dict(stage='complete_with_missing' if missing else 'complete',
        updated_unix=time.time(),missing_original_manifests=missing))


if __name__=='__main__':
    c.init()
    c.save(c.ROOT/'coverage-queue-status.json',dict(stage='queued',pid=os.getpid(),updated_unix=time.time()))
    lock=open(str(c.ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        c.save(c.ROOT/'coverage-queue-status.json',dict(stage='running',pid=os.getpid(),updated_unix=time.time()))
        run()
    except Exception as error:
        c.save(c.ROOT/'coverage-queue-status.json',dict(stage='failed',error=str(error),updated_unix=time.time()))
        raise
