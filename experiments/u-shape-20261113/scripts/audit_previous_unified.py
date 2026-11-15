#!/usr/bin/env python3
"""Apply the same read-only audit to the prior corrected-metric source graphs."""
import fcntl,pathlib,time
import stage2_queue as Q
def main():
    Q.status=lambda stage,**kw:Q.save(Q.O/'prior-audit-status.json',dict(stage=stage,pid=__import__('os').getpid(),updated_unix=time.time(),**kw))
    lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+')
    deadline=time.monotonic()+6*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:raise RuntimeError('audit lock >6h')
            time.sleep(10)
    jobs=[('openimage',Q.E/'metric-consistent-20261111/scratch/dual/float-graph.bin',pathlib.Path('/mnt/nvme3/wz/dataset/Openimage_disk'),12),
          ('laion-full',pathlib.Path('/mnt/nvme2/wz/hvs-laion-six-cell-20261111/full/float-graph.bin'),pathlib.Path('/mnt/nvme2/wz/disk/dataset/Laion'),96)]
    summary={}
    for name,graph,data,minimum in jobs:
        if not graph.exists():summary[name]=dict(stage='pending',reason='source graph absent');continue
        node=pathlib.Path('/sys/devices/system/node/node3/meminfo').read_text().splitlines()
        free=int(next(x.split()[-2] for x in node if 'MemFree:' in x))*1024
        if free<minimum*2**30:summary[name]=dict(stage='pending',reason='NUMA3 free RAM below audit guard',required_GiB=minimum,free_GiB=free/2**30);continue
        out=Q.O/'analysis'/('prior-'+name+'-final-audit-v1.json')
        if not out.exists():
            Q.execute([Q.O/'bin/audit-unified',graph,data/'base_img_emb.fvecs',data/'base_text_emb.fvecs',out,4],'audit-prior-'+name,{},False,3600,'72-75')
        x=Q.load(out);summary[name]={k:v for k,v in x.items() if k!='examples'}
        Q.save(Q.O/'analysis/prior-unified-audit-summary.json',summary)
    Q.save(Q.O/'analysis/prior-unified-audit-summary.json',summary)
    Q.status('complete',summary='analysis/prior-unified-audit-summary.json')
if __name__=='__main__':main()
