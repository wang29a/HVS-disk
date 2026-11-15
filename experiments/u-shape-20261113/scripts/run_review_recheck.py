#!/usr/bin/env python3
"""Fresh timing cohort on one quiet NUMA3 block; immutable tuning is reused."""
import fcntl,json,os,pathlib,shutil,signal,time
import stage2_queue as Q
import review_monitor as M
R=Q.R;PARENT=R/'review-20261113';V2=R/'stage2-v2';Q.O=PARENT/'cpu-recheck';O=Q.O
def status(stage,**kw):Q.save(O/'status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
Q.status=status
def main():
    for name in ('analysis','results','frozen','timings','commands','logs','bin','ssd'):(O/name).mkdir(parents=True,exist_ok=True)
    status('waiting_global_lock');lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+');deadline=time.monotonic()+6*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:status('deferred_lock_timeout');return
            time.sleep(10)
    choice=M.choose();Q.save(O/'affinity-choice.json',choice)
    # A busy node cannot produce accepted timings; wait at most 180s to avoid
    # occupying CPUs with measurements that are known to be uninformative.
    until=time.monotonic()+180
    while choice['choice']['external_busy_fraction']>.10:
        if time.monotonic()>until:status('deferred_no_quiet_numa3_block');return
        time.sleep(5);choice=M.choose();Q.save(O/'affinity-choice.json',choice)
    original=Q.execute
    def execute(cmd,label,extra=None,timed=False,timeout=1200,cpus='72-75'):
        return original(cmd,label,extra,timed,min(timeout,240) if timed else timeout,','.join(map(str,M.CPUS)))
    Q.execute=execute;Q.sample=M.sample;Q.validate_window=M.validate;Q.timings=lambda specs:M.timings(Q,specs)
    target=O/'bin/pointer-search'
    if not target.exists():target.symlink_to(R/'stage2/bin/pointer-search')
    for cohort,ids in Q.load(Q.D/'identity.json')['query_cohorts'].items():(O/(cohort+'-ids.txt')).write_text('\n'.join(map(str,ids))+'\n')
    for label in ('centre-U1','centre-S'):Q.save(O/'analysis'/(label+'-matched.json'),Q.load(PARENT/'analysis'/(label+'-matched.json')))
    u=Q.spec_laion('0.5',V2/'graphs/unified',True);s=Q.spec_laion('0.5',V2/'graphs/fixed-0.5');s['env']={'PTR_ENTRY_META':str(V2/'graphs/unified/full.meta')}
    # No timings or accepted files are copied from the prior CPU block.
    Q.timings({'centre-U1':u,'centre-S':s})
    reused={}
    for p in (PARENT/'results').glob('ssd-*.json'):
        if '-rep' in p.name or '-attempt' in p.name:continue
        target=O/'results'/p.name
        if not target.exists():target.symlink_to(p)
        reused[str(p)]=Q.sha(p)
    for p in (PARENT/'ssd').glob('*-frozen.json'):
        target=O/'ssd'/p.name;shutil.copyfile(str(p),str(target));reused[str(p)]=Q.sha(p)
    Q.save(O/'reused-tuning.json',dict(files=reused,note='only deterministic work and training-frozen configurations reused; report timing cohort is entirely new'))
    import ssd_priority_review
    material=Q.load(PARENT/'analysis/openimage-change-gate.json')['material']
    ssd_priority_review.run(Q,Q.E/'metric-consistent-20261111/scratch/dual',PARENT/'graphs/openimage',material)
    status('complete' if not Q.load(O/'timings/pending.json') else 'complete_with_pending',cpus=M.CPUS)
if __name__=='__main__':
    signal.signal(signal.SIGTERM,Q.stop);signal.signal(signal.SIGINT,Q.stop)
    try:main()
    except Exception as e:Q.stop();status('failed',error=str(e));raise
