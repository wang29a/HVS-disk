"""After priority 1, run 2/3/4 serially; preserve independent work on bounded failure."""
import fcntl,signal,time
import common as c
import openimages,native_compare,no_sidecar_repeats

def main():
    for n in ('logs','commands','guards'):(c.RUN/n).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'followups.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import hashlib
    frozen=c.load(c.ROOT/'FREEZE.json');text=(c.ROOT/'README.md').read_bytes()
    if hashlib.sha256(text[:frozen['bytes']]).hexdigest()!=frozen['sha256']:raise RuntimeError('frozen protocol prefix overwritten')
    plan=dict(protocol_prefix=frozen,readme_sha256=c.sha(c.ROOT/'README.md'),sources={p.name:c.sha(p) for p in (c.ROOT/'scripts').iterdir() if p.suffix in ('.py','.h','.cpp')})
    stamp=c.RUN/'followups-plan.json'
    if stamp.exists() and c.load(stamp)!=plan:raise RuntimeError('followup sources changed after freeze; amendment required')
    c.save(stamp,plan)
    deadline=time.monotonic()+24*3600
    while not (c.RUN/'priority1.json').exists():
        state=c.load(c.RUN/'status.json') if (c.RUN/'status.json').exists() else {}
        if state.get('stage')=='failed':
            c.save(c.RUN/'priority1.json',dict(complete=False,error=state.get('error'),dependent='relaxed-graph branch stopped; existing corrected builders independent'));break
        c.save(c.RUN/'followups-status.json',dict(stage='waiting_priority1',pid=c.os.getpid()))
        if time.monotonic()>deadline:raise RuntimeError('priority1 wait exceeded24h')
        time.sleep(10)
    lock=c.lock();results={}
    for number,task in [(2,openimages.main),(3,native_compare.main),(4,no_sidecar_repeats.main)]:
        c.save(c.RUN/'followups-status.json',dict(stage='running',priority=number,pid=c.os.getpid()))
        if number==3 and not (c.RUN/'openimages/manifest.txt').exists():
            result=dict(complete=False,status='gated',reason='OpenImages source preparation/manifest from priority2 unavailable')
        else:
            try:result=task()
            except Exception as e:c.stop();result=dict(complete=False,status='failed_or_gated',error=str(e))
        c.save(c.RUN/('priority'+str(number)+'.json'),result);results[str(number)]=result
    c.save(c.RUN/'priority5.json',dict(status='planned_not_launched',tasks=['third dataset (CC3M candidate; current corrected graph/GT availability audit required)','alpha endpoints require matching GT inventory','weighted alpha=.3 existing-data audit already completed'],prior_weighted_audit=str(c.E/'paper-priorities-20261114/runtime/weighted-anomaly/summary.json')))
    c.save(c.RUN/'followups-status.json',dict(stage='complete' if all(r.get('complete') for r in results.values()) else 'complete_with_pending',results=results))
    c.status('edbt_queue_finished',priorities=results,priority1=c.load(c.RUN/'priority1.json'),priority5='planned, not silently launched')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.save(c.RUN/'followups-status.json',dict(stage='failed',error=str(e)));raise
