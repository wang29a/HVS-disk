"""Amendment: resume only P2/P3 after fixing native Vamana metadata selection."""
import fcntl,signal,hashlib
import common as c
import openimages,native_compare

def main():
    own=(c.RUN/'resume23.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=c.load(c.RUN/'followups-plan.json');frozen=c.load(c.ROOT/'FREEZE.json');raw=(c.ROOT/'README.md').read_bytes()
    if hashlib.sha256(raw[:frozen['bytes']]).hexdigest()!=frozen['sha256']:raise RuntimeError('original protocol overwritten')
    proof=c.load(c.RUN/'recovery-meta-check.json')
    if proof['status']!='PASS':raise RuntimeError('metadata regression check not passed')
    plan=dict(parent_plan_sha256=c.sha(c.RUN/'followups-plan.json'),reason='M0 source-only Vamana metadata is compact.meta, not full.meta; no graph/data/GT/parameters changed',source={p.name:c.sha(p) for p in (c.ROOT/'scripts').iterdir() if p.suffix in ('.py','.cpp','.h')},metadata_gate_sha256=c.sha(c.RUN/'recovery-meta-check.json'))
    path=c.RUN/'resume23-plan.json'
    if path.exists() and c.load(path)!=plan:raise RuntimeError('recovery source plan changed')
    c.save(path,plan);lock=c.lock();results={}
    for number,task in ((2,openimages.main),(3,native_compare.main)):
        c.save(c.RUN/'resume23-status.json',dict(stage='running',priority=number,pid=c.os.getpid()))
        if number==3 and not (c.RUN/'openimages/manifest.txt').exists():r=dict(complete=False,status='gated',reason='missing P2 source inputs')
        else:
            try:r=task()
            except Exception as e:c.stop();r=dict(complete=False,error=str(e))
        p=c.RUN/('priority'+str(number)+'.json');previous=c.RUN/('priority'+str(number)+'-before-meta-repair.json')
        if p.exists() and not previous.exists():previous.write_bytes(p.read_bytes())
        c.save(p,r);results[str(number)]=r
    c.save(c.RUN/'resume23-status.json',dict(stage='complete' if all(r.get('complete') for r in results.values()) else 'complete_with_pending',results=results))
    c.status('edbt_resume23_finished',priorities={str(n):c.load(c.RUN/('priority'+str(n)+'.json')) for n in (1,2,3,4)})
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.save(c.RUN/'resume23-status.json',dict(stage='failed',error=str(e)));raise
