#!/usr/bin/env python3
"""Queued low-L follow-up, selected only because training recall exceeds 95%."""
import fcntl,json,pathlib,time,signal
import stage2_queue as Q
def main():
    Q.status=lambda stage,**kw: Q.save(Q.O/'followup-status.json',dict(stage=stage,pid=__import__('os').getpid(),updated_unix=time.time(),**kw))
    Q.status('waiting_global_lock')
    lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+')
    deadline=time.monotonic()+6*3600
    while True:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB); break
        except BlockingIOError:
            if time.monotonic()>deadline: Q.status('deferred_lock_timeout'); return
            time.sleep(15)
    # Interleave U0/S/U1 after every specialized graph has been built.
    # Earlier per-stage timings remain diagnostics, never substitute for this
    # joint schedule when comparing primary methods.
    joint={}
    for a in ('0.1','0.5','0.9'):
        for label in ('unified-'+a+'-off','fixed-'+a,'unified-'+a+'-on'):
            p=Q.O/'analysis'/(label+'-matched.json'); f=Q.O/'frozen'/(label+'.json')
            if p.exists() and f.exists() and Q.load(p).get('covered'):
                name='joint-'+label; Q.save(Q.O/'analysis'/(name+'-matched.json'),Q.load(p))
                joint[name]=Q.load(f)['identity']['spec']
    if all('joint-fixed-'+a in joint for a in ('0.1','0.5','0.9')):
        Q.save(Q.O/'analysis/joint-schedule.json',dict(order=list(joint),repeats=3,minimum_seconds=10,scope='U0/S/U1 interleaved after all graph construction; original frozen configurations'))
        Q.timings(joint)
    else:
        Q.save(Q.O/'analysis/joint-schedule.json',dict(stage='not_ready',reason='specialized comparison gated or incomplete'))
    frozen=Q.load(Q.O/'frozen/openimage-single.json')
    if frozen['report_L']: Q.status('not_needed_original_bracket_covered'); return
    # Only training results triggered this amendment; original freeze remains.
    coarse=Q.load(Q.O/'results/openimage-single-tune-coarse.json')
    if min(p['recall'] for p in coarse)<=.95:
        Q.status('not_a_low_L_gap',reason='no automatic extension beyond upper L cap'); return
    s=frozen['identity']['spec']; Q.GRID=[10,12,15,20,40]
    result=Q.study(s,'openimage-single-low-L')
    if result['covered']:
        Q.timings({'openimage-single-low-L':s}); Q.status('complete',matched=result)
    else:
        Q.status('95_not_bracketed_at_top10_floor',reason='retain observed >=95% operating point; do not extrapolate below L=10')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,Q.stop); signal.signal(signal.SIGINT,Q.stop)
    try: main()
    except Exception as exc: Q.stop(); Q.status('failed',error=str(exc)); raise
