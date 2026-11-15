#!/usr/bin/env python3
"""Serial follow-up for the failed full-graph closeness check."""
import fcntl,os,pathlib,signal,time
import stage2_queue as Q
R=Q.R;V2=R/'stage2-v2';Q.O=R/'review-20261113';O=Q.O
def status(stage,**kw):Q.save(O/'explanation-status.json',dict(stage=stage,pid=os.getpid(),updated_unix=time.time(),**kw))
Q.status=status
def main():
    status('waiting_global_lock');lock=(Q.E/'budget-reorder-20261103/queue.lock').open('a+');deadline=time.monotonic()+6*3600
    while True:
        try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
        except BlockingIOError:
            if time.monotonic()>deadline:status('deferred_lock_timeout');return
            time.sleep(10)
    out=O/'analysis/centre-candidate-explanation.json'
    if not out.exists():Q.execute([O/'bin/diagnose-centre',V2/'graphs/unified/float-graph.bin',V2/'graphs/fixed-0.5/float-graph.bin',Q.D/'base_img_emb.fvecs',Q.D/'base_text_emb.fvecs',out],'diagnose-centre-candidates',{},False,1200,'72-75')
    x=Q.load(out)
    status('complete' if not x['positive_margin_predicate_mismatches'] else 'needs_semantic_review',result=str(out),strength_sweep='not started; centre gate failed')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,Q.stop);signal.signal(signal.SIGINT,Q.stop)
    try:main()
    except Exception as exc:Q.stop();status('failed',error=str(exc));raise
