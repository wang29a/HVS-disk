"""Run centre degree study then existing-data weighted analysis after priorities 1/2."""
import fcntl,signal,time
import common as c
import degree_study,analyze_weighted
def main():
    for name in ('logs','guards','commands'):(c.RUN/name).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'remaining.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+4*3600
    while not (c.RUN/'phase2.json').exists():
        c.save(c.RUN/'remaining-status.json',dict(stage='waiting_phase2',pid=c.os.getpid(),updated_unix=time.time()))
        if time.monotonic()>deadline:raise RuntimeError('priority2 did not finish within bounded wait')
        time.sleep(10)
    lock=c.lock();c.save(c.RUN/'remaining-status.json',dict(stage='degree_study_running',pid=c.os.getpid()))
    try:result=degree_study.main();c.save(c.RUN/'phase3.json',result)
    except Exception as error:
        c.stop();result=dict(complete=False,error=str(error));c.save(c.RUN/'phase3.json',result)
    fourth=analyze_weighted.main();c.save(c.RUN/'phase4.json',fourth)
    c.status('four_priorities_complete' if result.get('complete') else 'completed_with_degree_issue',phase3=result,phase4=fourth,next='stop; no OpenImages multigraph run authorised')
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as error:c.stop();c.status('stopped',error=str(error));raise
