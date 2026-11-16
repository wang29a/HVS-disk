"""Priority two follows completion of priority one, with no need for polling."""
import fcntl,signal,time
import common as c
import no_sidecar
def main():
    for name in ('logs','guards','commands'):(c.RUN/name).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'second.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    deadline=time.monotonic()+2*3600
    while not (c.RUN/'phase1.json').exists():
        c.status('waiting_phase1')
        if time.monotonic()>deadline:raise RuntimeError('phase1 has not completed within bounded wait')
        time.sleep(5)
    lock=c.lock();result=no_sidecar.main();c.save(c.RUN/'phase2.json',result)
    c.status('phase2_complete' if result['complete'] else 'phase2_with_pending',result=result)
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as error:c.stop();c.status('stopped',error=str(error));raise
