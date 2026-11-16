"""Run priority one immediately; no data generation or new performance tuning."""
import fcntl,signal
import common as c
import openimage_pairs
def main():
    for name in ('logs','guards','commands'):(c.RUN/name).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'first.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=c.lock();result=openimage_pairs.main();c.save(c.RUN/'phase1.json',result)
    c.status('phase1_complete' if result['complete'] else 'phase1_with_pending',result=result)
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as error:c.stop();c.status('stopped',error=str(error));raise
