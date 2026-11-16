"""Compile and validate, then run the first EDBT priority under one tmux lock."""
import fcntl,signal
import common as c
import compile_relaxed,density

def main():
    for n in ('logs','commands','guards'):(c.RUN/n).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'coordinator.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    lock=c.lock();c.capacity(8*2**30,'EDBT isolated compile and first relaxed graph')
    compile_relaxed.main();c.save(c.RUN/'compile.json',dict(complete=True))
    result=density.main();c.save(c.RUN/'priority1.json',result);c.status('priority1_complete' if result['complete'] else 'priority1_pending',result=result)
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as e:c.stop();c.status('failed',error=str(e));raise
