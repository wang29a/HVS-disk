"""One extra diagnostic sample; do not replace the three pure throughput runs."""
import fcntl
import json
import shutil

import run_suite as suite
import analyse


def main():
    out = suite.OUT
    lock = (suite.E / 'budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX)
    try:
        state = json.loads((out / 'status.json').read_text())
        assert state['stage'] == 'PERF_AUDIT_COMPLETE', state
        target = out / 'extra-measurements/exact-pure-profile'
        target.mkdir(parents=True, exist_ok=False)
        root = suite.fixture('exact-pure-profile-probe')
        suite.run([out / 'driver-profile', root, 'exact', 4, 0, 'e1',
                   1000, target, 0], 'insert1000', target)
        result = dict(
            reason='three pure exact runs differ consistently from mixed runs; '
                   'inspect locks and phases without changing the throughput reference',
            repetitions=1, query_threads=0, update_threads=4,
            result=json.loads((target / 'result.json').read_text()),
            profile=analyse.profile(target),
            graph_sha256=suite.sha(root / 'graph.pages'))
        suite.save(target / 'analysis.json', result)
        assert json.loads((root / 'owner.json').read_text())['owner'] == str(suite.ROOT)
        shutil.rmtree(str(root))
        suite.status('PERF_AUDIT_COMPLETE', pure_exact_probe_complete=True)
    except Exception as error:
        suite.status('STOPPED_AT_PURE_EXACT_PROBE', reason=str(error))
        raise
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


if __name__ == '__main__':
    main()
