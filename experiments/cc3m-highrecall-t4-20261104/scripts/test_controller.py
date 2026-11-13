#!/usr/bin/env python3
"""Synthetic-only checks: predecessor gates and covered/uncovered recall exports."""
import importlib.util
import json
from pathlib import Path
import tempfile

HERE = Path(__file__).resolve().parent


def load(name):
    spec = importlib.util.spec_from_file_location(name, str(HERE/(name+'.py')))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    queue = load('run_queue')
    summary = load('summarize')
    assert queue.bracket([dict(L=400, recall=94), dict(L=600, recall=96)]) == [400, 600]
    assert queue.bracket([dict(L=400, recall=94), dict(L=600, recall=95)]) == [600]
    assert queue.bracket([dict(L=400, recall=94), dict(L=1600, recall=94.9)]) == []

    class StillWaiting(Exception):
        pass

    with tempfile.TemporaryDirectory(prefix='hvs-cc3m-gate-test-') as tmp:
        root = Path(tmp)
        queue.ROOT = root
        queue.PREDECESSORS = [root/'p1', root/'p2']
        queue.TIMING = queue.PREDECESSORS[1]
        for i, p in enumerate(queue.PREDECESSORS):
            p.mkdir()
            (p/'queue.lock').touch()
            (p/'status.json').write_text(json.dumps(dict(status='complete' if i == 0 else 'running')))

        def stop_wait(_):
            raise StillWaiting()

        original_sleep = queue.time.sleep
        queue.time.sleep = stop_wait
        try:
            try:
                queue.wait_predecessors()
                raise AssertionError('Started before both predecessors completed')
            except StillWaiting:
                pass
            assert json.loads((root/'status.json').read_text())['status'] == 'waiting_predecessors'
            (queue.TIMING/'status.json').write_text(json.dumps(dict(status='paused_by_user')))
            held = (queue.PREDECESSORS[0]/'queue.lock').open('r')
            queue.fcntl.flock(held, queue.fcntl.LOCK_EX | queue.fcntl.LOCK_NB)
            try:
                try:
                    queue.wait_predecessors()
                    raise AssertionError('Started while predecessor lock was held')
                except StillWaiting:
                    pass
            finally:
                held.close()
            locks = queue.wait_predecessors()
            assert len(locks) == 2
            for lock in locks:
                lock.close()
        finally:
            queue.time.sleep = original_sleep

    with tempfile.TemporaryDirectory(prefix='hvs-cc3m-export-test-') as tmp:
        root = Path(tmp)
        for name in ['results', 'analysis']:
            (root/name).mkdir()
        summary.ROOT = root
        for method in summary.METHODS:
            for repeat in [1, 2, 3]:
                points = []
                levels = [400, 500, 600] if method == 'full' else [400, 600]
                for l in levels:
                    p = {k: 10.0+l/100+repeat for k in summary.METRICS}
                    recall = 94+(l-400)/100 if method != 'merge' else 94+(l-400)/1000
                    p.update(L=l, recall=recall, seconds=10.2,
                        reader_total_us=180, reader_coupled_us=180 if method == 'baseline' else 0,
                        reader_coupled_backend_us=150 if method == 'baseline' else 0,
                        reader_topology_us=0 if method == 'baseline' else 80,
                        reader_coordinate_us=0 if method == 'baseline' else 100,
                        external_service_us=250, coupled_backend_requests=2 if method == 'baseline' else 0,
                        topology_backend_requests=0 if method == 'baseline' else 1,
                        coordinate_backend_requests=0 if method == 'baseline' else 1,
                        coupled_backend_bytes=16384 if method == 'baseline' else 0,
                        topology_backend_bytes=0 if method == 'baseline' else 8192,
                        coordinate_backend_bytes=0 if method == 'baseline' else 8192)
                    points.append(p)
                on = dict(method=method, threads=4, one_pass=False, observer_enabled=True, points=points)
                off = dict(on, observer_enabled=False, points=[dict(points[-1])])
                for tag, run in [('on', on), ('off', off)]:
                    (root/'results'/'{}__r{}__{}.json'.format(method, repeat, tag)).write_text(json.dumps(run))
            (root/'analysis'/(method+'-parity.json')).write_text(json.dumps(dict(status='PASS')))
        summary.main()
        audit = json.loads((root/'analysis/completeness.json').read_text())
        assert audit['timed_runs'] == 24 and audit['target95_uncovered_methods'] == ['merge']
        assert audit['observer_control_warnings'] == 0
        data = json.loads((root/'data/measurements.json').read_text())
        assert len(data['means']) == 3 and len(data['matched_per_repeat']) == 9
        for row in data['means']:
            assert row['L'] == 500 and row['repeats'] == 3
            assert row['reader_total_us'] == 180 and row['submitted_pages'] == 2
            assert row['reader_io_share_pct'] == 72
            if row['method'] == 'baseline':
                assert row['reader_total_us'] == row['reader_coupled_us']
                assert row['reader_total_us'] != row['reader_coupled_us']+row['reader_coupled_backend_us']
        assert all(r['L_low'] == r['L_high'] == 500 for r in data['matched_per_repeat'] if r['method'] == 'full')
        assert not any(r['method'] == 'merge' for r in data['matched_per_repeat'])
    print('SYNTHETIC_PASS predecessor status+locks, exact/bracket/uncovered recall, 24-run export, no child-time double-counting')


if __name__ == '__main__':
    main()
