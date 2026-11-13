"""Exercise point failure isolation, resume and owned-process termination."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import phase_two as queue
from queue_guard import DeferredMeasurement, stop_owned_group


class Coordinator:
    @staticmethod
    def save(path, data):
        Path(path).write_text(json.dumps(data))

    @staticmethod
    def status(*args, **kwargs):
        pass


class QueueGuardTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.root = Path(self.directory.name)
        for name in ('analysis', 'results'):
            (self.root/name).mkdir()
        self.saved_root, self.saved_coordinator = queue.ROOT, queue.B
        queue.ROOT, queue.B = self.root, Coordinator()

    def tearDown(self):
        queue.ROOT, queue.B = self.saved_root, self.saved_coordinator
        self.directory.cleanup()

    def test_failed_point_does_not_block_other_points_and_recovery_is_bounded(self):
        calls = []
        failed = 'report-time-full-T_XY-exact-DRAM-f0-L20-r20-repeat1'
        def search(*args, **kwargs):
            calls.append(args[-1])
            if args[-1] == failed:
                raise DeferredMeasurement('external activity')
            return {'recall': 95}
        manifest = dict(arm='full', layout='T_XY', exact=True, tuning_frontier=[],
                        selected=[dict(filtering=False, L=20, r=20), dict(filtering=True, L=20, r=20)])
        with patch.object(queue, 'query', side_effect=search):
            queue.measure(manifest, ['DRAM'])
        self.assertEqual(len(calls), 7)
        self.assertEqual(calls.count(failed), 2)
        self.assertEqual(len(set(calls)), 6)
        pending = queue.pending_points()
        self.assertEqual(list(pending), [failed])
        self.assertEqual(pending[failed]['deferred_passes'], 2)

    def test_invalid_numa_or_data_is_not_silently_deferred(self):
        args = ('full', 'T_XY', 'DRAM', True, True, 20, 20, 'report', 'bad-input')
        with patch.object(queue, 'query', side_effect=RuntimeError('bad NUMA')):
            with self.assertRaises(RuntimeError):
                queue.timed_or_defer(args)
        self.assertFalse(queue.pending_points())

    def test_accepted_point_resumes_without_opening_index_and_clears_stale_pending(self):
        (self.root/'results/done.json').write_text('[{"recall":95}]')
        (self.root/'analysis/done-accepted.json').write_text('{}')
        (self.root/'analysis/pending-measurements.json').write_text('{"done":{}}')
        result = queue.query('full', 'T_XY', 'DRAM', True, True, 20, 20, 'report', 'done', timed=True)
        self.assertEqual(result['recall'], 95)
        self.assertFalse(queue.pending_points())

    def test_timeout_cleanup_reaps_only_owned_process_group(self):
        child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)
        try:
            stop_owned_group(child, grace=.2)
            self.assertIsNotNone(child.returncode)
        finally:
            if child.poll() is None:
                child.kill()
                child.wait()

    def test_untimed_diagnostics_do_not_wait_for_quiet_timing_window(self):
        files = {name:{'path':'unused-'+name} for name in ('query1', 'query2', 'alpha', 'gt')}
        def run(args, label, cpus, env):
            self.assertEqual(cpus, [72, 73, 74, 75])
            self.assertEqual(env['LAYOUT_ONE_PASS'], '1')
            Path(args[7]).write_text('[{"recall":95}]')
        with patch.object(queue, 'SPLIT', {'cohorts':{'tune':{'files':files}}}), \
             patch.object(queue, 'layout', return_value=('meta', 'topo', 'layout')), \
             patch.object(queue.B, 'run', side_effect=run, create=True):
            # Coordinator deliberately has no quiet() implementation.
            point = queue.query('full', 'T_XY', 'DRAM', False, True, 20, 20, 'tune', 'diagnostic')
        self.assertEqual(point['recall'], 95)


if __name__ == '__main__':
    unittest.main()
