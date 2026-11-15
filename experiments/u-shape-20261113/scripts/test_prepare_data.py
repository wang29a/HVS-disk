"""Independent small-data checks for subset IDs, exact metric and resumability."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import prepare_data as p


class Preparation(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.out = self.root / 'out'
        rng = np.random.RandomState(42)
        self.arrays = [rng.normal(size=(91, 8)), rng.normal(size=(91, 12)),
                       rng.normal(size=(6, 8)), rng.normal(size=(6, 12))]
        for name, data in zip(p.NAMES, self.arrays):
            p.write_matrix(self.source / name, data, '<f4')
        self.split = self.root / 'split.json'
        p.save(self.split, dict(cohorts=dict(tune=dict(query_ids=[3, 0, 5]), report=dict(query_ids=[2, 1, 4]))))

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, count=37):
        identity = p.plan(self.source, self.out, self.split, count, 20261113)
        p.extract(self.source, self.out, identity, 13)
        return identity

    def test_deterministic_joint_ids_and_unchanged_query_order(self):
        identity = self.prepare()
        ids = p.matrix(self.out / 'source_ids.bin', '<u4')[:, 0]
        self.assertEqual(len(ids), 37)
        self.assertTrue(np.all(ids[1:] > ids[:-1]))
        for name in p.NAMES[:2]:
            np.testing.assert_array_equal(p.matrix(self.out / name), p.matrix(self.source / name)[ids])
        for name in p.NAMES[2:]:
            np.testing.assert_array_equal(p.matrix(self.out / name), p.matrix(self.source / name))
        self.assertEqual(identity['query_cohorts']['tune'], [3, 0, 5])
        p.plan(self.source, self.out, self.split, 37, 20261113)

    def test_exhaustive_gt_matches_direct_full_scan_and_resume(self):
        identity = self.prepare()
        p.ground_truth(self.out, identity, 2, 13)
        x, y, qx, qy = (np.asarray(p.matrix(self.out / n), dtype=np.float64) for n in p.NAMES)
        local_ids = np.arange(len(x))
        hashes = []
        for alpha in p.WEIGHTS:
            effective = float(np.float32(alpha))
            expected = []
            for a, b in zip(qx, qy):
                scores = effective * np.linalg.norm(x - a, axis=1) + (1 - effective) * np.linalg.norm(y - b, axis=1)
                expected.append(np.lexsort((local_ids, scores))[:10])
            path = self.out / ('gt-alpha-{}.bin'.format(alpha))
            np.testing.assert_array_equal(p.matrix(path, '<u4'), expected)
            np.testing.assert_array_equal(p.matrix(self.out / ('query-alpha-{}.bin'.format(alpha))), np.full((6, 1), alpha, dtype='<f4'))
            hashes.append(p.digest(path))
        p.ground_truth(self.out, identity, 2, 13)
        self.assertEqual(hashes, [p.digest(self.out / ('gt-alpha-{}.bin'.format(a))) for a in p.WEIGHTS])

    def test_duplicate_vectors_tie_by_lowest_local_id_across_blocks(self):
        x = np.zeros((31, 4), dtype=np.float64)
        q = np.zeros((2, 4), dtype=np.float64)
        ids, distances = p.exact_batch(x, x, q, q, p.WEIGHTS, 10, 7)
        np.testing.assert_array_equal(ids, np.broadcast_to(np.arange(10), (3, 2, 10)))
        np.testing.assert_array_equal(distances, 0)

    def test_two_fusion_objectives_can_reverse_the_ranking(self):
        x = np.array([[1., 0], [2.2, 0]])
        y = np.array([[3., 0], [2.2, 0]])
        q = np.array([[0., 0]])
        ids, _ = p.exact_batch(x, y, q, q, [.5], 2, 1)
        self.assertEqual(ids[0, 0].tolist(), [0, 1])
        squared = .5 * np.sum(x * x, axis=1) + .5 * np.sum(y * y, axis=1)
        self.assertEqual(np.argsort(squared).tolist(), [1, 0])

    def test_overlapping_query_cohorts_rejected(self):
        p.save(self.split, dict(cohorts=dict(tune=dict(query_ids=[0, 1, 2]), report=dict(query_ids=[2, 3, 4]))))
        with self.assertRaisesRegex(RuntimeError, 'disjoint'):
            p.plan(self.source, self.out, self.split, 37, 20261113)

    def test_misaligned_base_counts_rejected(self):
        p.write_matrix(self.source / p.NAMES[1], self.arrays[1][:-1], '<f4')
        with self.assertRaisesRegex(RuntimeError, 'row counts'):
            p.plan(self.source, self.out, self.split, 37, 20261113)

    def test_changed_seed_is_not_silently_resumed(self):
        self.prepare()
        with self.assertRaisesRegex(RuntimeError, 'namespace'):
            p.plan(self.source, self.out, self.split, 37, 20261114)

    def test_changed_extracted_content_rejected(self):
        identity = self.prepare()
        path = self.out / p.NAMES[0]
        with path.open('r+b') as stream:
            stream.seek(8)
            stream.write(b'bad!')
        with self.assertRaisesRegex(RuntimeError, 'changed'):
            p.extract(self.source, self.out, identity, 13)

    def test_changed_gt_grid_is_not_silently_resumed(self):
        identity = self.prepare()
        p.ground_truth(self.out, identity, 2, 13)
        with self.assertRaisesRegex(RuntimeError, 'configuration changed'):
            p.ground_truth(self.out, identity, 3, 13)

    def test_nonfinite_data_rejected(self):
        values = np.full((91, 8), np.nan, dtype='<f4')
        p.write_matrix(self.source / p.NAMES[0], values, '<f4')
        identity = p.plan(self.source, self.out, self.split, 37, 20261113)
        with self.assertRaisesRegex(RuntimeError, 'nonfinite'):
            p.extract(self.source, self.out, identity, 13)


if __name__ == '__main__':
    unittest.main()
