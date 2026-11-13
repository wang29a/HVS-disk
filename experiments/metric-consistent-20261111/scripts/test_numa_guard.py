"""Regression for process teardown and strict measurement-window checks."""
import unittest
from numa_guard import parse_maps, validate


class GuardTest(unittest.TestCase):
    def entry(self, t, mask='72-75', policy='bind:3'):
        result = parse_maps('1000 {} anon=100 N3=100\n'.format(policy))
        result.update(time=t, cpu_mask=mask, cpuset_mems='0-3')
        return result

    def test_zombie_empty_maps_do_not_erase_live_evidence(self):
        self.assertIsNone(parse_maps(''))
        points = [self.entry(t) for t in (2, 3, 4)]+[None]
        self.assertTrue(validate(points, 1, 5)['verified'])

    def test_observations_outside_window_cannot_validate_measurement(self):
        points = [self.entry(t) for t in (-3, -2, -1)]+[None]
        self.assertFalse(validate(points, 1, 5)['verified'])

    def test_wrong_affinity_is_rejected(self):
        points = [self.entry(t, mask='0-191') for t in (2, 3, 4)]
        self.assertFalse(validate(points, 1, 5)['verified'])

    def test_wrong_memory_policy_is_rejected(self):
        points = [self.entry(t, policy='bind:2') for t in (2, 3, 4)]
        self.assertFalse(validate(points, 1, 5)['verified'])

    def test_valid_cpuset_can_allow_other_memory_nodes(self):
        points = [self.entry(t) for t in (2, 3, 4)]
        self.assertTrue(validate(points, 1, 5)['verified'])


if __name__ == '__main__':
    unittest.main()
