import unittest
from quality_metrics import ceilings, prediction_scores, matched_work


class RecallRules(unittest.TestCase):
    def test_boundary_alternative_avoids_id_penalty(self):
        s=prediction_scores(list(range(10)),list(range(9))+[10],range(9),[9,10])
        self.assertEqual(s['recall_id'],.9)
        self.assertEqual(s['rank_tie_recall'],1.)

    def test_boundary_surplus_does_not_replace_strict_neighbors(self):
        s=prediction_scores(list(range(10)),range(9,19),range(9),range(9,19))
        self.assertEqual(s['distance_threshold_acceptance'],1.)
        self.assertEqual(s['rank_tie_recall'],.1)

    def test_unreachable_boundary_can_have_reachable_substitute(self):
        c=ceilings(list(range(10)),{9},9,0,2,1)
        self.assertEqual(c['id_ceiling'],.9)
        self.assertEqual(c['rank_tie_ceiling'],1.)

    def test_unreachable_strict_point_is_not_cured_by_boundary_ties(self):
        c=ceilings(list(range(10)),{0},9,1,20,0)
        self.assertEqual(c['distance_threshold_ceiling'],1.)
        self.assertEqual(c['rank_tie_ceiling'],.9)

    def test_duplicate_predictions_rejected(self):
        with self.assertRaises(ValueError):
            prediction_scores(range(10),[0]*10,range(9),[9])

    def test_no_extrapolation(self):
        points=[dict(L=10,recall_id=.8,expanded=20),dict(L=20,recall_id=.9,expanded=30)]
        self.assertFalse(matched_work(points)['covered'])

    def test_interpolation_records_real_bracket(self):
        m=matched_work([dict(L=10,recall_id=.9,expanded=20),dict(L=20,recall_id=1.,expanded=40)])
        self.assertTrue(m['covered'])
        self.assertAlmostEqual(m['expanded'],30)


if __name__=='__main__':unittest.main()
