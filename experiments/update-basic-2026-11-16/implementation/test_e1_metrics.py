import unittest
from e1_metrics import id_ceiling,trend


class Checkpoints(unittest.TestCase):
    def test_ceiling_uses_corresponding_GT_and_fixed_cohort(self):
        rows=[list(range(10)),list(range(10,20))]
        self.assertEqual(id_ceiling(rows,{0,10,11},[1])['theoretical_ceiling_loss_pp'],20)
        self.assertEqual(id_ceiling(rows,{0,10,11},[0])['theoretical_ceiling_loss_pp'],10)

    def test_missing_middle_checkpoint_cannot_claim_four_batch_trend(self):
        result=trend([dict(inserted=0,expanded=100),dict(inserted=200000,expanded=130)],100)
        self.assertFalse(result['strictly_increasing_all_checkpoints'])

    def test_rising_then_recovery_is_not_persistent_rise(self):
        result=trend([dict(inserted=i*50000,expanded=v) for i,v in enumerate([100,102,108,105,107])],100)
        self.assertFalse(result['strictly_increasing_all_checkpoints'])
        self.assertAlmostEqual(result['net_change_pp'],7)

    def test_report_persistent_rise(self):
        result=trend([dict(inserted=i*50000,expanded=v) for i,v in enumerate([100,102,108,109,111])],100)
        self.assertTrue(result['strictly_increasing_all_checkpoints'])
        self.assertEqual(result['inserted'],[0,50000,100000,150000,200000])


if __name__=='__main__':unittest.main()
