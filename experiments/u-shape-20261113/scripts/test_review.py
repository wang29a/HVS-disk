import pathlib,sys,unittest
sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent))
import ssd_priority_review as S
import review_monitor as M
class SelectionAndEvidence(unittest.TestCase):
    def test_work_selection_does_not_pick_lowest_latency(self):
        rows=[dict(L=40,rerank_limit=40,recall=95,submitted_pages=10,pq_space_evals=100,latency_us=99),dict(L=80,rerank_limit=20,recall=95,submitted_pages=20,pq_space_evals=50,latency_us=1)]
        x=S.choose(rows);self.assertEqual(x['pages'][0]['L'],40);self.assertEqual(x['compute'][0]['L'],80)
    def test_no_recall_extrapolation(self):
        self.assertIsNone(S.bracket([dict(recall=90),dict(recall=94)]))
    def test_bracket_preserves_frozen_order(self):
        a,b,w=S.bracket([dict(recall=93),dict(recall=97)]);self.assertEqual((a['recall'],b['recall']), (93,97));self.assertEqual(w,.5)
    def observations(self,mask='84-87'):
        M.CPUS=[84,85,86,87]
        return [dict(time=i,cpu=400*i,own=400*i,numa=dict(time=i,cpu_mask=mask,verified=True,conflicting_policies=[]),disks={d:dict(write_bytes=0,read_bytes=0) for d in ('nvme2n1','nvme3n1')}) for i in range(11)]
    def test_actual_selected_affinity_accepted(self):
        self.assertTrue(M.validate(self.observations(),dict(point_start_unix=0,measured_seconds=10))['accepted'])
    def test_wrong_affinity_is_not_normalized(self):
        self.assertFalse(M.validate(self.observations('72-75'),dict(point_start_unix=0,measured_seconds=10))['accepted'])
    def test_missing_live_maps_rejected(self):
        x=self.observations()
        for p in x:p['numa']=None
        self.assertFalse(M.validate(x,dict(point_start_unix=0,measured_seconds=10))['accepted'])
    def test_external_cpu_rejected(self):
        x=self.observations()
        for p in x:p['cpu']+=360*p['time']
        self.assertFalse(M.validate(x,dict(point_start_unix=0,measured_seconds=10))['accepted'])
if __name__=='__main__':unittest.main()
