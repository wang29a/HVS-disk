import unittest
import hashlib
import json
from pathlib import Path
from layer_analysis import overlap,summary_overlap
T=dict(mean_candidate_coverage_min=.99,query_candidate_coverage_min=.95)
class Layers(unittest.TestCase):
    def test_original_frozen_protocol_and_amendment_parent_preserved(self):
        root=Path(__file__).resolve().parent.parent
        text=(root/'README.md').read_bytes()
        for file,size_key,hash_key in [('FREEZE.json','bytes','sha256'),('AMENDMENT-20261114.json','parent_readme_bytes','parent_readme_sha256')]:
            record=json.loads((root/file).read_text())
            self.assertEqual(hashlib.sha256(text[:record[size_key]]).hexdigest(),record[hash_key])
    def test_raw_recall(self):
        r=overlap([[1,1],[2,2],[3,3]],[[1,1],[2,2],[4,4]])
        self.assertAlmostEqual(r['candidate_coverage'],2/3)
    def test_legitimate_boundary_tie(self):
        r=overlap([[1,1],[2,2],[3,2]],[[1,1],[7,2],[8,2]])
        self.assertEqual(r['candidate_coverage'],1);self.assertAlmostEqual(r['raw_ID_overlap'],1/3)
    def test_ties_do_not_hide_strict_miss(self):
        r=overlap([[1,1],[2,2],[3,2]],[[7,2],[8,2],[9,2]])
        self.assertAlmostEqual(r['candidate_coverage'],2/3);self.assertEqual(r['strict_missing_B_IDs'],[1])
    def test_one_bad_query_not_averaged_away(self):
        self.assertFalse(summary_overlap([dict(candidate_coverage=1,raw_ID_overlap=1)]*99+[dict(candidate_coverage=.94,raw_ID_overlap=.94)],T)['passed'])
    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(RuntimeError):overlap([[1,1],[2,2]],[[1,1],[1,1]])
if __name__=='__main__':unittest.main()
