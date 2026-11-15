"""Tests of scientific boundaries, not timing or graph quality claims."""
import importlib.util, pathlib, sys, unittest
METRIC=pathlib.Path(__file__).resolve().parents[2]/'metric-consistent-20261111'
sys.path.insert(0,str(METRIC/'scripts'))
def module(name):
    spec=importlib.util.spec_from_file_location(name,pathlib.Path(__file__).parent/(name+'.py'))
    obj=importlib.util.module_from_spec(spec); spec.loader.exec_module(obj); return obj
Q=module('stage2_queue'); S=module('prepare_specialized')
def point(L,recall,expanded):
    return dict(L=L,recall=recall,expanded=expanded,exact_space_evals=expanded*20,inspected_edges=expanded*40,alpha_rejected=expanded*10,mean_degree=40,reachable=3,nodes=3,queries=[dict(query_id=q,expanded=expanded+q-1) for q in range(3)])
class Boundaries(unittest.TestCase):
    def test_no_extrapolation(self):
        self.assertFalse(Q.matched([point(20,.7,30),point(40,.9,50)])['covered'])
    def test_match_records_both_endpoints(self):
        m=Q.matched([point(40,.9,50),point(80,1.,100)])
        self.assertTrue(m['covered']); self.assertEqual(m['L_bracket'],[40,80]); self.assertAlmostEqual(m['expanded'],75.)
    def test_exact_target_needs_no_interpolation(self):
        m=Q.matched([point(40,.95,50),point(80,1.,100)])
        self.assertEqual(m['L_bracket'],[40,40]); self.assertEqual(m['interpolation_weight'],0)
    def test_pairing_is_required(self):
        a=point(40,.9,50); b=point(80,1.,100); b['queries'][0]['query_id']=99
        with self.assertRaises(RuntimeError): Q.matched([a,b])
    def test_negative_gain_preserved(self):
        a=Q.matched([point(40,.95,100)]); b=Q.matched([point(40,.95,120)])
        self.assertLess(Q.bootstrap_gain(a,b)['gain'],0)
    def test_specialization_requires_all_sites(self):
        with self.assertRaises(ValueError): S.specialize('hvs_metric_distance(_distance->compare(_data1, node_coords1, _aligned_dim1), _dist_metric)')
    def test_full_specialization_covers_all_calls(self):
        from prepare_variant import rewrite_calls
        p=METRIC/'sources/hybrid_index.cpp'
        corrected,count=rewrite_calls(p.read_text(encoding='utf-8'),'_dist_metric'); self.assertEqual(count,14)
        text=S.specialize(corrected); self.assertEqual(text.count('hvs_fixed_fuse('),15)
        self.assertIn('hvs_metric_distance(_distance->compare(_data2',text)
if __name__=='__main__': unittest.main()
