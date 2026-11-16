"""Focused checks of routing, source hooks, work accounting and native geometry."""
import unittest,sys
import common as c
import compile_relaxed,compile_native,openimages as O,native_compare as N
class Tests(unittest.TestCase):
 def test_three_routes(self):
    source=(c.MG/'scripts/multigraph.cpp').read_bytes().decode('utf-8');out=O.transform(source)
    a=out[out.index('static unsigned route'):out.index('  Engine(')]
    self.assertIn('indices[] = {2,4,6}',a);self.assertNotIn('.3f',a);self.assertNotIn('.7f',a)
    self.assertNotIn(chr(92)*3+'"expanded',out)
 def test_relaxed_hook_only_actual_pruner(self):
    source=(c.U/'stage2-v2/variant/unified/hybrid_index.cpp').read_bytes().decode('utf-8');out=compile_relaxed.transform(source)
    self.assertEqual(out.count('xq_e_dist *= hvs_unified_rho()'),1)
    self.assertEqual(out.count('xq_s_dist *= hvs_unified_rho()'),1)
 def test_native_counts_and_objective(self):
    source=(c.E.parent/'source/src/search/beam_search.cpp').read_bytes().decode('utf-8')
    a=compile_native.transform(source,False);b=compile_native.transform(source,True)
    self.assertNotIn('native_measure::counts.pq_candidates +=',a)
    self.assertEqual(b.count('native_measure::counts.pq_candidates += n_ids'),1)
    self.assertEqual(b.count('native_measure::counts.pq_spaces += n_ids'),2)
    self.assertIn('stats->read_size = native_measure::counts.requested_bytes',b)
 def test_openimages_compute_dimensions(self):
    p=dict(graphs=[dict(topology_pages=2,pq_space_evals=3)],coordinate_pages=4,reranked=5)
    self.assertEqual(O.work(p)['compute'],3*64+5*1536)
 def test_native_frontier_never_latency(self):
    ps=[dict(L=40,recall=.94,coupled_backend_pages=10,pq_space_evals=100,exact_space_evals=10,latency_us=1),dict(L=80,recall=.96,coupled_backend_pages=20,pq_space_evals=150,exact_space_evals=20,latency_us=100)]
    a=N.native_frontier(ps)
    for p in ps:p['latency_us']*=1e9
    self.assertEqual(a,N.native_frontier(ps))
 def test_vamana_source_only_meta(self):
    import tempfile,struct
    from pathlib import Path
    with tempfile.TemporaryDirectory() as folder:
        root=Path(folder);meta=bytearray(8192);struct.pack_into('<5I',meta,0,256,512,512,40,0);struct.pack_into('<Q',meta,20,49)
        (root/'compact.meta').write_bytes(meta)
        self.assertEqual(O.source_meta(root),root/'compact.meta')
        (root/'full.meta').write_bytes(meta)
        self.assertEqual(O.source_meta(root),root/'full.meta')
 def test_noncontiguous_numa_affinity_guard(self):
    import edbt_monitor as M
    old=M.CPUS;M.CPUS=list(range(72,96))+list(range(168,176))
    try:
        samples=[dict(time=float(i),cpu=0,own=0,disks={'nvme3n1':dict(write_bytes=0)},numa=dict(cpu_mask='72-95,168-175',verified=True,conflicting_policies=[])) for i in range(12)]
        point=dict(point_start_unix=0.,measured_seconds=11.)
        self.assertTrue(M.validate(samples,point)['accepted'])
        samples[5]['numa']['cpu_mask']='72-103'
        self.assertFalse(M.validate(samples,point)['accepted'])
    finally:M.CPUS=old
if __name__=='__main__':unittest.main()
