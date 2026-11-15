import unittest
import protocol as p

def point(label,recall,pages=10,pq=100,latency=1):
    return dict(label=label,recall=recall,L=40,kx=20,ky=20,reranked=20,coordinate_pages=5,coordinate_submitted_pages=5,coordinate_wait_us=1,candidates_before_union=40,candidates_after_union=30,latency_us=latency,qps=100,p95_us=9,graphs=[dict(topology_pages=pages,pq_space_evals=pq,expanded=30,topology_submitted_pages=pages,topology_wait_us=2)])
class Protocol(unittest.TestCase):
    def test_work_ignores_timing(self):
        a=point('a',.96,10,100,1);b=point('b',.96,5,200,1000)
        f=p.frontiers([a,b]);a['latency_us']=1e9;b['latency_us']=.001
        self.assertEqual(f,p.frontiers([a,b]));self.assertEqual(f['pages'],['b'])
    def test_allocation_fixed_total(self):
        self.assertEqual(p.allocations(10,.1),(2,18));self.assertEqual(p.allocations(10,.9),(18,2))
        for a in (0,.1,.5,.9,1):self.assertEqual(sum(p.allocations(10,a)),20)
    def test_grid_bounds(self):
        self.assertEqual(len(p.LEVELS),8)
        for method in p.METHODS:
            for j in p.jobs(method,.1,.2):self.assertLessEqual(max(j['kx'],j['ky']),j['L'])
    def test_no_extrapolation(self):self.assertFalse(p.matched([point('a',.92)],.95)['covered'])
    def test_tail_not_interpolated(self):
        a=point('a',.9);b=point('b',1.)
        r=p.matched([a,b],.95);self.assertNotIn('p95_us',r);self.assertEqual(r['raw_p95_us'],{'a':9,'b':9})
    def test_frozen_order_preserved(self):
        self.assertIsNone(p.crossing([point('high',.97),point('low',.90)],.95))
if __name__=='__main__':unittest.main()
