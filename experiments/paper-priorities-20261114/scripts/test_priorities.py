import json,unittest
from pathlib import Path
import degree_study as D
class Rules(unittest.TestCase):
    def test_pilot_choice_is_degree_only_and_distinct(self):
        data={str(r):dict(degree=d,latency=1000-d) for r,d in [(1,18),(1.025,20),(1.05,24),(1.1,30),(1.2,35),(1.4,38),(1.8,39),(2,40)]}
        self.assertEqual(D.choose(data),[1.05,1.1,1.2])
        for row in data.values():row['latency']=-10000
        self.assertEqual(D.choose(data),[1.05,1.1,1.2])
    def test_no_degree_extrapolation(self):
        rows={'unified':dict(degree=33),'a':dict(degree=18),'b':dict(degree=30)}
        self.assertFalse(D.compare(rows)['covered'])
    def test_equal_degree_rule_uses_both_work_metrics(self):
        def point(d,e,p):return dict(degree=d,exact=dict(covered=True,expanded=e),ssd=dict(covered=True,topology_pages=p))
        rows={'unified':point(30,100,100),'a':point(20,80,70),'b':point(35,80,70)}
        self.assertEqual(D.compare(rows)['verdict'],'centre_graph_work_reducible')
        rows['b']['ssd']['topology_pages']=160
        self.assertEqual(D.compare(rows)['verdict'],'tradeoff_or_inconclusive')
if __name__=='__main__':unittest.main()
