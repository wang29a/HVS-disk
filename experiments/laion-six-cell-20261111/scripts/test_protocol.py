"""Boundary checks for the actual observed final-partial-page failure and reporting rules."""
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
import common as c
import coordinate_check
import run_queue as q
import summarize as s


class Protocol(unittest.TestCase):
    def coordinate_fixture(self,n,corrupt=None):
        original=c.DATA,c.COORD,c.ROOT
        with tempfile.TemporaryDirectory(prefix='laion-protocol-') as folder:
            root=Path(folder)
            c.DATA,c.COORD,c.ROOT=root,root/'coordinates',root
            values=np.arange(n*1024,dtype=np.float32).reshape(n,1024)
            for name,x in [('base_img_emb.fvecs',values[:,:512]),('base_text_emb.fvecs',values[:,512:])]:
                with (root/name).open('wb') as out:
                    out.write(struct.pack('<II',n,512)); x.tofile(out)
            stored=np.zeros(((n+1)//2*2,1024),dtype=np.float32)
            stored[:n]=values
            if corrupt=='data': stored[-2,5]+=2
            if corrupt=='padding': stored[-1,0]=1
            stored.tofile(str(c.COORD))
            try:
                coordinate_check.main()
            finally:
                c.DATA,c.COORD,c.ROOT=original

    def test_complete_page(self): self.coordinate_fixture(4)
    def test_partial_page(self): self.coordinate_fixture(5)
    def test_partial_final_batch(self): self.coordinate_fixture(16385)
    def test_real_vector_corruption_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'coordinate identity'): self.coordinate_fixture(5,'data')
    def test_nonzero_padding_rejected(self):
        with self.assertRaisesRegex(RuntimeError,'padding'): self.coordinate_fixture(5,'padding')
    def test_no_extrapolation(self):
        x={k:1. for k in s.METRICS}
        x.update(recall=94.,L=60,rerank_limit=40,p95_us=50.)
        y=dict(x,recall=94.5,L=80)
        self.assertIsNone(s.interpolate([('x',x),('y',y)],95))
    def test_matched_recall_keeps_endpoints(self):
        x={k:2. for k in s.METRICS}
        x.update(recall=94.,L=60,rerank_limit=40,p95_us=500.)
        y={k:4. for k in s.METRICS}
        y.update(recall=96.,L=80,rerank_limit=60,p95_us=800.)
        result=s.interpolate([('x',x),('y',y)],95)
        self.assertEqual(result['values']['latency_us'],3.)
        self.assertNotIn('p95_us',result['values'])
        self.assertEqual([e['r'] for e in result['endpoints']],[40,60])
    def test_scope_rejects_extra_arms(self):
        with self.assertRaisesRegex(RuntimeError,'only six-cell'): q.layout('compact','T_XY')
        with self.assertRaisesRegex(RuntimeError,'only six-cell'): q.layout('full','TXY')


if __name__=='__main__': unittest.main()
