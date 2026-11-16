"""Portable parser/gate boundary checks; real C++ semantic tests run on node3."""
import contextlib,io,json,struct,tempfile,unittest
from pathlib import Path
import function_check as V

def payload(hi=1.,id=1):return struct.pack('<IIffIIff',1,id,2.,3.,0,1,0.,hi)

class FunctionGateTest(unittest.TestCase):
 def run_case(self,exact=10000,unexplained=False):
  with tempfile.TemporaryDirectory() as t:
   root=Path(t);(root/'results').mkdir();(root/'validation').mkdir();old=V.F;V.F=root
   try:
    with (root/'capture.bin').open('wb') as f:
     f.write(b'HVSREV2\0')
     for i in range(10000):f.write(struct.pack('<QIIIIffff',23*i,0,1,1,1,2.,3.,0.,0.)+payload())
    for mode in ['B']+['author-'+str(m) for m in range(8)]:
     with (root/'results'/(mode+'.bin')).open('wb') as f:
      f.write(b'HVSOUT2\0')
      for i in range(10000):
       hi=1. if mode=='B' or mode=='author-7' and i<exact else .5
       if mode=='author-7' and i>=exact:hi=struct.unpack('<f',struct.pack('<I',0x3f7fffff))[0]
       f.write(struct.pack('<Q',23*i)+payload(hi,id=2 if unexplained and mode=='author-7' and i==9999 else 1))
    try:V.main()
    except RuntimeError:pass
    return json.loads((root/'validation/function-gate.json').read_text())
   finally:V.F=old
 def test_exact(self):
  r=self.run_case();self.assertEqual(r['status'],'PASS');self.assertEqual(r['minimal_fix_set_counts'],{'7':10000})
 def test_9990_exact_plus_explained_last_bits(self):
  r=self.run_case(exact=9990);self.assertEqual(r['status'],'PASS');self.assertEqual(r['corrected_residual_count'],10)
 def test_9989_fails_even_if_all_residuals_explained(self):self.assertEqual(self.run_case(exact=9989)['status'],'STOP_AT_FUNCTION_GATE')
 def test_unexplained_fails_even_above_percentage_threshold(self):self.assertEqual(self.run_case(unexplained=True)['status'],'STOP_AT_FUNCTION_GATE')

if __name__=='__main__':unittest.main()
