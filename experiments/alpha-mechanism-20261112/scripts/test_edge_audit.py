"""Independent tiny graph: inclusive endpoints and first-discovery attribution."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile


def main(binary):
    with tempfile.TemporaryDirectory(prefix='alpha-edge-test-') as name:
        root=Path(name)
        R,M,n=12,2,12
        width=4+R*(4+2*M)
        meta=bytearray(8192)
        struct.pack_into('<5IQI',meta,0,n,8,8,R,M,8192//width,1)
        struct.pack_into('<I',meta,32,0)
        (root/'meta').write_bytes(meta)
        topo=bytearray(8192)
        for i in range(n):
            edges=([(j,0,50) if j<=6 else (j,50,100) for j in range(1,12)] if i==0
                   else [(7 if i==1 else 0,0,100)])
            struct.pack_into('<I',topo,i*width,len(edges))
            for e,(to,a,b) in enumerate(edges):struct.pack_into('<IBB',topo,i*width+4+e*8,to,a,b)
        (root/'topo').write_bytes(topo)
        subprocess.check_call([binary,'static',str(root/'meta'),str(root/'topo'),str(root/'static.json')])
        a=json.loads((root/'static.json').read_text())
        assert a['edges']==22
        assert [a['activity'][i]['active_edges'] for i in (0,25,50,100)]==[17,17,22,16]
        trace=[0x48565331,1,2,1,0,1,1,10]+list(range(1,11))
        (root/'trace').write_bytes(struct.pack('<'+'I'*len(trace),*trace))
        (root/'alpha').write_bytes(struct.pack('<IIf',1,1,.25))
        raw_ids=struct.pack('<10I',*range(10))
        (root/'ids').write_bytes(raw_ids)
        (root/'gt').write_bytes(struct.pack('<II',1,10)+raw_ids)
        command=[binary,'trace',str(root/'meta'),str(root/'topo'),str(root/'trace'),str(root/'alpha'),str(root/'ids'),str(root/'gt'),str(root/'trace.json')]
        subprocess.check_call(command)
        a=json.loads((root/'trace.json').read_text())[0]
        assert a['expanded_nodes']==2 and a['inactive_first_discovery_edges']==5
        assert (a['candidate_nonentry'],a['candidate_inactive_parent'])==(10,4)
        assert (a['returned_nonentry'],a['returned_inactive_parent'])==(9,3)
        assert (a['true_hits'],a['true_nonentry'],a['true_inactive_parent'])==(10,9,3)
        # Node 7 has an alternative active edge from node 1. It must retain
        # its original inactive discovery edge, not its last observed edge.
        struct.pack_into('<BB',topo,8,90,10)
        (root/'topo').write_bytes(topo)
        assert subprocess.call([binary,'static',str(root/'meta'),str(root/'topo'),str(root/'bad.json')],stderr=subprocess.DEVNULL)!=0
    print('PASS: exact endpoint counts, unused slots, original first parent, seed exclusion, corrupt interval rejection')


if __name__=='__main__':main(sys.argv[1])
