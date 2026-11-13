#!/usr/bin/env python3
"""Validate Starling's raw partition and produce position -> global ID order."""
import json
from pathlib import Path
import struct
import sys
import numpy as np

def convert(partition, output, report):
    raw=Path(partition).read_bytes()
    capacity,pages,n=struct.unpack_from('<QQQ',raw)
    if not capacity or pages!=(n+capacity-1)//capacity:raise ValueError('partition header')
    matrix=np.full((pages,capacity),np.iinfo(np.uint32).max,dtype=np.uint32)
    seen=np.zeros(n,dtype=np.uint8);offset=24
    for p in range(pages):
        count=struct.unpack_from('<I',raw,offset)[0];offset+=4
        if count>capacity:raise ValueError('partition overflow')
        ids=np.frombuffer(raw,dtype='<u4',count=count,offset=offset);offset+=count*4
        if np.any(ids>=n) or len(np.unique(ids))!=count or np.any(seen[ids]):raise ValueError('duplicate/out-of-range IDs')
        seen[ids]=1;matrix[p,:count]=ids
    if offset+n*4!=len(raw) or not np.all(seen):raise ValueError('partition coverage/length')
    page_of=np.frombuffer(raw,dtype='<u4',count=n,offset=offset)
    sentinel=np.iinfo(np.uint32).max
    for p in range(pages):
        ids=matrix[p];ids=ids[ids!=sentinel]
        if np.any(page_of[ids]!=p):raise ValueError('ID to page mismatch')
    # Fill internal holes from the final occupied positions before flattening.
    # Otherwise a single early hole would shift every subsequent page boundary.
    flat=matrix.reshape(-1);right=len(flat)-1;moved=0
    for hole in np.flatnonzero(flat[:n]==sentinel):
        while right>hole and flat[right]==sentinel:right-=1
        if right<=hole:raise ValueError('hole compaction')
        flat[hole]=flat[right];flat[right]=sentinel;right-=1;moved+=1
    order=flat[:n]
    if np.any(order==sentinel) or len(np.unique(order))!=n:raise ValueError('order coverage')
    with open(str(output),'wb') as f:f.write(struct.pack('<Q',n));f.write(order.astype('<u4').tobytes())
    Path(report).write_text(json.dumps(dict(status='PASS',nodes=n,capacity=capacity,pages=pages,
        internal_holes_filled=moved,mapping_format='uint64 N; uint32 position_to_global_ID[N]',
        neighbor_ids_unchanged=True),indent=2)+'\n')

if __name__=='__main__':convert(*sys.argv[1:])
