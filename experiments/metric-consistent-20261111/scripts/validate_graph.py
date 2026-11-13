"""Independent float-source to Full/compact record and page-padding audit."""
import json
from pathlib import Path
import struct
import numpy as np


def validate(directory, n, d1, d2):
    directory = Path(directory)
    groups = {}
    for name in ('full', 'compact'):
        meta = directory/(name+'.meta')
        if not meta.exists() and name == 'full':
            continue
        header = meta.read_bytes()
        nn, dd1, dd2, degree, slots, npp = struct.unpack_from('<5IQ', header)
        if len(header) != 8192 or (nn, dd1, dd2, degree, npp) != (n, d1, d2, 40, 8192//(4+40*(4+2*slots))):
            raise RuntimeError('canonical metadata')
        width = 4+degree*(4+2*slots)
        topology = np.memmap(str(directory/(name+'.topo')), mode='r', dtype=np.uint8)
        if len(topology) != ((n+npp-1)//npp)*8192:
            raise RuntimeError('canonical size/tail')
        groups[name] = (topology, width, npp, slots)
    edges, pairs = 0, 0
    with (directory/'float-graph.bin').open('rb') as source:
        if struct.unpack('<QII', source.read(16)) != (n, d1, d2):
            raise RuntimeError('float source header')
        for i in range(n):
            degree, = struct.unpack('<I', source.read(4))
            rows = {}
            for name, (data, width, npp, slots) in groups.items():
                offset = (i//npp)*8192+(i%npp)*width
                row = memoryview(data[offset:offset+width])
                if struct.unpack_from('<I', row)[0] != degree:
                    raise RuntimeError('degree identity')
                rows[name] = (row, slots)
                if any(row[4+degree*(4+slots*2):]):
                    raise RuntimeError('nonzero unused edge slots')
            for e in range(degree):
                identity, count = struct.unpack('<II', source.read(8))
                floating = [struct.unpack('<ff', source.read(8)) for _ in range(count)]
                for name, (row, slots) in rows.items():
                    offset = 4+e*(4+slots*2)
                    if struct.unpack_from('<I', row, offset)[0] != identity:
                        raise RuntimeError('neighbor ID/order mismatch')
                    if slots:
                        encoded = [(int(row[offset+4+2*k]), int(row[offset+5+2*k])) for k in range(slots)]
                        active = [p for p in encoded if p != (0, 0)]
                        for lo, hi in floating:
                            if not any(a/100. <= lo and b/100. >= hi for a, b in active):
                                raise RuntimeError('narrowed float source interval')
                        if any(a > b or b > 100 for a, b in active) or active != sorted(active):
                            raise RuntimeError('range encoding/order')
                        for k, p in enumerate(encoded):
                            if p == (0, 0) and any(q != (0, 0) for q in encoded[k:]):
                                raise RuntimeError('active range after sentinel')
                edges += 1
                pairs += count
        if source.read(1):
            raise RuntimeError('trailing float graph bytes')
    for data, width, npp, slots in groups.values():
        for first in range(0, n, npp):
            page = first//npp
            used = min(npp, n-first)*width
            if np.any(data[page*8192+used:(page+1)*8192]):
                raise RuntimeError('page padding mismatch')
    result = dict(status='PASS', nodes=n, edges=edges, float_pairs=pairs,
                  id_order_mismatches=0, narrowed_ranges=0, padding_mismatches=0,
                  actual_full_slots=groups.get('full', groups['compact'])[3])
    return result
