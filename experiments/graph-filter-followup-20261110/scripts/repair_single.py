"""Attach a small unreachable tail while retaining an existing reachability tree."""
import os
import struct
import subprocess

import numpy as np


def repair(source, target, vectors, missing_count, page):
    n, dim = vectors.shape
    with source.open('rb') as stream:
        nodes, width_dim, entry, record, npp = struct.unpack_from('<5Q', stream.read(page), 8)
    if (nodes, width_dim, record, npp) != (n, dim, (dim+41)*4, 1):
        raise ValueError('repair requires the CC3M one-node-per-page layout')
    degree_limit = 40
    raw = np.memmap(str(source), mode='r', dtype=np.uint8)
    rows = np.ndarray((n, degree_limit+1), dtype='<u4', buffer=raw,
                      offset=page+dim*4, strides=(page, 4))
    graph = np.full((n, degree_limit), n, dtype=np.uint32)
    degrees = rows[:,0].copy()
    for lo in range(0,n,8192):
        hi = min(lo+8192,n)
        active = np.arange(degree_limit) < degrees[lo:hi,None]
        graph[lo:hi][active] = rows[lo:hi,1:][active]

    # Keep an explicit tree from the entry so replacing any other edge is safe.
    parents = np.full(n+1, -1, dtype=np.int32)
    parents[[entry,n]] = entry
    frontier = np.array([entry], dtype=np.uint32)
    while len(frontier):
        edges = graph[frontier].ravel()
        slots = np.flatnonzero(parents[edges] == -1)
        children, first = np.unique(edges[slots], return_index=True)
        parents[children] = frontier[slots[first]//degree_limit]
        frontier = children
    missing = np.flatnonzero(parents[:n] == -1)
    if len(missing) != missing_count or not 0 < len(missing) <= 16:
        raise RuntimeError('unexpected unreachable-node count: {}'.format(len(missing)))

    # Find distinct nearby reachable anchors that can spare an edge.
    targets = vectors[missing].astype(np.float64)
    best = [[] for _ in missing]
    for lo in range(0,n,8192):
        hi = min(lo+8192,n)
        ids = np.arange(lo,hi)
        can_replace = (parents[graph[lo:hi]] != ids[:,None]) & (graph[lo:hi] != n)
        eligible = (parents[ids] != -1) & ((degrees[ids] < degree_limit) | can_replace.any(axis=1))
        ids = ids[eligible]
        if not len(ids):
            continue
        block = vectors[ids].astype(np.float64)
        # Target norms are constant while ranking candidates for each target.
        scores = np.einsum('ij,ij->i',block,block)[:,None] - 2*block.dot(targets.T)
        for j, col in enumerate(scores.T):
            top = np.lexsort((ids,col))[:len(missing)]
            best[j] = sorted(best[j]+[(float(col[k]),int(ids[k])) for k in top])[:len(missing)]

    tmp = target.with_suffix('.partial')
    subprocess.run(['cp','--reflink=auto','--',str(source),str(tmp)],check=True)
    changes, used = [], set()
    with tmp.open('r+b') as stream:
        for dst, candidates in zip(missing,best):
            anchor = next(node for _,node in candidates if node not in used)
            used.add(anchor)
            record = rows[anchor].copy()
            degree = int(record[0])
            replaced = None
            if degree < degree_limit:
                record[degree+1] = dst
                record[0] += 1
            else:
                neighbors = record[1:]
                slots = np.flatnonzero(parents[neighbors] != anchor)
                delta = vectors[neighbors[slots]].astype(np.float64) - vectors[anchor]
                slot = slots[np.argmax(np.einsum('ij,ij->i',delta,delta))]
                replaced = int(neighbors[slot])
                neighbors[slot] = dst
            stream.seek((1+anchor)*page+dim*4)
            stream.write(record.tobytes())
            changes.append(dict(anchor=anchor, added=int(dst), removed=replaced))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(str(tmp),str(target))
    return dict(original_entry_reachable=n-len(missing), added_edges=changes,
                policy='nearest eligible reachable anchor; preserve original reachability tree')
