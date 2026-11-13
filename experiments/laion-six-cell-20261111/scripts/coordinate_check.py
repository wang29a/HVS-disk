"""Verify shared original-ID coordinate pages and freeze all data hashes."""
import hashlib
import json
import numpy as np
import common as c
from gt_audit import matrix


def main():
    x, y = matrix(c.DATA/'base_img_emb.fvecs'), matrix(c.DATA/'base_text_emb.fvecs')
    n, dim = len(x), x.shape[1]+y.shape[1]
    npp = 8192//(dim*4)
    physical_rows = ((n+npp-1)//npp)*npp
    if c.COORD.stat().st_size != ((n+npp-1)//npp)*8192:
        raise RuntimeError('coordinate page size')
    data = np.memmap(str(c.COORD), dtype=np.float32, mode='r', shape=(physical_rows, dim))
    hashes = {k:hashlib.sha256() for k in ('base_x', 'base_y', 'coordinate')}
    for first in range(0, n, 16384):
        end = min(n, first+16384)
        xx, yy, actual = np.asarray(x[first:end]), np.asarray(y[first:end]), data[first:end]
        if not np.array_equal(actual[:, :x.shape[1]], xx) or not np.array_equal(actual[:, x.shape[1]:], yy):
            raise RuntimeError('coordinate identity at node '+str(first))
    if np.any(data[n:]):
        raise RuntimeError('nonzero final coordinate padding')
    # Hash complete files, including headers and final page padding.
    result = dict(status='PASS', nodes=n, dimensions=[x.shape[1], y.shape[1]], coordinates_per_page=npp,
                  coordinate_bytes=c.COORD.stat().st_size, all_values_compared=True,
                  hashes={str(p):c.digest(p) for p in (c.DATA/'base_img_emb.fvecs', c.DATA/'base_text_emb.fvecs', c.COORD)})
    c.save(c.ROOT/'analysis/coordinate-identity.json', result)


if __name__ == '__main__':
    main()
