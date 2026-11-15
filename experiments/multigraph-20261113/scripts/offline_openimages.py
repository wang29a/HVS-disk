"""Regroup existing exact-union evidence; never launch a remote experiment."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np

EDGES = [0., .05, .2, .4, .6, .8, .95, 1.]

def matrix(path, dtype='<f4'):
    with Path(path).open('rb') as f:
        n, d = struct.unpack('<II', f.read(8))
    if Path(path).stat().st_size != 8 + n*d*4:
        raise RuntimeError('matrix size: '+str(path))
    return np.memmap(str(path), dtype=dtype, mode='r', offset=8, shape=(n, d))

def group(rows, key):
    result = []
    for lo, hi in zip(EDGES, EDGES[1:]):
        selected = [r for r in rows if r[key] is not None and lo <= r[key] and
                    (r[key] < hi or hi == 1. and r[key] == hi)]
        budgets = sorted(set(p['per_space_k'] for r in selected for p in r['points']))
        result.append(dict(lo=lo, hi=hi, queries=len(selected), points=[dict(
            per_space_k=k,
            recall_percent=float(np.mean([next(p for p in r['points'] if p['per_space_k']==k)['covered_gt'] for r in selected])*10),
            mean_union_candidates=float(np.mean([next(p for p in r['points'] if p['per_space_k']==k)['union_candidates'] for r in selected]))) for k in budgets]))
    return result

def run(source, output):
    source, output = Path(source), Path(output)
    coverage = source/'analysis/dual-index-oracle-coverage.json'
    c = json.loads(coverage.read_text())
    s = source/'snapshots/openimage'
    meta = json.loads((s/'manifest.json').read_text())
    ids = matrix(s/'base_ids.bin', '<u4')[:, 0]
    x, y = matrix(s/'base_img.bin'), matrix(s/'base_text.bin')
    qx, qy = matrix(s/'query_img.bin'), matrix(s/'query_text.bin')
    truth = matrix(s/'gt-mixed.bin', '<u4')
    alphas = matrix(s/'alpha-mixed.bin')[:, 0]
    by_id = {int(q):i for i,q in enumerate(meta['query_ids'])}
    rows = []
    for r in c['per_query']:
        i = by_id[r['query_id']]
        if float(alphas[i]) != r['alpha']:
            raise RuntimeError('native weight mismatch')
        pos = np.searchsorted(ids, truth[i])
        if not np.array_equal(ids[pos], truth[i]):
            raise RuntimeError('GT neighbor snapshot missing')
        dx = np.linalg.norm(x[pos].astype(np.float64)-qx[i].astype(np.float64), axis=1)
        dy = np.linalg.norm(y[pos].astype(np.float64)-qy[i].astype(np.float64), axis=1)
        a = float(alphas[i]); denom = a*dx+(1-a)*dy
        finite = denom > 0
        share = float(np.mean(a*dx[finite]/denom[finite])) if finite.any() else None
        rows.append(dict(r, image_contribution=share, zero_distance_neighbors=int((~finite).sum())))
    result = dict(dataset='OpenImages', queries=len(rows),
        source_sha256=hashlib.sha256(coverage.read_bytes()).hexdigest(),
        definition='per-query mean image contribution over native GT top10, zero-total pairs excluded; undefined queries separate',
        scope='existing 256 sampled queries; exact per-space candidate coverage, not ANN or SSD timing',
        nominal_alpha=group(rows, 'alpha'), actual_image_contribution=group(rows, 'image_contribution'),
        undefined_contribution_query_ids=[r['query_id'] for r in rows if r['image_contribution'] is None], per_query=rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    for key in ('nominal_alpha','actual_image_contribution'):
        print(key)
        for row in result[key]:
            p=next((p for p in row['points'] if p['per_space_k']==320),None)
            print(row['lo'], row['hi'], row['queries'], p)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('source');p.add_argument('output');a=p.parse_args();run(a.source,a.output)
