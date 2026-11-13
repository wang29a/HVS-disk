"""Float64 full-base scan, fixed sample, two existing alpha-specific GTs."""
import json
from pathlib import Path
import struct
import numpy as np
import common as c


def matrix(path, dtype=np.float32):
    path = Path(path)
    with path.open('rb') as stream:
        n, d = struct.unpack('<II', stream.read(8))
    if path.stat().st_size != 8+n*d*4:
        raise RuntimeError('matrix format/size: '+str(path))
    return np.memmap(str(path), dtype=dtype, mode='r', offset=8, shape=(n, d))


def update(best, score, offset):
    candidate = np.argpartition(score, 9, axis=1)[:, :10]
    values = score[np.arange(len(score))[:, None], candidate]
    ids = candidate.astype(np.uint32)+offset
    old_values, old_ids = best
    values = np.concatenate([old_values, values], axis=1)
    ids = np.concatenate([old_ids, ids], axis=1)
    order = np.lexsort((ids, values), axis=1)[:, :10]
    return values[np.arange(len(score))[:, None], order], ids[np.arange(len(score))[:, None], order]


def main():
    inventory = json.loads((c.ROOT/'analysis/inventory.json').read_text())
    x, y = matrix(c.DATA/'base_img_emb.fvecs'), matrix(c.DATA/'base_text_emb.fvecs')
    qx, qy = matrix(c.DATA/'query_img_emb.fvecs'), matrix(c.DATA/'query_text_emb.fvecs')
    chosen = np.random.RandomState(20261111).permutation(len(qx))[:32]
    a, b = np.asarray(qx[chosen], dtype=np.float64), np.asarray(qy[chosen], dtype=np.float64)
    qa, qb = np.sum(a*a, axis=1)[:, None], np.sum(b*b, axis=1)[:, None]
    best = {}
    for w in inventory['workloads']:
        alpha = matrix(w['alpha'])[:, 0]
        w['sample_alpha'] = np.asarray(alpha[chosen], dtype=np.float64)
        for metric in ('root', 'squared'):
            best[(w['condition'], metric)] = (np.full((len(chosen), 10), np.inf),
                                              np.full((len(chosen), 10), 2**32-1, dtype=np.uint32))
    for start in range(0, len(x), 32768):
        xx, yy = np.asarray(x[start:start+32768], dtype=np.float64), np.asarray(y[start:start+32768], dtype=np.float64)
        d1 = np.maximum(0, qa+np.einsum('ij,ij->i', xx, xx)[None, :]-2*a.dot(xx.T))
        d2 = np.maximum(0, qb+np.einsum('ij,ij->i', yy, yy)[None, :]-2*b.dot(yy.T))
        e1, e2 = np.sqrt(d1), np.sqrt(d2)
        for w in inventory['workloads']:
            weight = w['sample_alpha'][:, None]
            for metric, score in [('root', weight*e1+(1-weight)*e2),
                                  ('squared', weight*d1+(1-weight)*d2)]:
                key = (w['condition'], metric)
                best[key] = update(best[key], score, start)
    result = dict(status='PASS', dataset='Laion', nodes=len(x), dimensions=[x.shape[1], y.shape[1]],
                  sample_query_ids=chosen.tolist(), sampled_queries_per_alpha=len(chosen),
                  precision='float64 streaming exhaustive scan over every base vector',
                  target='alpha*sqrt(squared_l2_x)+(1-alpha)*sqrt(squared_l2_y)', workloads=[])
    for w in inventory['workloads']:
        gt = matrix(w['gt'], np.uint32)
        roots, squares = best[(w['condition'], 'root')][1], best[(w['condition'], 'squared')][1]
        rows = [dict(query_id=int(q), gt_root_overlap=len(set(gt[q]) & set(roots[j]))/10.,
                     root_squared_overlap=len(set(roots[j]) & set(squares[j]))/10.,
                     root_top10=roots[j].tolist()) for j, q in enumerate(chosen)]
        overlap = float(np.mean([row['gt_root_overlap'] for row in rows]))
        result['workloads'].append(dict(alpha=w['fixed_alpha'], provided_gt_root_top10_overlap=overlap,
            root_squared_top10_overlap=float(np.mean([row['root_squared_overlap'] for row in rows])), queries=rows))
        if overlap < .99:
            result['status'] = 'FAIL_GT_SEMANTICS'
    c.save(c.ROOT/'analysis/gt-objective-audit.json', result)
    if result['status'] != 'PASS':
        raise RuntimeError('existing GT does not match unified objective; no full graph build permitted')


if __name__ == '__main__':
    main()
