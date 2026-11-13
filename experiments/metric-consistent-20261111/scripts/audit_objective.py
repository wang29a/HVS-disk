"""Exact sampled objective check before rebuilding. BLAS thread cap set by runner."""
import json
import os
from pathlib import Path
import struct
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = Path('/mnt/nvme3/wz/dataset/Openimage_disk')


def matrix(path, dtype=np.float32):
    with Path(path).open('rb') as stream:
        n, d = struct.unpack('<II', stream.read(8))
    if Path(path).stat().st_size != 8 + n * d * 4:
        raise RuntimeError('matrix size: ' + str(path))
    return np.memmap(str(path), dtype=dtype, mode='r', offset=8, shape=(n, d))


def top10(scores):
    ids = np.argpartition(scores, 10)[:10]
    return ids[np.argsort(scores[ids], kind='mergesort')]


def sample_check():
    x = matrix(DATA / 'base_img_emb.fvecs')
    y = matrix(DATA / 'base_text_emb.fvecs')
    qx = matrix(DATA / 'query_img_emb.fvecs')
    qy = matrix(DATA / 'query_text_emb.fvecs')
    alpha = matrix(DATA / 'base_gt_round_00/query_alpha.fvecs')[:, 0]
    gt = matrix(DATA / 'base_gt_round_00/top10_results.ivecs', np.uint32)
    chosen = np.random.RandomState(20261111).permutation(len(qx))[:32]
    scores_root = np.empty((len(chosen), len(x)), dtype=np.float64)
    scores_sq = np.empty_like(scores_root)
    for first in range(0, len(x), 8192):
        a = np.asarray(x[first:first+8192], dtype=np.float64)
        b = np.asarray(y[first:first+8192], dtype=np.float64)
        d1 = np.maximum(0, np.einsum('ij,ij->i', a, a)[None, :] +
                        np.sum(np.asarray(qx[chosen], dtype=np.float64)**2, axis=1)[:, None] -
                        2 * np.asarray(qx[chosen], dtype=np.float64).dot(a.T))
        d2 = np.maximum(0, np.einsum('ij,ij->i', b, b)[None, :] +
                        np.sum(np.asarray(qy[chosen], dtype=np.float64)**2, axis=1)[:, None] -
                        2 * np.asarray(qy[chosen], dtype=np.float64).dot(b.T))
        weights = alpha[chosen, None]
        scores_root[:, first:first+len(a)] = weights * np.sqrt(d1) + (1-weights) * np.sqrt(d2)
        scores_sq[:, first:first+len(a)] = weights * d1 + (1-weights) * d2
    rows = []
    for i, q in enumerate(chosen):
        root, squared = top10(scores_root[i]), top10(scores_sq[i])
        rows.append(dict(query_id=int(q), alpha=float(alpha[q]),
            root_squared_top10_overlap=len(set(root) & set(squared))/10.,
            provided_gt_root_overlap=len(set(root) & set(gt[q]))/10.,
            provided_gt_squared_overlap=len(set(squared) & set(gt[q]))/10.,
            exact_root_top10=root.tolist(), exact_squared_top10=squared.tolist()))
    result = dict(status='PASS', queries=rows, sample_size=len(rows),
                  precision='float64, exhaustive base scan',
                  mean_root_squared_top10_overlap=float(np.mean([r['root_squared_top10_overlap'] for r in rows])),
                  mean_provided_gt_root_overlap=float(np.mean([r['provided_gt_root_overlap'] for r in rows])),
                  mean_provided_gt_squared_overlap=float(np.mean([r['provided_gt_squared_overlap'] for r in rows])),
                  formula='alpha*sqrt(sum((qx-x)^2))+(1-alpha)*sqrt(sum((qy-y)^2))',
                  extra_distance_normalization=False)
    result['sample_vector_norms'] = {}
    for name, array in [('base_x', x), ('base_y', y), ('query_x', qx), ('query_y', qy)]:
        norms = np.linalg.norm(np.asarray(array[:4096], dtype=np.float64), axis=1)
        result['sample_vector_norms'][name] = dict(min=float(norms.min()), max=float(norms.max()), mean=float(norms.mean()))
    if result['mean_provided_gt_root_overlap'] < .99:
        result['status'] = 'FAIL_GT_SEMANTICS'
    output = ROOT / 'analysis/objective-audit.json'
    temp = output.with_suffix('.partial')
    temp.write_text(json.dumps(result, indent=2)+'\n')
    os.replace(str(temp), str(output))
    if result['status'] != 'PASS':
        raise RuntimeError('provided GT does not match root objective; inspect audit before graph build')


def concat_resources(directory):
    directory = Path(directory)
    directory.mkdir(exist_ok=True)
    x, y = matrix(DATA/'base_img_emb.fvecs'), matrix(DATA/'base_text_emb.fvecs')
    n, d = len(x), x.shape[1]+y.shape[1]
    target = directory/'concat-base.bin'
    if not target.exists():
        tmp = target.with_suffix('.partial')
        with tmp.open('wb') as out:
            out.write(struct.pack('<II', n, d))
            for first in range(0, n, 4096):
                np.concatenate([x[first:first+4096], y[first:first+4096]], axis=1).astype('<f4').tofile(out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(str(tmp), str(target))
    if target.stat().st_size != 8+n*d*4:
        raise RuntimeError('concat base size')
    # This is a different metric, so it requires its own GT. No existing
    # mixed-alpha GT is regenerated or substituted.
    qx, qy = matrix(DATA/'query_img_emb.fvecs'), matrix(DATA/'query_text_emb.fvecs')
    best_ids = np.empty((len(qx), 10), dtype=np.uint32)
    for qfirst in range(0, len(qx), 32):
        a = np.asarray(qx[qfirst:qfirst+32], dtype=np.float64)
        b = np.asarray(qy[qfirst:qfirst+32], dtype=np.float64)
        scores = np.empty((len(a), n), dtype=np.float64)
        for first in range(0, n, 8192):
            bx = np.asarray(x[first:first+8192], dtype=np.float64)
            by = np.asarray(y[first:first+8192], dtype=np.float64)
            scores[:, first:first+len(bx)] = (np.sum(a*a, axis=1)[:, None] +
                np.sum(b*b, axis=1)[:, None] + np.einsum('ij,ij->i', bx, bx)[None, :] +
                np.einsum('ij,ij->i', by, by)[None, :] - 2*a.dot(bx.T) - 2*b.dot(by.T))
        for i, score in enumerate(scores):
            best_ids[qfirst+i] = top10(score)
    path = directory/'concat-gt.bin'
    with path.with_suffix('.partial').open('wb') as out:
        out.write(struct.pack('<II', len(qx), 10))
        best_ids.tofile(out)
    os.replace(str(path.with_suffix('.partial')), str(path))
    return target, path


if __name__ == '__main__':
    import sys
    if len(sys.argv) == 3 and sys.argv[1] == 'concat':
        concat_resources(sys.argv[2])
    else:
        sample_check()
