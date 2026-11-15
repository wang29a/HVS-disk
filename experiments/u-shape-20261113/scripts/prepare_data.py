#!/usr/bin/env python3
"""Two-core resumable subset extraction and exhaustive weighted-Euclidean GT.

This prepares data only: it never builds a graph or launches timed search.
"""
import argparse
import contextlib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import time

NAMES = ('base_img_emb.fvecs', 'base_text_emb.fvecs',
         'query_img_emb.fvecs', 'query_text_emb.fvecs')
WEIGHTS = (.1, .5, .9)
OBJECTIVE = 'alpha*sqrt(sum((qx-x)^2))+(1-alpha)*sqrt(sum((qy-y)^2)); no additional normalization'


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.partial')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    os.replace(str(temp), str(path))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def shape(path):
    path = Path(path)
    with path.open('rb') as stream:
        n, d = struct.unpack('<II', stream.read(8))
    if not n or not d or path.stat().st_size != 8 + n * d * 4:
        raise RuntimeError('not a headered four-byte matrix: ' + str(path))
    return n, d


def matrix(path, dtype='<f4'):
    import numpy as np
    n, d = shape(path)
    return np.memmap(str(path), dtype=dtype, mode='r', offset=8, shape=(n, d))


def write_matrix(path, value, dtype):
    import numpy as np
    path = Path(path)
    temp = path.with_name(path.name + '.partial')
    array = np.asarray(value, dtype=dtype)
    with temp.open('wb') as out:
        out.write(struct.pack('<II', *array.shape))
        array.tofile(out)
    os.replace(str(temp), str(path))


def source_id(path):
    n, d = shape(path)
    stat = Path(path).stat()
    return dict(path=str(Path(path).resolve()), rows=n, dims=d,
                bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def validate_split(split, q):
    import numpy as np
    cohorts = split['cohorts']
    ids = {name: list(map(int, cohorts[name]['query_ids'])) for name in ('tune', 'report')}
    joined = ids['tune'] + ids['report']
    if len(set(joined)) != q or sorted(joined) != list(range(q)) or any(not v for v in ids.values()):
        raise RuntimeError('query split must be a complete disjoint partition')
    return ids


def plan(source, out, split_path, count, seed):
    import numpy as np
    sources = {name: source_id(source / name) for name in NAMES}
    x, y, qx, qy = (sources[name] for name in NAMES)
    if x['rows'] != y['rows'] or qx['rows'] != qy['rows']:
        raise RuntimeError('space row counts disagree')
    if x['dims'] != qx['dims'] or y['dims'] != qy['dims']:
        raise RuntimeError('base/query dimensions disagree')
    if not 10 <= count <= x['rows']:
        raise RuntimeError('invalid subset size')
    split = json.loads(Path(split_path).read_text())
    cohorts = validate_split(split, qx['rows'])
    value = dict(source=sources, subset_nodes=count, subset_seed=seed,
                 sampling='numpy RandomState(MT19937).choice without replacement; sorted original IDs',
                 split_sha256=digest(split_path), query_cohorts=cohorts,
                 alpha_decimal=list(WEIGHTS), objective=OBJECTIVE, gt_top_k=10,
                 identity_note='source size/mtime/header gates plus hashes of extracted contents; not a full-source hash')
    path = out / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != value:
        raise RuntimeError('inputs or parameters changed: use a new output namespace')
    out.mkdir(parents=True, exist_ok=True)
    save(path, value)
    ids_path = out / 'source_ids.bin'
    if not ids_path.exists():
        ids = np.sort(np.random.RandomState(seed).choice(x['rows'], count, replace=False)).astype('<u4')
        write_matrix(ids_path, ids[:, None], '<u4')
    ids = matrix(ids_path, '<u4')[:, 0]
    expected = np.sort(np.random.RandomState(seed).choice(x['rows'], count, replace=False)).astype('<u4')
    if not np.array_equal(ids, expected):
        raise RuntimeError('subset original-ID map changed')
    value['source_ids_sha256'] = digest(ids_path)
    save(out / 'identity.json', value)
    return value


def extract_file(source, ids, dest, out, block_rows):
    import numpy as np
    original = source_id(source)
    data = matrix(source)
    temp = dest.with_name(dest.name + '.partial')
    progress = dest.with_name(dest.name + '.progress.json')
    done = 0
    if progress.exists():
        state = json.loads(progress.read_text())
        if state['source'] != original or state['source_ids_sha256'] != digest(out / 'source_ids.bin'):
            raise RuntimeError('extraction resume identity changed')
        done = state['completed_rows']
    width = data.shape[1] * 4
    if not 0 <= done <= len(ids):
        raise RuntimeError('invalid extraction checkpoint')
    if not temp.exists():
        if done:
            raise RuntimeError('missing partial extraction')
        with temp.open('wb') as stream:
            stream.write(struct.pack('<II', len(ids), data.shape[1]))
    if temp.stat().st_size < 8 + done * width:
        raise RuntimeError('partial extraction shorter than its checkpoint')
    mapping_sha = digest(out / 'source_ids.bin')
    with temp.open('r+b') as stream:
        stream.truncate(8 + done * width)
        stream.seek(8 + done * width)
        for first in range(done, len(ids), block_rows):
            end = min(len(ids), first + block_rows)
            values = np.asarray(data[ids[first:end]], dtype='<f4')
            if not np.isfinite(values).all():
                raise RuntimeError('nonfinite source vector')
            if shutil.disk_usage(str(out)).free < values.nbytes + (2 << 30):
                raise RuntimeError('disk reserve below 2 GiB during extraction; partial output retained')
            values.tofile(stream)
            stream.flush()
            os.fsync(stream.fileno())
            save(progress, dict(source=original, source_ids_sha256=mapping_sha, completed_rows=end))
            save(out / 'status.json', dict(stage='extracting', file=dest.name, completed=end,
                                          total=len(ids), updated_unix=time.time(), pid=os.getpid()))
    if source_id(source) != original:
        raise RuntimeError('source changed while extracting')
    os.replace(str(temp), str(dest))
    progress.unlink()


def extract(source, out, identity, block_rows):
    import numpy as np
    manifest = out / 'subset.json'
    if manifest.exists():
        proof = json.loads(manifest.read_text())
        if proof['identity_sha256'] != digest(out / 'identity.json'):
            raise RuntimeError('subset identity changed')
        for name, record in proof['files'].items():
            if digest(out / name) != record['sha256']:
                raise RuntimeError('extracted data changed: ' + name)
        return proof
    ids = matrix(out / 'source_ids.bin', '<u4')[:, 0]
    outputs = {}
    for name in NAMES:
        if source_id(source / name) != identity['source'][name]:
            raise RuntimeError('source changed after planning')
        dest = out / name
        selected = ids if name.startswith('base_') else np.arange(identity['source'][name]['rows'], dtype='<u4')
        # Even an already written file is rechecked against its source before reuse.
        if not dest.exists():
            extract_file(source / name, selected, dest, out, block_rows)
        stored = matrix(dest)
        original = matrix(source / name)
        if stored.shape != (len(selected), original.shape[1]):
            raise RuntimeError('extracted matrix shape mismatch')
        for start in range(0, len(selected), block_rows):
            end = min(len(selected), start + block_rows)
            if not np.array_equal(stored[start:end], original[selected[start:end]]):
                raise RuntimeError('extracted row byte mismatch')
        outputs[name] = dict(bytes=dest.stat().st_size, sha256=digest(dest), shape=list(stored.shape))
    proof = dict(status='PASS', identity_sha256=digest(out / 'identity.json'), files=outputs,
                 numpy_version=np.__version__, original_ids_sha256=digest(out / 'source_ids.bin'),
                 row_validation='all selected rows in both spaces; queries unchanged')
    save(manifest, proof)
    return proof


def stable_top(scores, ids, k):
    import numpy as np
    if len(scores) <= k:
        keep = np.arange(len(scores))
    else:
        threshold = np.partition(scores, k - 1)[k - 1]
        lower = np.flatnonzero(scores < threshold)
        equal = np.flatnonzero(scores == threshold)
        equal = equal[np.argsort(ids[equal], kind='mergesort')[:k - len(lower)]]
        keep = np.concatenate((lower, equal))
    return keep[np.lexsort((ids[keep], scores[keep]))]


def distances(queries, values):
    import numpy as np
    qnorm = np.einsum('ij,ij->i', queries, queries)
    bnorm = np.einsum('ij,ij->i', values, values)
    squared = np.maximum(0., qnorm[:, None] + bnorm[None, :] - 2. * queries.dot(values.T))
    # Avoid cancellation changing rankings for exact duplicates / near-zero pairs.
    for row in range(len(queries)):
        near = np.flatnonzero(squared[row] <= 1e-12 * np.maximum(1., qnorm[row] + bnorm))
        if len(near):
            delta = values[near] - queries[row]
            squared[row, near] = np.einsum('ij,ij->i', delta, delta)
    return np.sqrt(squared)


def exact_batch(x, y, qx, qy, weights, k, block_rows):
    import numpy as np
    best_ids = np.empty((len(weights), len(qx), 0), dtype='<u4')
    best_dist = np.empty((len(weights), len(qx), 0), dtype=np.float64)
    for start in range(0, len(x), block_rows):
        end = min(len(x), start + block_rows)
        dx = distances(qx, np.asarray(x[start:end], dtype=np.float64))
        dy = distances(qy, np.asarray(y[start:end], dtype=np.float64))
        block_ids = np.arange(start, end, dtype='<u4')
        next_ids = np.empty((len(weights), len(qx), min(k, end)), dtype='<u4')
        next_dist = np.empty(next_ids.shape, dtype=np.float64)
        for a, weight in enumerate(weights):
            scores = weight * dx + (1. - weight) * dy
            for j in range(len(qx)):
                ids = np.concatenate((best_ids[a, j], block_ids))
                values = np.concatenate((best_dist[a, j], scores[j]))
                keep = stable_top(values, ids, k)
                next_ids[a, j], next_dist[a, j] = ids[keep], values[keep]
        best_ids, best_dist = next_ids, next_dist
    return best_ids, best_dist


def ground_truth(out, identity, query_batch, block_rows):
    import numpy as np
    x, y, qx, qy = (matrix(out / name) for name in NAMES)
    weights = [float(np.float32(a)) for a in WEIGHTS]
    signature = dict(implementation_sha256=digest(__file__), identity_sha256=digest(out / 'identity.json'), subset_sha256=digest(out / 'subset.json'),
                     k=10, objective=OBJECTIVE, effective_alpha=weights, query_batch=query_batch,
                     block_rows=block_rows, algorithm='float64 exhaustive streaming; deterministic (distance, local ID) ties; v1')
    signature_sha = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
    folder = out / 'gt-chunks'
    folder.mkdir(exist_ok=True)
    old = out / 'gt-plan.json'
    if old.exists() and json.loads(old.read_text()) != signature:
        raise RuntimeError('GT configuration changed; preserve old output and use a new namespace')
    save(old, signature)
    started = time.time()
    for first in range(0, len(qx), query_batch):
        end = min(len(qx), first + query_batch)
        path = folder / ('q-{:05d}.npz'.format(first))
        if path.exists():
            with np.load(str(path), allow_pickle=False) as chunk:
                if str(chunk['signature']) != signature_sha or int(chunk['end']) != end:
                    raise RuntimeError('GT chunk identity changed')
                if chunk['ids'].shape != (3, end - first, 10) or chunk['distances'].shape != chunk['ids'].shape:
                    raise RuntimeError('GT chunk shape changed')
        else:
            ids, scores = exact_batch(x, y, np.asarray(qx[first:end], dtype=np.float64),
                                      np.asarray(qy[first:end], dtype=np.float64), weights, 10, block_rows)
            # Independently recompute stored distances by direct subtraction.
            for a, alpha in enumerate(weights):
                direct_x = np.linalg.norm(np.asarray(x[ids[a]], dtype=np.float64) - qx[first:end, None, :], axis=2)
                direct_y = np.linalg.norm(np.asarray(y[ids[a]], dtype=np.float64) - qy[first:end, None, :], axis=2)
                direct = alpha * direct_x + (1. - alpha) * direct_y
                if not np.allclose(direct, scores[a], atol=1e-10, rtol=1e-10):
                    raise RuntimeError('GT distances disagree with direct differences')
            temp = path.with_name(path.name + '.partial')
            with temp.open('wb') as stream:
                np.savez(stream, ids=ids, distances=scores, signature=np.array(signature_sha), end=np.array(end))
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(str(temp), str(path))
        save(out / 'status.json', dict(stage='computing_gt', completed_queries=end, total_queries=len(qx),
                                      elapsed_this_run_seconds=time.time() - started,
                                      updated_unix=time.time(), pid=os.getpid()))
        print('GT queries {}/{}'.format(end, len(qx)), flush=True)
    all_ids = np.empty((3, len(qx), 10), dtype='<u4')
    for first in range(0, len(qx), query_batch):
        with np.load(str(folder / ('q-{:05d}.npz'.format(first))), allow_pickle=False) as chunk:
            all_ids[:, first:int(chunk['end'])] = chunk['ids']
    files = {}
    for a, alpha in enumerate(WEIGHTS):
        name = 'gt-alpha-{}.bin'.format(alpha)
        write_matrix(out / name, all_ids[a], '<u4')
        alpha_name = 'query-alpha-{}.bin'.format(alpha)
        write_matrix(out / alpha_name, np.full((len(qx), 1), alpha, dtype='<f4'), '<f4')
        files[str(alpha)] = dict(gt=name, gt_sha256=digest(out / name), alpha=alpha_name,
                                 alpha_sha256=digest(out / alpha_name), query_rows=len(qx), top_k=10)
    save(out / 'gt.json', dict(status='PASS', protocol=signature, files=files, exhaustive_base_nodes=len(x),
                              ties='distance then ascending subset-local ID', native_query_order=True))


@contextlib.contextmanager
def lock(path, status_root, timeout):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        started = time.monotonic()
        while True:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                save(status_root / 'status.json', dict(stage='waiting_for_global_queue', lock=str(path),
                     elapsed_seconds=time.monotonic() - started, updated_unix=time.time(), pid=os.getpid()))
                if time.monotonic() - started > timeout:
                    raise RuntimeError('bounded global queue wait expired; resume later')
                time.sleep(30)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', type=Path, required=True)
    parser.add_argument('--count', type=int, default=2000000)
    parser.add_argument('--seed', type=int, default=20261113)
    parser.add_argument('--threads', type=int, default=2, choices=(1, 2))
    parser.add_argument('--query-batch', type=int, default=16)
    parser.add_argument('--block-rows', type=int, default=16384)
    parser.add_argument('--queue-lock', type=Path, required=True)
    parser.add_argument('--lock-timeout', type=int, default=21600)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.query_batch < 1 or args.block_rows < 1:
        parser.error('batch sizes must be positive')
    for name in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT'):
        os.environ[name] = str(args.threads)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        with lock(args.queue_lock, args.out, args.lock_timeout):
            identity = plan(args.source, args.out, args.split, args.count, args.seed)
            if args.plan_only:
                save(args.out / 'status.json', dict(stage='planned', updated_unix=time.time()))
                return
            if not (args.out / 'subset.json').exists() and shutil.disk_usage(str(args.out)).free < 2 * args.count * (identity['source'][NAMES[0]]['dims'] + identity['source'][NAMES[1]]['dims']) * 4 + (2 << 30):
                raise RuntimeError('insufficient disk headroom for independent subset output')
            extract(args.source, args.out, identity, args.block_rows)
            ground_truth(args.out, identity, args.query_batch, args.block_rows)
            save(args.out / 'status.json', dict(stage='complete', updated_unix=time.time(), pid=os.getpid(),
                 subset_nodes=args.count, alpha=list(WEIGHTS), gt_top_k=10, graphs_built=False))
    except Exception as error:
        save(args.out / 'status.json', dict(stage='failed', error=str(error), updated_unix=time.time(), pid=os.getpid()))
        raise


if __name__ == '__main__':
    main()
