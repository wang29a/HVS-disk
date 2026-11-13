"""Eight-query functional checks on old assets, never performance evidence.

The old graph is deliberately reused only to exercise query paths. New
scientific comparisons must wait for the new, metric-consistent source graph.
"""
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import numpy as np
from audit_objective import matrix, DATA

ROOT = Path(__file__).resolve().parent.parent
COORD = Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
OLD = ROOT.parent/'alpha-page-factorial-20261110'


def main():
    scratch = ROOT/'scratch/engine-preflight'
    scratch.mkdir(exist_ok=True)
    sources = dict(full=(Path('/mnt/nvme3/wz/Index/Openimage/_disk.index'),
                         Path('/mnt/nvme3/wz/Index/Openimage/disk_index_graph')),
                   compact=(OLD/'scratch/openimage-b0-convert/meta.bin',
                            OLD/'scratch/openimage-b0-convert/topo.bin'))
    for kind, (meta, topo) in sources.items():
        folder = scratch/kind
        folder.mkdir(exist_ok=True)
        for name, source in [('group-1.bin', topo), ('group-6.bin', COORD)]:
            path = folder/name
            if not path.exists():
                path.symlink_to(source)
    files = dict(query1=DATA/'query_img_emb.fvecs', query2=DATA/'query_text_emb.fvecs',
                 alpha=DATA/'base_gt_round_00/query_alpha.fvecs',
                 gt=DATA/'base_gt_round_00/top10_results.ivecs')
    for name, source in files.items():
        values = matrix(source, np.uint32 if name == 'gt' else np.float32)
        with (scratch/(name+'.bin')).open('wb') as out:
            out.write(struct.pack('<II', 8, values.shape[1]))
            np.asarray(values[:8]).tofile(out)
    cases = [('on-DRAM', 'full', True, False, False, False),
             ('on-SSD', 'full', False, False, False, False),
             ('off-DRAM', 'full', True, True, False, False),
             ('compact-DRAM', 'compact', True, False, False, False),
             ('exact-on', 'full', True, False, True, False),
             ('exact-off', 'full', True, True, True, False),
             ('concat-DRAM', 'compact', True, False, False, True),
             ('concat-SSD', 'compact', False, False, False, True)]
    for name, kind, dram, bypass, exact, concat in cases:
        env = {k:v for k, v in os.environ.items() if not k.startswith(('LAYOUT_', 'METRIC_'))}
        env.update(OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
                   LAYOUT_ONE_PASS='1', LAYOUT_TRACE=str(scratch/name), LAYOUT_DUMP=str(scratch/name),
                   METRIC_RERANK='20')
        if dram:
            env['LAYOUT_DRAM'] = '1'
        if bypass:
            env['LAYOUT_BYPASS_ALPHA'] = '1'
        if exact:
            env['METRIC_NAV_EXACT'] = '1'
        if concat:
            env['METRIC_CONCAT'] = '1'
        meta, topo = sources[kind]
        args = [ROOT/'bin/metric-search', 'endpoint' if concat else 'query', meta, topo, COORD,
                scratch/kind, 'T_XY', scratch/(name+'.json'),
                '/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/',
                scratch/'query1.bin', scratch/'query2.bin', scratch/'alpha.bin', scratch/'gt.bin', 2, 0, 40]
        subprocess.check_call(list(map(str, args)), env=env)
        point = json.loads((scratch/(name+'.json')).read_text())[0]
        if exact and any(point[k] for k in ('pq_space_evals', 'reranked_nodes', 'coordinate_submitted_pages')):
            raise RuntimeError('exact-navigation arm ran PQ or rerank')
        if concat and point['recall'] != -1:
            raise RuntimeError('concat functional smoke must not use weighted GT')
    for first, second in [('on-DRAM', 'on-SSD'), ('off-DRAM', 'compact-DRAM'), ('concat-DRAM', 'concat-SSD')]:
        for suffix in ('', '.ids', '.dists'):
            a = scratch/(first+'.L40'+suffix)
            b = scratch/(second+'.L40'+suffix)
            if a.read_bytes() != b.read_bytes():
                raise RuntimeError('functional path mismatch: '+first+' vs '+second)
    qx, qy = matrix(scratch/'query1.bin'), matrix(scratch/'query2.bin')
    alpha = matrix(scratch/'alpha.bin')[:, 0]
    coords = np.memmap(str(COORD), mode='r', dtype=np.float32, shape=(456699, 2048))
    error = 0.
    for name, kind, dram, bypass, exact, concat in cases:
        ids = np.fromfile(str(scratch/(name+'.L40.ids')), dtype=np.uint32).reshape(8, 10)
        actual = np.fromfile(str(scratch/(name+'.L40.dists')), dtype=np.float32).reshape(8, 10)
        for q in range(8):
            vectors = np.asarray(coords[ids[q]], dtype=np.float64)
            a = np.sum((vectors[:, :768]-qx[q])**2, axis=1)
            b = np.sum((vectors[:, 768:1536]-qy[q])**2, axis=1)
            expected = np.sqrt(a+b) if concat else alpha[q]*np.sqrt(a)+(1-float(alpha[q]))*np.sqrt(b)
            error = max(error, float(np.max(np.abs(actual[q]-expected)/np.maximum(1e-8, np.abs(expected)))))
    if error > 5e-6:
        raise RuntimeError('functional returned score mismatch '+str(error))
    engine_hash = hashlib.sha256((ROOT/'bin/metric-search').read_bytes()).hexdigest()
    result = dict(status='PASS', queries=8, L=40, r=20, threads=2, cases=len(cases),
        source='historical graph only for query-path correctness',
        performance_evidence=False, gt_semantics_audit=False,
        dram_ssd_parity=True, full_bypass_compact_parity=True,
        exact_navigation_without_pq_or_rerank=True, concat_128_chunk_parity=True,
        independent_score_max_relative_error=error, engine_sha256=engine_hash)
    target = ROOT/'analysis/engine-smoke.json'
    tmp = target.with_suffix('.partial')
    tmp.write_text(json.dumps(result, indent=2)+'\n')
    os.replace(str(tmp), str(target))


if __name__ == '__main__':
    main()
