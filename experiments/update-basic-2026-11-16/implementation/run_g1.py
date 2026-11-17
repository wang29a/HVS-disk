"""Node3 serialized structural gate. First failure stops all dependent work."""
from pathlib import Path
import fcntl
import json
import os
import shutil
import struct
import time
import run_g0 as c


def matrix(path, dtype='<f4'):
    import numpy as np
    with Path(path).open('rb') as f:
        n,d = struct.unpack('<II',f.read(8))
    return np.memmap(str(path),dtype=dtype,mode='r',offset=8,shape=(n,d))


def fixture(mode):
    import numpy as np
    root = c.RUN/mode
    root.mkdir(exist_ok=True)
    if (root/'fixture-identity.json').exists():
        raise RuntimeError('existing mutable G1 fixture: use a new run namespace')
    ab = c.HOST_ROOT/'experiments/backlink-ab-20261115/runtime'
    src = ab/'statistical-v2'
    pq = c.HOST_ROOT/'experiments/multigraph-20261113/runtime/data'
    source_files = [src/'packed/B/compact.meta',src/'packed/B/compact.topo',src/'xy.pages']
    for source,name in zip(source_files,('index.meta','graph.pages','coordinates.pages')):
        shutil.copyfile(source,root/name)
    ids = np.fromfile(str(ab/'data-10k/subset_ids.u32'),dtype='<u4')
    assert len(ids) == 10000
    for space in ('emb','loc'):
        shutil.copyfile(pq/(space+'_pq_pivots.bin'),root/(space+'_pq_pivots.bin'))
        codes = matrix(pq/(space+'_pq_compressed.bin'),'u1')
        selected = np.asarray(codes[ids],dtype='u1')
        (root/(space+'_pq_compressed.bin')).write_bytes(struct.pack('<II',*selected.shape)+selected.tobytes())
    # Reconstruct the exact build-time arithmetic: sequential float32 sum by
    # input ID, then float32 division, matching HybridIndex::link's centers.
    centers = []
    for filename in ('base_img_emb.fvecs','base_text_emb.fvecs'):
        a = matrix(ab/'data-10k'/filename)
        value = np.zeros(a.shape[1],dtype='<f4')
        for row in a:
            np.add(value,row,out=value)
        value /= np.float32(len(a))
        centers.append(value)
    centers = np.concatenate(centers).astype('<f4')
    (root/'centers.bin').write_bytes(struct.pack('<II',1,len(centers))+centers.tobytes())
    source_data = Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')
    rng = np.random.RandomState(20261116)
    candidates = rng.choice(2000000,1200,replace=False)
    existing = set(map(int,ids))
    fresh = np.array([int(v) for v in candidates if int(v) not in existing][:1000],dtype='<u4')
    assert len(fresh) == 1000
    (root/'fresh-source-ids.u32').write_bytes(fresh.tobytes())
    for name,filename in [('x','base_img_emb.fvecs'),('y','base_text_emb.fvecs')]:
        a = matrix(source_data/filename)
        selected = np.asarray(a[fresh],dtype='<f4')
        (root/('insert-'+name+'.bin')).write_bytes(struct.pack('<II',*selected.shape)+selected.tobytes())
    identity = dict(initial=10000,max_points=15000,source_graph='corrected DEG B',
        reprune_dist=mode,reverse_policy='append below R, prune only at R',
        L_ins=200,C_ins=100,R=40,consolidate_ratio=.05,seed=20261116,
        center_method='builder sequential float32 sum in initial subset ID order, then float32 division; immutable',
        source_files={str(p):c.sha(p) for p in source_files},
        initial_files={p.name:c.sha(p) for p in root.iterdir() if p.is_file()})
    (root/'fixture-identity.json').write_text(json.dumps(identity,indent=2)+'\n')
    return root


def main():
    c.RUN = c.ROOT/'runtime'/os.environ.get('HVS_UPDATE_G1_RUN','g1-in-degree-protection')
    c.RUN.mkdir(parents=True,exist_ok=True)
    lock = (c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:
        assert json.loads((c.ROOT/'runtime/prune-gate/status.json').read_text())['stage'] == 'PRUNE_SEMANTICS_PASSED'
        c.save('status.json',dict(stage='compiling',updated_unix=time.time()))
        src = c.ROOT/'implementation'
        flags = json.loads((c.ROOT/'runtime/g0-attempt2/build-identity.json').read_text())['flags']
        c.command(['clang-format','-style=LLVM','-i',src/'in_degree_guard.h',src/'in_degree_guard_check.cpp'],'format-in-degree-guard')
        c.command(['g++-11']+flags+[src/'in_degree_guard_check.cpp','-pthread','-o',c.RUN/'in_degree_guard_check'],'compile-in-degree-guard',300)
        c.command([c.RUN/'in_degree_guard_check'],'in-degree-guard-directed')
        c.command(['clang-format','-style=LLVM','-i',src/'entry_frontier.h',src/'entry_frontier_check.cpp'],'format-entry-parity')
        c.command(['g++-11']+flags+[src/'entry_frontier_check.cpp','-o',c.RUN/'entry_frontier_check'],'compile-entry-parity',300)
        c.command([c.RUN/'entry_frontier_check'],'entry-parity')
        c.save('entry-parity-identity.json',dict(header=c.sha(src/'entry_frontier.h'),test=c.sha(src/'entry_frontier_check.cpp'),
            source=c.sha(c.HOST_ROOT/'experiments/backlink-ab-20261115/runtime/build/hybrid_index.cpp'),
            reference='UpdateEnterpointSet: reverse lexicographic (space2,space1,ID), strictly increasing space1; zero sentinel'))
        cpp = [src/'update_store.cpp',src/'deg_prune.cpp',src/'dual_update.cpp',src/'update_check.cpp']
        c.command(['clang-format','-style=LLVM','-i',src/'update_store.h',src/'dual_update.h']+cpp,'format-executor')
        command = ['g++-11']+flags+cpp+[c.BUILD/'src/libpipeann.a',
            '-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',c.RUN/'update_check']
        c.command(command,'compile-executor',300)
        c.save('build-identity.json',dict(flags=flags,link=list(map(str,command)),
            sources={p.name:c.sha(p) for p in cpp+[src/'dual_update.h',src/'update_store.h',src/'deg_prune.h',src/'entry_frontier.h',src/'in_degree_guard.h']},
            binary=c.sha(c.RUN/'update_check'),upstream_library=c.sha(c.BUILD/'src/libpipeann.a')))
        for mode in ('pq','exact'):
            c.save('status.json',dict(stage='running_G1',reprune_dist=mode,updated_unix=time.time()))
            root = fixture(mode)
            c.command([c.RUN/'update_check',root,mode,1000],'G1-'+mode,3600)
            if 'G1_PASS operations=1000' not in (c.RUN/('G1-'+mode+'.log')).read_text():
                raise RuntimeError('G1 completion marker absent')
        c.save('status.json',dict(stage='G1_PASSED_BOTH_DISTANCES',updated_unix=time.time(),
            G2='NOT_RUN',G3='NOT_RUN',G4='requires G2 in both modes'))
    except Exception as e:
        c.save('status.json',dict(stage='STOPPED_AT_G1',reason=str(e),updated_unix=time.time(),
            dependent_experiments='not started'))
        raise


if __name__ == '__main__':
    main()
