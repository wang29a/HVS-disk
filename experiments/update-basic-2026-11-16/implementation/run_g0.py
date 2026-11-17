"""Node3 only: serialized G0, bounded small fixture, no production mutation."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import shlex
import subprocess
import time

ROOT = Path(__file__).resolve().parent.parent
HOST_ROOT = Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
SOURCE = HOST_ROOT / 'source'
BUILD = HOST_ROOT / 'build-separated'
RUN = ROOT / 'runtime' / os.environ.get('HVS_UPDATE_G0_RUN', 'g0')
ENV = dict(os.environ, LC_ALL='C', OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2',
           OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def save(name, data):
    p = RUN / name
    p.write_text(json.dumps(data, indent=2) + '\n')


def command(args, label, timeout=180):
    args = [str(x) for x in args]
    wrapped = ['taskset', '-c', '76,77', 'numactl', '--membind=3',
               'nice', '-n', '15', 'ionice', '-c', '2', '-n', '7'] + args
    start = time.time()
    with (RUN / (label + '.log')).open('w') as log:
        p = subprocess.run(wrapped, stdout=log, stderr=subprocess.STDOUT,
                           env=ENV, timeout=timeout)
    record = dict(command=wrapped, returncode=p.returncode, elapsed_s=time.time()-start,
                  environment={k: ENV[k] for k in ('LC_ALL', 'OMP_NUM_THREADS', 'OMP_THREAD_LIMIT', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'LD_LIBRARY_PATH')})
    save(label + '.json', record)
    if p.returncode:
        raise RuntimeError(label + ' failed: ' + str(p.returncode))


def fixture():
    import numpy as np
    import struct
    d = RUN / 'fixture'
    d.mkdir(exist_ok=True)
    rng = np.random.default_rng(20261116)
    x = rng.normal(size=(256, 8)).astype('<f4')
    def matrix(p, a):
        a = np.ascontiguousarray(a)
        return struct.pack('<II', *a.shape) + a.tobytes()
    def write_matrix(name, a):
        (d/name).write_bytes(matrix(d/name, a))
    write_matrix('all.bin', x[:136])
    q = x[128:136].copy()
    write_matrix('queries.bin', q)
    # Exact one-chunk PQ for this IO fixture only, never a recall/quality study.
    parts = [matrix(None, x), matrix(None, np.zeros((8,1), '<f4')),
             matrix(None, np.arange(8, dtype='<u4').reshape(-1,1)),
             matrix(None, np.array([0,8], '<u4').reshape(-1,1))]
    offsets = []
    off = 48
    for b in parts:
        offsets.append(off)
        off += len(b)
    offsets.append(off)
    pivot = struct.pack('<II5Q', 5, 1, *offsets) + b''.join(parts)
    prefix = d/'small'
    for name in ('emb', 'loc'):
        Path(str(prefix)+name+'_pq_pivots.bin').write_bytes(pivot)
        Path(str(prefix)+name+'_pq_compressed.bin').write_bytes(
            matrix(None, np.arange(128, dtype=np.uint8).reshape(-1,1)))
    Path(str(prefix)+'_pq_pivots_2.bin').write_bytes(pivot)
    Path(str(prefix)+'_pq_compressed_2.bin').write_bytes(
        matrix(None, np.arange(128, dtype=np.uint8).reshape(-1,1)))
    Path(str(prefix)+'_map.bin').write_bytes(matrix(None, np.zeros((128,1), np.uint8)))
    metadata = struct.pack('<IIIIIQI', 128, 8, 8, 40, 0, 8192//164, 1) + struct.pack('<I',0)
    Path(str(prefix)+'_disk.index').write_bytes(metadata.ljust(8192, b'\0'))
    # Both physical files have initialized spare pages: old insert reads first.
    topology = bytearray(8192*5)
    for i in range(128):
        dist = ((x[:128].astype('float64')-x[i])**2).sum(1)
        ids = np.argsort(dist, kind='stable')[1:41].astype('<u4')
        off = i//49*8192+i%49*164
        topology[off:off+164] = struct.pack('<I',40)+ids.tobytes()
    (d/'disk_index_graph').write_bytes(topology)
    coords = bytearray(8192*3)
    for i in range(128):
        off = i//128*8192+i%128*64
        coords[off:off+32] = x[i].tobytes()
    (d/'disk_index_data').write_bytes(coords)
    truth = d/'truth'
    truth.mkdir(exist_ok=True)
    for total, label in ((128,0), (136,8)):
        dist = ((q.astype('float64')[:,None,:]-x[None,:total,:])**2).sum(2)
        ids = np.argsort(dist, axis=1, kind='stable')[:,:10].astype('<u4')
        values = np.take_along_axis(dist, ids, axis=1).astype('<f4')
        (truth/('gt_'+str(label)+'.bin')).write_bytes(
            struct.pack('<II',8,10)+ids.tobytes()+values.tobytes())
    save('fixture.json', dict(seed=20261116, initial=128, insertions=8, dim=8,
         unused_second_coordinate_space='8 zero floats per record for dual-format loader compatibility',
         purpose='single-space upstream IO smoke, not DEG or ANN quality evidence',
         files={str(p.relative_to(d)):sha(p) for p in d.rglob('*') if p.is_file()}))
    return d, prefix


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    lock = (HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        save('status.json', dict(stage='waiting_global_lock', updated_unix=time.time()))
        return 75
    save('status.json', dict(stage='building_g0', pid=os.getpid(), updated_unix=time.time()))
    try:
        commands = json.loads((BUILD/'compile_commands.json').read_text())
        cfg = next(x for x in commands if x['file'].endswith('/test_insert_search.cpp'))
        flags = shlex.split(cfg['command'])[1:]
        flags = flags[:flags.index('-o')]
        flags += ['-I'+str(ROOT/'implementation'), '-fno-fast-math', '-ffp-contract=off']
        lib = BUILD/'src/libpipeann.a'
        link = ['-L/mnt/nvme2/wz/disk/local/openblas/lib', '-laio', '-ljemalloc', '-lopenblas']
        cpp = ROOT/'implementation'
        command(['g++-11']+flags+[cpp/'update_store.cpp', cpp/'store_smoke.cpp', lib]+link+['-pthread','-o', RUN/'store_smoke'], 'compile-store')
        command([RUN/'store_smoke',RUN], 'store-smoke', 30)
        # Upstream driver has no strategy CLI argument, and strategy=0 never
        # opens separated fds. The single-space router also unconditionally
        # reads PQ2/map. Strategy=5 loads both; all map bytes are zero so the
        # fixture still uses only its one single-space PQ. Both codebooks match.
        # Driver and library source/default behavior remain untouched.
        shim = RUN/'test-include/v2'
        shim.mkdir(parents=True, exist_ok=True)
        original_header = SOURCE/'include/v2/dynamic_index.h'
        header = original_header.read_text()
        needle = 'int strategy = 0'
        if header.count(needle) != 1:
            raise RuntimeError('unexpected upstream constructor declaration')
        (shim/'dynamic_index.h').write_text(header.replace(needle,'int strategy = 5'))
        # Make the copied header's relative include resolvable.
        (shim/'journal.h').symlink_to(SOURCE/'include/v2/journal.h') if not (shim/'journal.h').exists() else None
        test_flags = ['-I'+str(shim.parent)] + flags
        test_source = SOURCE/'tests/test_insert_search.cpp'
        command(['g++-11']+test_flags+[test_source,lib]+link+['-pthread','-o',RUN/'test_insert_search'], 'compile-upstream-test')
        d,prefix = fixture()
        command([RUN/'test_insert_search','float',d/'all.bin',64,8,1,1,1,3,prefix,
                 d/'queries.bin',d/'truth',0,10,1,1,0,64], 'upstream-insert-search', 90)
        log = (RUN/'upstream-insert-search.log').read_text()
        if 'Inserted 8 points' not in log or 'Insertions complete!' not in log:
            raise RuntimeError('upstream test did not confirm completed insertions')
        save('build-identity.json', dict(flags=flags, test_flags=test_flags, link=link,
             source_sha256={str(p):sha(p) for p in [cpp/'update_store.h',cpp/'update_store.cpp',cpp/'store_smoke.cpp',test_source,original_header,shim/'dynamic_index.h']},
             base_library_sha256=sha(lib), binaries={str(p.name):sha(p) for p in [RUN/'store_smoke',RUN/'test_insert_search']},
             test_configuration='unmodified upstream test source, test-local constructor default strategy=5; single-space identical PQ2 and all-zero map; production header untouched'))
        save('status.json', dict(stage='G0_PASSED', updated_unix=time.time()))
        return 0
    except Exception as e:
        save('status.json', dict(stage='G0_STOPPED', reason=str(e), updated_unix=time.time()))
        raise


if __name__ == '__main__':
    raise SystemExit(main())
