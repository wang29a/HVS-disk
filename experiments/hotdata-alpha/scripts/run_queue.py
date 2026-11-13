#!/usr/bin/env python3
"""Serial low-core diagnostic; supplied GT only; unchanged search library."""
import collections
import datetime
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import statistics
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
OLD = BUILD / 'experiments/concurrency-alpha'
INDEX = BUILD / 'experiments/laion'
DATA = Path('/mnt/nvme2/wz/disk/dataset/Laion')
MEM = Path('/dev/shm/hvs-hotdata-alpha-20261028-wz')
CPUS = '32,35,39,41'
COMMON_L = [20, 40, 50, 60, 70, 80, 100, 120, 140, 160, 200, 300, 400]
BOUNDARY_L = [50, 60, 70, 100, 120, 140, 160]
ENV = dict(os.environ, LC_ALL='C', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OMP_THREAD_LIMIT='4', OMP_PROC_BIND='close',
           OMP_PLACES='{32},{35},{39},{41}', HVS_POINT_SECONDS='10',
           HVS_MEMORY_DIR=str(MEM))
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')
ENV.pop('HVS_ONE_PASS', None)
ENV.pop('HVS_DUMP_PREFIX', None)
OWN_QUEUE = False


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def log(msg):
    print(now(), msg, flush=True)


def write_json(p, value):
    p = Path(p)
    temp = p.with_name(p.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    os.replace(str(temp), str(p))


def sha(p):
    h = hashlib.sha256()
    with open(str(p), 'rb') as f:
        while True:
            b = f.read(4 << 20)
            if not b:
                return h.hexdigest()
            h.update(b)


def status(**kw):
    write_json(ROOT / 'status.json', dict(updated_at=now(), **kw))


def limited_process():
    resource.setrlimit(resource.RLIMIT_AS, (80 << 30, 80 << 30))


def command(scheme, threads, ls, original=False):
    binary = OLD / 'bin/steady-separated' if original else ROOT / 'bin/hot-search'
    return ['numactl', '--physcpubind=' + CPUS, '--interleave=0,1', str(binary),
            'float', str(INDEX / scheme) + '/', str(threads), '4',
            str(DATA / 'query_img_emb.fvecs'), str(DATA / 'query_text_emb.fvecs'),
            str(DATA / 'alpha7.bin'),
            str(DATA / 'gt/base_gt_round_00/top10_results_alpha_0.7.ivecs'),
            '10', 'l2', '6', '0', '17'] + list(map(str, ls))


def points(path):
    text = Path(path).read_text()
    if 'HVS_STEADY_COMPLETE' not in text:
        raise RuntimeError('Search did not finish: ' + str(path))
    return [json.loads(line[len('HVS_POINT '):]) for line in text.splitlines()
            if line.startswith('HVS_POINT ')]


def build_driver():
    source = BUILD / 'source'
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=str(source)).decode().strip()
    if commit != 'd2a39e2bd50e6d03716c4268f7b90f931e5a6a1e':
        raise RuntimeError('Unexpected source commit')
    if subprocess.check_output(['git', 'status', '--porcelain'], cwd=str(source)).strip():
        raise RuntimeError('Source checkout is dirty')
    spec = json.loads((OLD / 'analysis/build.json').read_text())['separated']
    if sha(spec['library']) != spec['library_sha256']:
        raise RuntimeError('Search library changed')
    # The server clang-format is older than the repository config. Use its
    # supported LLVM preset for these two new standalone diagnostic sources.
    subprocess.run(['clang-format', '-i', '-style=LLVM',
                    str(ROOT / 'scripts/hot_search.cpp'), str(ROOT / 'scripts/hot_reader.h')], check=True)
    cmd = ['g++-11'] + spec['flags'] + [str(ROOT / 'scripts/hot_search.cpp'),
        spec['library'], '-L/mnt/nvme2/wz/disk/local/openblas/lib',
        '-laio', '-ljemalloc', '-lopenblas', '-o', str(ROOT / 'bin/hot-search')]
    with open(str(ROOT / 'logs/build.log'), 'w') as f:
        subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
    ident = dict(commit=commit, library=spec['library'], library_sha256=spec['library_sha256'],
                 binary_sha256=sha(ROOT / 'bin/hot-search'), command=cmd,
                 driver_sha256=sha(ROOT / 'scripts/hot_search.cpp'),
                 reader_sha256=sha(ROOT / 'scripts/hot_reader.h'))
    write_json(ROOT / 'analysis/build.json', ident)
    return ident


def stage_images():
    inputs = {'coord.bin': INDEX / 'merge/disk_index_data',
              'merge-topo.bin': INDEX / 'merge/disk_index_graph',
              'no-alpha-topo.bin': INDEX / 'no-alpha/disk_index_graph'}
    if os.path.realpath(str(inputs['coord.bin'])) != os.path.realpath(str(INDEX / 'no-alpha/disk_index_data')):
        raise RuntimeError('Schemes do not share the same coordinate source')
    total = sum(p.stat().st_size for p in inputs.values())
    if total > 64 << 30 or shutil.disk_usage(str(MEM.parent)).free < total + (8 << 30):
        raise RuntimeError('Memory image budget or tmpfs space insufficient')
    available = int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines()
                         if x.startswith('MemAvailable:'))) * 1024
    if available < total + (64 << 30):
        raise RuntimeError('Not enough memory headroom')
    MEM.mkdir(exist_ok=True)
    records = []
    for name, source in inputs.items():
        status(stage='staging', file=name, bytes=source.stat().st_size)
        target = MEM / name
        prior = ROOT / 'analysis' / (name + '.json')
        if target.exists() and prior.exists():
            old = json.loads(prior.read_text())
            if target.stat().st_size == old['bytes'] and sha(target) == old['sha256']:
                records.append(old)
                continue
        partial = MEM / (name + '.partial')
        h = hashlib.sha256()
        start, copied = time.monotonic(), 0
        last_log = start
        with open(str(source), 'rb') as src, open(str(partial), 'wb') as dst:
            while True:
                block = src.read(4 << 20)
                if not block:
                    break
                dst.write(block)
                h.update(block)
                copied += len(block)
                # One sequential reader, no more than 128 MiB/s.
                delay = copied / float(128 << 20) - (time.monotonic() - start)
                if delay > 0:
                    time.sleep(delay)
                if time.monotonic() - last_log >= 30:
                    log('STAGING {} {:.1f}%'.format(name, copied * 100 / source.stat().st_size))
                    last_log = time.monotonic()
        os.replace(str(partial), str(target))
        # Verify the copy by a separate tmpfs read before use.
        if sha(target) != h.hexdigest():
            raise RuntimeError('Memory image hash mismatch')
        record = dict(name=name, source=os.path.realpath(str(source)),
                      bytes=copied, sha256=h.hexdigest(), seconds=time.monotonic()-start)
        write_json(prior, record)
        records.append(record)
        log('STAGED ' + name)
    write_json(ROOT / 'analysis/memory-images.json', records)
    return records


def run_logged(label, cmd, env, timed=False):
    path = ROOT / 'logs' / (label + '.log')
    with open(str(path) + '.running', 'w') as f:
        started = time.time()
        proc = subprocess.Popen(cmd, env=env, stdout=f, stderr=subprocess.STDOUT,
                                preexec_fn=limited_process)
        monitor = None
        with open(str(ROOT / 'logs' / (label + '.pidstat')), 'w') as out:
            if timed:
                monitor = subprocess.Popen(['pidstat', '-h', '-u', '-r', '-w', '-p', str(proc.pid), '1'],
                                           stdout=out, stderr=subprocess.STDOUT)
            code = proc.wait()
            if monitor:
                monitor.terminate()
                monitor.wait()
        if code:
            raise RuntimeError('Search failed: {} exit {}'.format(label, code))
    os.replace(str(path) + '.running', str(path))
    return points(path), started, time.time(), path


def validate():
    reports = []
    for scheme in ('merge', 'no-alpha'):
        ref = None
        for backend in ('original', 'ssd', 'all-hot', 'coord-hot'):
            label = 'parity-' + scheme + '-' + backend
            dump = ROOT / 'analysis' / label
            env = dict(ENV, HVS_ONE_PASS='1', HVS_SCHEME=scheme,
                       HVS_BACKEND='ssd' if backend == 'original' else backend)
            if backend != 'original':
                env['HVS_DUMP_PREFIX'] = str(dump)
            actual, _, _, _ = run_logged(label, command(scheme, 1, [40, 100, 200], backend == 'original'), env)
            if ref is None:
                ref = actual
                continue
            for a, b in zip(ref, actual):
                for metric in ('recall', 'hops', 'comparisons', 'topology_ios', 'rerank_ios'):
                    if abs(a[metric] - b[metric]) > 1e-6:
                        raise RuntimeError('Parity mismatch: {} {}'.format(label, metric))
                if backend != 'ssd':
                    for suffix in ('ids', 'dists'):
                        first = ROOT / 'analysis' / ('parity-' + scheme + '-ssd.L{}.'.format(a['L']) + suffix)
                        second = Path(str(dump) + '.L{}.'.format(a['L']) + suffix)
                        if sha(first) != sha(second):
                            raise RuntimeError('Result bytes differ: ' + label)
            check_backend(actual, backend)
            reports.append(dict(scheme=scheme, backend=backend, status='PASS'))
            log('PARITY_PASS ' + label)
    write_json(ROOT / 'analysis/validation.json', dict(status='PASS', checks=reports,
        note='Original library driver counters equal; IDs and exact-distance bytes equal across SSD/all-hot/coord-hot.'))


def check_backend(rows, backend):
    for p in rows:
        mem_t, mem_c = p['memory_topology_requests'], p['memory_coordinate_requests']
        ssd_t, ssd_c = p['ssd_topology_requests'], p['ssd_coordinate_requests']
        expected = dict(ssd=(0, 0, p['topology_ios'], p['rerank_ios']),
                        **{'coord-hot': (0, p['rerank_ios'], p['topology_ios'], 0),
                           'all-hot': (p['topology_ios'], p['rerank_ios'], 0, 0)})[backend]
        if any(abs(a-b) > 1e-6 for a, b in zip((mem_t, mem_c, ssd_t, ssd_c), expected)):
            raise RuntimeError('Backend request accounting mismatch')
        if p['max_rss_kib'] > 64 << 20:
            raise RuntimeError('RSS budget exceeded')


def summarize():
    groups = collections.defaultdict(list)
    runs = [json.loads(p.read_text()) for p in (ROOT / 'results').glob('*.json')]
    for r in runs:
        for p in r['points']:
            groups[r['backend'], r['threads'], r['scheme'], p['L']].append(p)
    means = []
    metrics = ('qps', 'recall', 'outer_latency_us', 'outer_p50_us', 'outer_p95_us', 'outer_p99_us',
               'comparisons', 'topology_ios', 'rerank_ios', 'cpu_us', 'io_us',
               'memory_topology_requests', 'memory_coordinate_requests',
               'ssd_topology_requests', 'ssd_coordinate_requests', 'max_rss_kib')
    for key, samples in sorted(groups.items()):
        row = dict(zip(('backend', 'threads', 'scheme', 'L'), key))
        row['repeats'] = len(samples)
        for k in metrics:
            row[k] = statistics.mean(p[k] for p in samples)
            row[k + '_stddev'] = statistics.stdev(p[k] for p in samples) if len(samples) > 1 else None
        means.append(row)
    curves = collections.defaultdict(list)
    for r in means:
        if r['repeats'] == 3:
            curves[r['backend'], r['threads'], r['scheme']].append(r)
    matched = []
    for key, curve in sorted(curves.items()):
        required = COMMON_L if key[0] == 'all-hot' else BOUNDARY_L
        if len(curve) != len(required):
            continue
        curve.sort(key=lambda r:r['recall'])
        for target in (90, 95):
            for a, b in zip(curve, curve[1:]):
                if a['recall'] <= target <= b['recall']:
                    f = (target-a['recall']) / (b['recall']-a['recall']) if b['recall'] != a['recall'] else 0
                    row = dict(zip(('backend', 'threads', 'scheme'), key))
                    row['target_recall'] = target
                    for k in ('L', 'qps', 'outer_latency_us', 'comparisons', 'topology_ios', 'rerank_ios'):
                        row[k] = a[k]+f*(b[k]-a[k])
                    matched.append(row)
                    break
    write_json(ROOT / 'analysis/summary.json', dict(completed_runs=len(runs), mean_curves=means,
        matched_recall=matched, note='Hot-data mechanism diagnostic. Interpolation only within measured recall.'))
    return len(runs)


def main():
    global OWN_QUEUE
    for d in ('bin', 'logs', 'results', 'analysis'):
        (ROOT / d).mkdir(exist_ok=True)
    lock = open(str(ROOT / 'queue.lock'), 'w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    OWN_QUEUE = True
    prior_status = ROOT / 'status.json'
    if prior_status.exists() and json.loads(prior_status.read_text()).get('status') == 'complete':
        log('ALREADY_COMPLETE')
        return
    if not set(os.sched_getaffinity(0)).issubset({32,35,39,41}):
        raise RuntimeError('Launch must be confined to the four documented CPUs')
    status(stage='build', status='running')
    build = build_driver()
    files = [DATA / 'query_img_emb.fvecs', DATA / 'query_text_emb.fvecs', DATA / 'alpha7.bin',
             DATA / 'gt/base_gt_round_00/top10_results_alpha_0.7.ivecs']
    for source in files:
        if not source.is_file():
            raise RuntimeError('Missing supplied input: ' + str(source))
    import struct
    raw = (DATA / 'alpha7.bin').read_bytes()
    n, dim = struct.unpack('<II', raw[:8])
    if n != 1000 or dim != 1 or any(abs(v-.7)>1e-6 for v in struct.unpack('<1000f', raw[8:])):
        raise RuntimeError('Expected supplied 1000-query alpha=0.7')
    images = stage_images()
    manifest = dict(commit=build['commit'], library_sha256=build['library_sha256'],
        driver_sha256=build['driver_sha256'], reader_sha256=build['reader_sha256'],
        inputs={str(p):sha(p) for p in files}, memory_images=images,
        cpus=CPUS, numa_memory=[0,1], max_threads=4, search_threads=[1,4],
        max_memory_image_gib=64, max_rss_gib=64, address_space_gib=80,
        seconds=10, repetitions=3, common_L=COMMON_L, boundary_L=BOUNDARY_L,
        alpha=.7, supplied_GT_only=True)
    previous = ROOT / 'analysis/manifest.json'
    if previous.exists():
        prior = json.loads(previous.read_text())
        # Staging time is not part of input identity on a resumed run.
        for doc in (manifest, prior):
            for image in doc['memory_images']:
                image.pop('seconds', None)
        if prior != manifest:
            raise RuntimeError('Resume manifest differs')
    write_json(previous, manifest)
    status(stage='validation', status='running')
    validate()
    tasks=[]
    for backend in ('all-hot', 'coord-hot', 'ssd'):
        for rep in (1,2,3):
            for t in ((1,4) if rep % 2 else (4,1)):
                for scheme in (('merge','no-alpha') if rep % 2 else ('no-alpha','merge')):
                    tasks.append((backend, rep, t, scheme))
    monitoring = []
    for cmd, name in ((['iostat', '-t', '-dxm', '1', 'nvme3n1'], 'iostat-timestamped.log'),
                      (['mpstat', '-P', CPUS, '1'], 'mpstat-timestamped.log')):
        f = open(str(ROOT/'logs'/name), 'a')
        monitoring.append((subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT), f))
    try:
        for ordinal, (backend, rep, threads, scheme) in enumerate(tasks, 1):
            label='{}__T{}__{}__rep{}'.format(backend, threads, scheme, rep)
            output=ROOT/'results'/(label+'.json')
            if output.exists():
                continue
            ls=COMMON_L if backend=='all-hot' else BOUNDARY_L
            status(stage='search', status='running', task=label, ordinal=ordinal, total_runs=len(tasks),
                   completed_runs=summarize())
            log('RUN {}/{} {}'.format(ordinal,len(tasks),label))
            env=dict(ENV, HVS_BACKEND=backend, HVS_SCHEME=scheme)
            rows, start, end, path=run_logged(label, command(scheme,threads,ls), env, timed=True)
            if [p['L'] for p in rows] != ls or any(p['seconds'] < 10 or p['samples'] % 1000 for p in rows):
                raise RuntimeError('Incomplete or too-short point')
            check_backend(rows, backend)
            write_json(output, dict(backend=backend, scheme=scheme, threads=threads, repetition=rep,
                started_unix=start, finished_unix=end, points=rows, raw_log=str(path)))
            log('DONE {}/{} {}'.format(ordinal,len(tasks),label))
            summarize()
        status(stage='complete', status='complete', completed_runs=summarize(),
               measured_points=324, total_runs=len(tasks))
        log('HOTDATA_ALPHA_QUEUE_COMPLETE')
    finally:
        for p,f in monitoring:
            p.terminate()
            p.wait()
            f.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        if OWN_QUEUE:
            status(status='failed', error=str(e))
        traceback.print_exc()
        raise
    finally:
        # Release only this queue's explicitly named tmpfs images, never indexes.
        if OWN_QUEUE and MEM.is_dir():
            for name in ('coord.bin','merge-topo.bin','no-alpha-topo.bin'):
                for suffix in ('', '.partial'):
                    p=MEM/(name+suffix)
                    if p.is_file():
                        p.unlink()
            try:
                MEM.rmdir()
            except OSError:
                pass
