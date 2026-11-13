import csv
import datetime
import fcntl
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import time
import traceback

import build
from build import BUILD, ENV, ROOT, sha

EXPERIMENTS = BUILD / 'experiments'
PREDECESSOR = EXPERIMENTS / 'budget-reorder-20261103'
REFERENCE = EXPERIMENTS / 'concurrency-alpha/results'
BINARY = ROOT / 'bin/cache-search'
DATASETS = {'openimage': 'mixed', 'cc3m': 'mixed', 'howto100m': 'mixed', 'laion': 'fixed-0.5'}
LOW_LEVELS = [10, 20, 40, 60, 80, 100, 120, 160, 200, 300, 400]
HIGH_LEVELS = [500, 600, 800, 1200, 1600]
TARGETS = [70, 80, 85, 90, 92, 95, 97, 98, 99]
SECONDS = 10
REPEATS = 3
INVARIANTS = ('recall', 'comparisons', 'topology_ios', 'rerank_ios', 'query_local_hits')
CACHE_BYTES = 256 << 20
METRICS = ('qps', 'latency_us', 'external_service_us', 'external_p95_us', 'external_p99_us',
           'shared_requested', 'shared_hits', 'topology_backend_pages', 'coordinate_backend_pages',
           'backend_bytes', 'reader_topology_us', 'reader_coordinate_us')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save(path, data):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    os.replace(str(tmp), str(path))


def status(state, **detail):
    save(ROOT / 'status.json', dict(status=state, updated_at=now(), **detail))
    print(now(), state, detail, flush=True)


def wait_for_serial_window():
    lock_path = PREDECESSOR / 'queue.lock'
    last_report = 0
    while True:
        predecessor = json.loads((PREDECESSOR / 'status.json').read_text())
        state = predecessor.get('status')
        if state == 'failed':
            raise RuntimeError('predecessor failed; measurements remain serialized')
        if state == 'complete':
            lock = lock_path.open('r')
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                lock.close()
            else:
                return lock
        if time.monotonic() - last_report >= 30:
            status('waiting_for_serial_lock', predecessor_status=state,
                   predecessor_task=predecessor.get('task'))
            last_report = time.monotonic()
        time.sleep(2)


def interpolate(rows, target, keys):
    rows = sorted(rows, key=lambda p: p['L'])
    for p in rows:
        if p['recall'] == target:
            return dict(target=target, L=p['L'], **{k: p[k] for k in keys})
    for a, b in zip(rows, rows[1:]):
        if a['recall'] < target < b['recall']:
            f = (target - a['recall']) / (b['recall'] - a['recall'])
            return dict(target=target, L=a['L'] + f * (b['L'] - a['L']),
                        **{k: a[k] + f * (b[k] - a[k]) for k in keys})
    return None


def run(label, argv, levels, expected, enabled, parity=False, screen_high=False):
    env = dict(ENV, OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close', OMP_PLACES='cores',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_NESTED='FALSE',
               HVS_POINT_SECONDS=str(SECONDS), HVS_SHARED_CACHE_BYTES=str(CACHE_BYTES if enabled else 0))
    for key in ('HVS_DUMP_PREFIX', 'HVS_ONE_PASS'):
        env.pop(key, None)
    if parity:
        env.update(HVS_ONE_PASS='1', HVS_DUMP_PREFIX=str(ROOT / 'analysis' / label))
    command = [str(BINARY)] + argv + [str(L) for L in levels]
    path = ROOT / 'logs' / (label + '.log')
    status('running', task=label, cache_enabled=enabled, levels=levels)
    started = time.time()
    with path.open('w') as log:
        timed = ['/usr/bin/time', '-v', '-o', str(ROOT / 'logs' / (label + '-rss.log'))] + command
        worker = subprocess.Popen(timed, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = worker.wait(timeout=1800)
        except BaseException:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            raise
    lines = path.read_text(errors='replace').splitlines()
    if code:
        if screen_high and any('HVS_TOPO_SCRATCH_LIMIT' in line for line in lines):
            return None
        raise subprocess.CalledProcessError(code, timed)
    points = [json.loads(line[10:]) for line in lines if line.startswith('HVS_POINT ')]
    if 'HVS_CACHE_COMPLETE' not in lines or [p['L'] for p in points] != levels:
        raise RuntimeError('missing completed points for ' + label)
    for point in points:
        old = expected.get(point['L'], {})
        for key in INVARIANTS:
            if key in old and abs(point[key] - old[key]) > 1e-7:
                raise RuntimeError('{} {} changed at L={}'.format(label, key, point['L']))
        if not parity and point['seconds'] < SECONDS:
            raise RuntimeError('short measurement')
        if (point['cache_bytes'] != (CACHE_BYTES if enabled else 0)
                or (not enabled and point['shared_hits'] != 0)):
            raise RuntimeError('cache state was not exercised as requested')
        if abs(point['coordinate_backend_pages'] - point['rerank_ios']) > 1e-7:
            raise RuntimeError('coordinate counter mismatch')
        if abs(point['shared_hits'] + point['topology_backend_pages'] - point['shared_requested']) > 1e-7:
            raise RuntimeError('cache counter mismatch')
    save(ROOT / 'results' / (label + '.json'), dict(label=label, command=command, cache_enabled=enabled,
        parity=parity, started_unix=started, finished_unix=time.time(), points=points))
    return points


def load_protocol(dataset, alpha):
    reference = REFERENCE / '{}__{}__T4__merge__rep1.json'.format(dataset, alpha)
    old = json.loads(reference.read_text())
    argv = old['command'][4:17]
    if argv[0] != 'float' or argv[2:4] != ['4', '4'] or argv[8:] != ['10', 'l2', '6', '0', '17']:
        raise RuntimeError('unexpected native argument layout')
    if set(argv[4:8]) != set(old['input_sha256']):
        raise RuntimeError('query input list differs from manifest')
    for path, digest in old['input_sha256'].items():
        if sha(path) != digest:
            raise RuntimeError('query/alpha/GT changed: ' + path)
    stats = {str(p): [p.stat().st_size, p.stat().st_mtime_ns]
             for p in Path(argv[1]).iterdir() if p.is_file()}
    protocol = dict(alpha=alpha, argv=argv, reference=str(reference), reference_sha256=sha(reference),
                    index_stats=stats, input_sha256=old['input_sha256'])
    return protocol, {p['L']: p for p in old['points']}


def parity(dataset, argv, expected):
    low_label = dataset + '__parity-off-low'
    off = run(low_label, argv, LOW_LEVELS, expected, False, parity=True)
    source_label = {L: low_label for L in LOW_LEVELS}
    rejected = {}

    def screen(L):
        label = '{}__parity-off-L{}'.format(dataset, L)
        points = run(label, argv, [L], expected, False, parity=True, screen_high=True)
        if points is None:
            rejected[L] = 'query-local topology scratch capacity exceeded'
            return
        source_label[L] = label
        off.extend(points)

    for L in HIGH_LEVELS:
        screen(L)
    if rejected:
        first_failure = min(rejected)
        lower = max(L for L in source_label if L < first_failure)
        midpoint = (lower + first_failure) // 2
        if midpoint not in source_label:
            screen(midpoint)
    levels = sorted(source_label)
    expected = {p['L']: p for p in off}
    on_label = dataset + '__parity-on'
    on = run(on_label, argv, levels, expected, True, parity=True)
    if not any(p['shared_hits'] > 0 for p in on):
        raise RuntimeError('shared cache never hit for ' + dataset)
    digests = {}
    for L in levels:
        digests[L] = {}
        for ext in ('ids', 'dists'):
            prefix = ROOT / 'analysis'
            a, b = (prefix / '{}.L{}.{}'.format(label, L, ext) for label in (source_label[L], on_label))
            if a.stat().st_size != 40000 or sha(a) != sha(b):
                raise RuntimeError('byte-level parity failed for {} L={} {}'.format(dataset, L, ext))
            digests[L][ext] = sha(a)
    save(ROOT / 'analysis' / (dataset + '-parity.json'),
         dict(status='PASS', levels=levels, rejected=rejected,
              recalls={L: expected[L]['recall'] for L in levels},
              ids_and_distances_sha256=digests, shared_hits={p['L']: p['shared_hits'] for p in on}))
    return levels, expected


def write_report(data):
    curves = []
    matched = []
    lines = ['# Shared topology cache, four datasets', '',
             'Same-batch Merge off/on. Four search threads, CPU and memory on NUMA 3.',
             'Each method uses one 1,000-query warm-up and at least 10 seconds of replay per L.',
             'Page counts below are backend submissions, not measured device reads.', '',
             '| Dataset | Measured L points | Recall range | Complete pairs |',
             '|---|---:|---:|---:|']
    for dataset, repeats in data.items():
        for rep, pair in repeats.items():
            for kind, points in pair.items():
                curves.extend(dict(dataset=dataset, repeat=rep, cache=kind, **p) for p in points)
        paired = {rep: pair for rep, pair in repeats.items() if len(pair) == 2}
        if not paired:
            continue
        points = next(iter(paired.values()))['off']
        recalls = [p['recall'] for p in points]
        lines.append('| {} | {} | {:.2f}–{:.2f}% | {} |'.format(dataset, len(points),
                     min(recalls), max(recalls), len(paired)))
        for target in TARGETS:
            pairs = []
            for rep, pair in paired.items():
                off, on = (interpolate(pair[kind], target, METRICS) for kind in ('off', 'on'))
                if off is not None and on is not None:
                    pairs.append(dict(repeat=rep, off=off, on=on))
            if pairs:
                row = dict(dataset=dataset, target=target, pairs=pairs)
                for kind in ('off', 'on'):
                    row[kind] = {k: statistics.mean(p[kind][k] for p in pairs) for k in METRICS + ('L',)}
                matched.append(row)
    lines += ['', '| Dataset | Recall | QPS off → on | Mean service off → on (µs) | Topology pages off → on | Coordinate pages off → on | Total page reduction |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in matched:
        off, on = row['off'], row['on']
        lines.append('| {} | {}% | {:.1f} → {:.1f} | {:.1f} → {:.1f} | {:.2f} → {:.2f} | {:.2f} → {:.2f} | {:.1%} |'.format(
            row['dataset'], row['target'], off['qps'], on['qps'], off['external_service_us'],
            on['external_service_us'], off['topology_backend_pages'], on['topology_backend_pages'],
            off['coordinate_backend_pages'], on['coordinate_backend_pages'], 1 - on['backend_bytes'] / off['backend_bytes']))
    lines += ['', 'Matched-recall rows interpolate within each repeat and average across complete pairs.',
              'Uncovered targets are omitted. The full measured curves and per-repeat metrics are in the CSV and JSON.']
    save(ROOT / 'analysis/summary.json', dict(matched=matched, curves=len(curves)))
    (ROOT / 'RESULTS.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if curves:
        with (ROOT / 'analysis/curves.csv').open('w', newline='') as output:
            writer = csv.DictWriter(output, fieldnames=list(curves[0]))
            writer.writeheader()
            writer.writerows(curves)


def main():
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'status.json').exists():
            raise RuntimeError('existing run status; preserve prior measurements')
        allowed = set(range(72, 96)) | set(range(168, 192))
        if os.sched_getaffinity(0) != allowed:
            raise RuntimeError('NUMA 3 CPU affinity mismatch')
        protocols, expected = {}, {}
        for dataset, alpha in DATASETS.items():
            protocols[dataset], expected[dataset] = load_protocol(dataset, alpha)
        save(ROOT / 'analysis/protocol.json',
             dict(datasets=protocols, threads=4, numa_cpu=3, numa_memory=3, K=10, beam=4,
                  mode=6, strategy=17, planned_levels=LOW_LEVELS + HIGH_LEVELS,
                  min_seconds=SECONDS, repeats=REPEATS, warmup_queries=1000, cache_data_bytes=CACHE_BYTES))
        with wait_for_serial_window():
            status('building', lock=str(PREDECESSOR / 'queue.lock'))
            build.main()
            tools = (
                ('iostat', ['iostat', '-x', '-t', '-y', '10']),
                ('pidstat', ['pidstat', '-h', '-d', '-u', '-C', 'cache-search', '10']),
            )
            monitors = []
            try:
                for name, command in tools:
                    with (ROOT / 'logs' / (name + '.log')).open('w') as log:
                        try:
                            monitors.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
                        except FileNotFoundError:
                            print(name, 'unavailable', flush=True)
                validated = {}
                for dataset, protocol in protocols.items():
                    validated[dataset] = parity(dataset, protocol['argv'], expected[dataset])
                measurements = {}
                total = sum(len(levels) * 2 * REPEATS for levels, _ in validated.values())
                completed = 0
                for number, (dataset, protocol) in enumerate(protocols.items()):
                    levels, counts = validated[dataset]
                    measurements[dataset] = {}
                    for rep in range(1, REPEATS + 1):
                        pair = measurements[dataset][rep] = {}
                        order = ('off', 'on') if (rep + number) % 2 else ('on', 'off')
                        for kind in order:
                            label = '{}__r{}-{}'.format(dataset, rep, kind)
                            pair[kind] = run(label, protocol['argv'], levels, counts, kind == 'on')
                            completed += len(levels)
                            save(ROOT / 'analysis/progress.json', dict(points=completed, total_points=total,
                                 last_curve=label, updated_at=now()))
                        write_report(measurements)
                for protocol in protocols.values():
                    for name, identity in protocol['index_stats'].items():
                        st = Path(name).stat()
                        if [st.st_size, st.st_mtime_ns] != identity:
                            raise RuntimeError('index changed during experiment: ' + name)
            finally:
                for monitor in monitors:
                    if monitor.poll() is None:
                        monitor.terminate()
                    monitor.wait()
        status('complete', datasets=list(DATASETS), curves=len(DATASETS) * REPEATS * 2,
               points=completed, parity='PASS', serial_lock_released=True)


def stop(signum, frame):
    raise InterruptedError('runner interrupted by signal {}'.format(signum))


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    for directory in ('analysis', 'logs', 'results'):
        (ROOT / directory).mkdir(exist_ok=True)
    try:
        main()
    except Exception as error:
        status('interrupted' if isinstance(error, InterruptedError) else 'failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)
