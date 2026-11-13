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

from build import BUILD, ENV, ROOT, sha

EXPERIMENTS = BUILD / 'experiments'
PREDECESSOR = EXPERIMENTS / 'budget-reorder-20261103'
REFERENCE = EXPERIMENTS / 'concurrency-alpha/results'
BINARY = ROOT / 'bin/cache-search'
LEVELS = [40, 60, 120, 160]
SECONDS = 6
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


def reference(rep):
    return REFERENCE / 'laion__fixed-0.5__T4__merge__rep{}.json'.format(rep)


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
    raise RuntimeError('target recall not covered: {}'.format(target))


def run(label, argv, expected, enabled, parity=False):
    env = dict(ENV, OMP_DYNAMIC='FALSE', OMP_PROC_BIND='close', OMP_PLACES='cores',
               OMP_MAX_ACTIVE_LEVELS='1', OMP_NESTED='FALSE',
               HVS_POINT_SECONDS=str(SECONDS), HVS_SHARED_CACHE_BYTES=str(CACHE_BYTES if enabled else 0))
    for key in ('HVS_DUMP_PREFIX', 'HVS_ONE_PASS'):
        env.pop(key, None)
    if parity:
        env.update(HVS_ONE_PASS='1', HVS_DUMP_PREFIX=str(ROOT / 'analysis' / label))
    command = [str(BINARY)] + argv + [str(L) for L in LEVELS]
    path = ROOT / 'logs' / (label + '.log')
    status('running', task=label, cache_enabled=enabled)
    started = time.time()
    with path.open('w') as log:
        timed = ['/usr/bin/time', '-v', '-o', str(ROOT / 'logs' / (label + '-rss.log'))] + command
        worker = subprocess.Popen(timed, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = worker.wait(timeout=160)
        except BaseException:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            raise
        if code:
            raise subprocess.CalledProcessError(code, timed)
    lines = path.read_text(errors='replace').splitlines()
    points = [json.loads(line[10:]) for line in lines if line.startswith('HVS_POINT ')]
    if 'HVS_CACHE_COMPLETE' not in lines or [p['L'] for p in points] != LEVELS:
        raise RuntimeError('missing completed points for ' + label)
    for point in points:
        old = expected[point['L']]
        for key in ('recall', 'comparisons', 'topology_ios', 'rerank_ios'):
            if abs(point[key] - old[key]) > 1e-7:
                raise RuntimeError('{} {} changed at L={}'.format(label, key, point['L']))
        if not parity and point['seconds'] < SECONDS:
            raise RuntimeError('short measurement')
        if (point['cache_bytes'] != (CACHE_BYTES if enabled else 0)
                or bool(point['shared_hits']) != enabled):
            raise RuntimeError('cache state was not exercised as requested')
        if abs(point['shared_hits'] + point['topology_backend_pages'] - point['shared_requested']) > 1e-7:
            raise RuntimeError('cache counter mismatch')
    save(ROOT / 'results' / (label + '.json'), dict(label=label, command=command, cache_enabled=enabled,
        parity=parity, started_unix=started, finished_unix=time.time(), points=points))
    return points


def summary(by_repeat, archived):
    rows = []
    for target in (90, 95):
        pairs = []
        for rep in sorted(by_repeat):
            result = by_repeat[rep]
            if len(result) != 2:
                continue
            off, on = (interpolate(result[key], target, METRICS) for key in ('off', 'on'))
            pairs.append(dict(repeat=rep, off=off, on=on,
                qps_change_pct=(on['qps'] / off['qps'] - 1) * 100,
                service_reduction_pct=(1 - on['external_service_us'] / off['external_service_us']) * 100))
        if pairs:
            row = dict(target=target, pairs=pairs, repeats=len(pairs))
            for name in ('off', 'on'):
                row[name] = {k: statistics.mean(p[name][k] for p in pairs) for k in METRICS + ('L',)}
            historical = [interpolate(p, target, ('qps', 'latency_us')) for p in archived]
            row['historical_merge'] = {k: statistics.mean(p[k] for p in historical) for k in ('qps', 'latency_us')}
            rows.append(row)
    save(ROOT / 'analysis/summary.json', dict(matched=rows, page_bytes=8192, cache_data_bytes=CACHE_BYTES,
        diagnostic='Same-batch off/on is the comparison; historical original merge is context only.'))


def main():
    with (ROOT / 'queue.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (ROOT / 'status.json').exists():
            raise RuntimeError('existing run status; preserve prior measurements')
        if not os.sched_getaffinity(0).issubset(set(range(48)) | set(range(96, 112))):
            raise RuntimeError('CPU affinity mismatch')
        build = json.loads((ROOT / 'analysis/build.json').read_text())
        if sha(BINARY) != build['binaries']['cache-search']:
            raise RuntimeError('unverified binary')
        for name, digest in build['sources'].items():
            if sha(ROOT / 'scripts' / name) != digest:
                raise RuntimeError('built source changed')
        archived = [json.loads(reference(i).read_text()) for i in (1, 2, 3)]
        old = archived[0]
        argv = old['command'][4:17]
        if argv[0] != 'float' or argv[2:4] != ['4', '4'] or argv[9:] != ['l2', '6', '0', '17']:
            raise RuntimeError('unexpected native argument layout')
        for path, digest in old['input_sha256'].items():
            if sha(path) != digest:
                raise RuntimeError('existing query/alpha/GT hash changed')
        index = Path(argv[1])
        stats = {p.name: (p.stat().st_size, p.stat().st_mtime_ns) for p in index.iterdir() if p.is_file()}
        expected = {p['L']: p for p in old['points']}
        save(ROOT / 'analysis/protocol.json', dict(threads=4, alpha=.5, K=10, beam=4, mode=6, strategy=17,
            levels=LEVELS, min_seconds=SECONDS, repeats=3, warmup_queries=1000, unique_queries=1000,
            cache_data_bytes=CACHE_BYTES, index_stats=stats, input_sha256=old['input_sha256'],
            original_references={str(reference(r)): sha(reference(r)) for r in (1, 2, 3)}))
        with wait_for_serial_window():
            status('serial_window', lock=str(PREDECESSOR / 'queue.lock'))
            for kind in ('off', 'on'):
                run('parity-' + kind, argv, expected, kind == 'on', parity=True)
            for L in LEVELS:
                for ext in ('ids', 'dists'):
                    a, b = (ROOT / 'analysis' / 'parity-{}.L{}.{}'.format(kind, L, ext) for kind in ('off', 'on'))
                    if a.read_bytes() != b.read_bytes():
                        raise RuntimeError('byte-level output differs at L={} {}'.format(L, ext))
            save(ROOT / 'analysis/parity.json', dict(status='PASS', levels=LEVELS,
                ids_and_distances_byte_equal=True, original_recall_and_counts_equal=True))
            by_repeat = {}
            for rep, order in enumerate([('off', 'on'), ('on', 'off'), ('off', 'on')], 1):
                by_repeat[rep] = {}
                for kind in order:
                    by_repeat[rep][kind] = run('r{}-{}'.format(rep, kind), argv, expected, kind == 'on')
                summary(by_repeat, [r['points'] for r in archived])
            for name, identity in stats.items():
                st = (index / name).stat()
                if (st.st_size, st.st_mtime_ns) != identity:
                    raise RuntimeError('index changed during test: ' + name)
        status('complete', curves=6, points=24, parity='PASS', serial_lock_released=True)


def stop(signum, frame):
    raise InterruptedError('runner interrupted by signal {}'.format(signum))


if __name__ == '__main__':
    for sig in (signal.SIGTERM, signal.SIGHUP, signal.SIGINT):
        signal.signal(sig, stop)
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        traceback.print_exc()
        sys.exit(1)
