#!/usr/bin/env python3
"""Native CC3M recall-95 supplement, isolated on NUMA node 3."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
TIMING = BUILD/'experiments/io-time-20261104'
PREDECESSORS = [BUILD/'experiments/budget-reorder-20261103', TIMING]
NUMA_CPUS = set(range(72, 96)) | set(range(168, 192))
METHODS = ['baseline', 'full', 'merge', 'no-alpha']
SCREEN_LEVELS = [600, 800, 1200, 1600]
ENV = dict(os.environ, LC_ALL='C', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='4', OMP_THREAD_LIMIT='4', OMP_MAX_ACTIVE_LEVELS='1',
           OMP_NESTED='FALSE', OMP_PROC_BIND='close', OMP_PLACES='cores', HVS_POINT_SECONDS='10')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+ENV.get('LD_LIBRARY_PATH', '')
for name in ['HVS_ONE_PASS', 'PHASE_DUMP_PREFIX', 'PHASE_DISABLE_OBSERVER']:
    ENV.pop(name, None)


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def save(path, data):
    path = Path(path)
    tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    os.replace(str(tmp), str(path))


def status(**data):
    save(ROOT/'status.json', dict(updated_at=now(), numa_node=3, **data))


def sha(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(8<<20), b''):
            h.update(chunk)
    return h.hexdigest()


def cases():
    selected = json.loads((ROOT/'analysis/input-cases.json').read_text())
    assert [c['method'] for c in selected] == METHODS
    for c in selected:
        assert c['dataset'] == 'cc3m' and c['condition'] == 'mixed'
        assert len(c['argv']) == 13 and c['argv'][2:4] == ['4', '4']
        assert c['argv'][8] == '10' and c['argv'][11] == '0'
        assert c['binary'].startswith(str(TIMING/'bin')+'/')
        assert str(400) in c['expected_points']
    return selected


def bracket(points, target=95):
    points = sorted(points, key=lambda p: p['L'])
    for p in points:
        if p['recall'] == target:
            return [p['L']]
    for a, b in zip(points, points[1:]):
        if a['recall'] < target < b['recall']:
            return [a['L'], b['L']]
    return []


def equal_counts(point, expected):
    for name in ['recall', 'comparisons', 'topology_ios', 'rerank_ios']:
        if abs(point[name]-expected[name]) > 1e-7:
            raise RuntimeError('Output/native counter changed: '+name)


def verify_identity():
    manifest = json.loads((ROOT/'analysis/identity.json').read_text())
    for rel, value in manifest['controller_sha256'].items():
        if sha(ROOT/rel) != value:
            raise RuntimeError('Controller changed; use a new experiment identity: '+rel)
    build = json.loads((TIMING/'analysis/build.json').read_text())
    if sha(TIMING/'analysis/build.json') != manifest['timing_build_sha256']:
        raise RuntimeError('Timing driver provenance changed')
    for name, value in build['binaries'].items():
        if sha(TIMING/'bin'/name) != value:
            raise RuntimeError('Timing binary changed')
    for source in build['libraries'].values():
        if sha(source['path']) != source['sha256']:
            raise RuntimeError('Native library changed')
    for c in cases():
        if sha(c['reference_json']) != c['reference_sha256']:
            raise RuntimeError('Archived native reference changed')
        for path, value in c['input_sha256'].items():
            if sha(path) != value:
                raise RuntimeError('Query/alpha/GT changed')
        for path, expected in c['index_stats'].items():
            s = Path(path).stat()
            if (s.st_size, s.st_mtime_ns) != (expected['bytes'], expected['mtime_ns']):
                raise RuntimeError('Index identity changed')


def wait_predecessors():
    while True:
        states = []
        for p in PREDECESSORS:
            try:
                d = json.loads((p/'status.json').read_text())
            except (OSError, ValueError):
                d = dict(status='unavailable')
            states.append(dict(path=str(p), status=d.get('status'), task=d.get('task'),
                               updated_at=d.get('updated_at')))
        status(status='waiting_predecessors', predecessors=states, threads=4,
               note='Require both locks, completed budget queue, and paused or completed I/O queue.')
        if all(s['status'] == 'complete' or
               (p == TIMING and s['status'] == 'paused_by_user')
               for p, s in zip(PREDECESSORS, states)):
            locks = []
            try:
                for p in PREDECESSORS:
                    f = (p/'queue.lock').open('r')
                    locks.append(f)
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return locks
            except BlockingIOError:
                for f in locks:
                    f.close()
        time.sleep(20)


def execute(case, label, levels, one_pass=False, observer=True, dump=False):
    result = ROOT/'results'/(label+'.json')
    if result.exists():
        run = json.loads(result.read_text())
        assert run['levels'] == levels and run['observer_enabled'] == observer
        assert run['one_pass'] == one_pass and run['numa_node'] == 3
        return run['points']
    assert levels == sorted(set(levels)) and all(400 <= l <= 1600 for l in levels)
    env = dict(ENV)
    if one_pass:
        env['HVS_ONE_PASS'] = '1'
    if not observer:
        env['PHASE_DISABLE_OBSERVER'] = '1'
    if dump:
        assert one_pass
        env['PHASE_DUMP_PREFIX'] = str(ROOT/'analysis'/label)
    cmd = [case['binary']]+case['argv']+[str(l) for l in levels]
    status(status='running', task=label, method=case['method'], threads=4)
    started = time.time()
    with (ROOT/'logs'/(label+'.log')).open('w') as output, (ROOT/'logs'/(label+'.pidstat')).open('w') as telemetry:
        process = subprocess.Popen(cmd, stdout=output, stderr=subprocess.STDOUT, env=env)
        monitor = subprocess.Popen(['pidstat', '-h', '-u', '-r', '-w', '-p', str(process.pid), '1'],
                                   stdout=telemetry, stderr=subprocess.STDOUT, env=ENV)
        try:
            code = process.wait()
        finally:
            monitor.terminate()
            monitor.wait()
    text = (ROOT/'logs'/(label+'.log')).read_text()
    if code or 'HVS_IO_PHASE_COMPLETE' not in text:
        raise RuntimeError('Incomplete native run '+label)
    points = [json.loads(line[len('HVS_POINT '):]) for line in text.splitlines() if line.startswith('HVS_POINT ')]
    assert [p['L'] for p in points] == levels
    for p in points:
        assert p['unique_queries'] == 1000 and p['samples'] % 1000 == 0
        assert p['observer_enabled'] == observer
        assert one_pass or p['seconds'] >= 10
        assert p['reader_total_us'] <= p['external_service_us']+.01
        assert abs(p['reader_coupled_us']+p['reader_topology_us']+p['reader_coordinate_us']-p['reader_total_us']) < 1e-6
        if p['L'] == 400:
            equal_counts(p, case['expected_points']['400'])
        if observer:
            requests = sum(p[n+'_backend_requests'] for n in ['coupled', 'topology', 'coordinate'])
            size = sum(p[n+'_backend_bytes'] for n in ['coupled', 'topology', 'coordinate'])
            assert abs(requests*8192-size) < .01
    save(result, dict(dataset='cc3m', condition='mixed', method=case['method'], threads=4, numa_node=3,
                     levels=levels, one_pass=one_pass, observer_enabled=observer, points=points,
                     command=cmd, started_unix=started, finished_unix=time.time(),
                     reference_sha256=case['reference_sha256']))
    print(now(), 'DONE', label, flush=True)
    return points


def prepare_levels(case):
    method = case['method']
    points = execute(case, method+'__screen__L400', [400], one_pass=True)
    for level in SCREEN_LEVELS:
        if bracket(points):
            break
        points += execute(case, method+'__screen__L'+str(level), [level], one_pass=True)
    bounds = bracket(points)
    while len(bounds) == 2 and bounds[1]-bounds[0] > 100:
        level = (bounds[0]+bounds[1])//2
        points += execute(case, method+'__screen__L'+str(level), [level], one_pass=True)
        bounds = bracket(points)
    levels = sorted(set([400]+bounds)) if bounds else sorted(p['L'] for p in points)
    if max(levels) > 1600:
        raise RuntimeError('Native sector scratch safety limit exceeded')
    on = execute(case, method+'__parity__on', levels, one_pass=True, dump=True)
    off = execute(case, method+'__parity__off', levels, one_pass=True, observer=False, dump=True)
    expected = {p['L']: p for p in points}
    for a, b in zip(on, off):
        equal_counts(a, b)
        equal_counts(a, expected[a['L']])
        for suffix in ['ids', 'dists']:
            stem = '__parity__{}.L{}.{}'.format('{}', a['L'], suffix)
            assert sha(ROOT/'analysis'/(method+stem.format('on'))) == sha(ROOT/'analysis'/(method+stem.format('off')))
    save(ROOT/'analysis'/(method+'-parity.json'), dict(status='PASS', levels=levels,
        ids_and_distances_byte_equal=True, screen_and_native_counters_equal=True,
        L400_archived_native_match=True))
    record = dict(method=method, levels=levels, bracket95=bounds, target_covered=bool(bounds),
                  screen=sorted(points, key=lambda p: p['L']), expected=expected)
    save(ROOT/'analysis'/(method+'-levels.json'), record)
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--validate-config', action='store_true')
    args = parser.parse_args()
    selected = cases()
    if args.validate_config:
        print('CONFIG_PASS four native methods, four threads, existing GT, L<=1600; no searches run')
        return
    for d in ['results', 'analysis', 'logs']:
        (ROOT/d).mkdir(exist_ok=True)
    lock = (ROOT/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    cpus = set(os.sched_getaffinity(0))
    if not cpus or not cpus.issubset(NUMA_CPUS):
        raise RuntimeError('CPU affinity must be restricted to NUMA node 3')
    binding = subprocess.check_output(['numactl', '--show'], env=ENV, universal_newlines=True)
    policy = dict(line.split(':', 1) for line in binding.splitlines() if ':' in line)
    if policy.get('policy', '').strip() != 'bind' or policy.get('membind', '').split() != ['3']:
        raise RuntimeError('Memory allocation must be bound to NUMA node 3')
    print(now(), 'NUMA3_BINDING_VERIFIED', sorted(cpus), flush=True)
    held = wait_predecessors()
    monitors = []
    try:
        verify_identity()
        for command, name in [(['iostat', '-t', '-dxm', '1', 'nvme3n1'], 'iostat.log'),
                              (['mpstat', '-P', ','.join(map(str, sorted(cpus))), '5'], 'mpstat.log')]:
            f = (ROOT/'logs'/name).open('a')
            monitors.append((subprocess.Popen(command, stdout=f, stderr=subprocess.STDOUT, env=ENV), f))
        plans = {c['method']: prepare_levels(c) for c in selected}
        for repeat in [1, 2, 3]:
            for method in METHODS[repeat-1:]+METHODS[:repeat-1]:
                case = next(c for c in selected if c['method'] == method)
                plan = plans[method]
                for observer in ([True, False] if repeat % 2 else [False, True]):
                    levels = plan['levels'] if observer else [plan['levels'][-1]]
                    label = '{}__r{}__{}'.format(method, repeat, 'on' if observer else 'off')
                    points = execute(case, label, levels, observer=observer)
                    for p in points:
                        equal_counts(p, plan['expected'][p['L']])
        subprocess.run(['python3', str(ROOT/'scripts/summarize.py')], env=ENV, check=True)
        audit = json.loads((ROOT/'analysis/completeness.json').read_text())
        assert audit['status'] == 'PASS' and audit['timed_runs'] == 24 and audit['parity_checks'] == 4
        status(status='complete', threads=4, timed_runs=24, parity_checks=4,
               target95_covered_methods=audit['target95_covered_methods'],
               target95_uncovered_methods=audit['target95_uncovered_methods'], no_new_GT=True)
        print(now(), 'CC3M_RECALL95_T4_COMPLETE', flush=True)
    finally:
        for process, f in monitors:
            process.terminate()
            process.wait()
            f.close()
        for f in held:
            f.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        status(status='failed', error=str(e))
        traceback.print_exc()
        raise
