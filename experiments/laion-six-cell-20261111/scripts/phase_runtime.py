"""Metric-consistent staircase, frozen L/r tuning and three-arm layout study.

Called by bootstrap with the already locked coordinator module. Short
diagnostics are workload/correctness measurements, never timing evidence.
"""
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import time
import numpy as np
import numa_guard
import queue_guard

ROOT = Path(__file__).resolve().parent.parent
CPUS = [72, 73, 74, 75]
GRID = [20, 40, 60, 80, 100, 120, 160, 200, 300, 400, 600, 800]
RERANK = [10, 20, 40, 80, 160, 320]
COORD = Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
PQ = '/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/'
EXTRA = dict(OMP_NUM_THREADS='4', OMP_THREAD_LIMIT='4', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
B = None
SPLIT = None


def pending_points():
    path = ROOT/'analysis/pending-measurements.json'
    return json.loads(path.read_text()) if path.exists() else {}


def clear_pending(label):
    pending = pending_points()
    if label in pending:
        del pending[label]
        B.save(ROOT/'analysis/pending-measurements.json', pending)


def save_progress(label):
    B.save(ROOT/'progress.json', dict(updated_unix=time.time(), last_completed_task=label,
        accepted_timing_points=len(list((ROOT/'analysis').glob('report-time-*-accepted.json'))),
        pending_timing_points=len(pending_points())))


def layout(arm, kind):
    graph = ROOT/'scratch'/('single' if arm == 'single' else 'dual')
    stem = 'full' if arm == 'full' else 'compact'
    directory = ROOT/'scratch'/('{}-{}'.format(arm, kind))
    directory.mkdir(exist_ok=True)
    if kind == 'T_XY':
        for name, target in [('group-1.bin', graph/(stem+'.topo')), ('group-6.bin', COORD)]:
            path = directory/name
            if not path.exists():
                path.symlink_to(target)
            if path.resolve() != target.resolve():
                raise RuntimeError('unexpected shared source layout')
    elif not (directory/'identity.json').exists():
        if shutil.disk_usage(str(ROOT)).free < COORD.stat().st_size + (1 << 30):
            raise RuntimeError('not enough space for one coupled layout; no old data will be deleted')
        B.quiet(CPUS)
        B.run([ROOT/'bin/metric-search', 'build', graph/(stem+'.meta'), graph/(stem+'.topo'), COORD,
               directory, 'TXY', directory/'identity.json'], 'build-{}-TXY'.format(arm), CPUS, EXTRA)
    return graph/(stem+'.meta'), graph/(stem+'.topo'), directory


def proc_cpu(pid):
    try:
        fields = Path('/proc/{}/stat'.format(pid)).read_text().rsplit(')', 1)[1].split()
        return int(fields[11])+int(fields[12])
    except (OSError, ValueError):
        return None


def disk_stats():
    a = list(map(int, Path('/sys/block/nvme3n1/stat').read_text().split()))
    return dict(read_bytes=a[2]*512, write_bytes=a[6]*512, io_ms=a[9])


def sample(pid):
    return dict(time=time.time(), cpu=B.cpu_ticks(CPUS+[c+96 for c in CPUS]),
                own_cpu=proc_cpu(pid), disk=disk_stats())


def monitored(args, label, env, result):
    command = ['nice', '-n', '10', 'ionice', '-c', '2', '-n', '7',
               'numactl', '--all', '--physcpubind='+','.join(map(str, CPUS)), '--membind=3']+list(map(str, args))
    # Rejections are based only on outside activity, never on measured QPS.
    attempt = 0
    run_id = str(int(time.time()*1e6))
    while attempt < 3:
        attempt += 1
        try:
            B.quiet(CPUS, disk=not env.get('LAYOUT_DRAM'), timeout=180)
        except TimeoutError as error:
            raise queue_guard.DeferredMeasurement(str(error))
        B.status('timing', task=label, attempt=attempt, cpus=CPUS)
        complete_env = dict(B.ENV)
        complete_env.update(env)
        stem = label+'-'+numa_guard.VERSION+'-run'+run_id+'-attempt'+str(attempt)
        logfile = ROOT/'logs'/(stem+'.log')
        with logfile.open('w') as log:
            child = subprocess.Popen(command, env=complete_env, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            B.CHILD = child
            child_start = time.monotonic()
            timed_out = False
            samples = [sample(child.pid)]
            observations = []
            while child.poll() is None:
                if time.monotonic()-child_start > 900:
                    queue_guard.stop_owned_group(child)
                    timed_out = True
                    break
                time.sleep(1)
                samples.append(sample(child.pid))
                observations.append(numa_guard.observe(child.pid))
            B.CHILD = None
        if timed_out:
            B.save(ROOT/'analysis'/(stem+'.json'), dict(accepted=False, timeout_seconds=900,
                command=command, environment=env, reason='measurement_process_timeout'))
            raise queue_guard.DeferredMeasurement('measurement process exceeded 900 seconds: '+label)
        if child.returncode:
            raise RuntimeError('{} failed with {}'.format(label, child.returncode))
        point = json.loads(result.read_text())[0]
        start, end = point['start_unix'], point['start_unix']+point['seconds']
        intervals = []
        for a, b in zip(samples, samples[1:]):
            if a['time'] < start or b['time'] > end or a['own_cpu'] is None or b['own_cpu'] is None:
                continue
            elapsed = b['time']-a['time']
            busy = sum(b['cpu'][c][1]-a['cpu'][c][1] for c in a['cpu'])
            own = b['own_cpu']-a['own_cpu']
            capacity = elapsed*os.sysconf('SC_CLK_TCK')*len(CPUS)
            intervals.append(dict(start=a['time'], seconds=elapsed,
                external_cpu_fraction=max(0., busy-own)/capacity,
                device_write_mib_s=(b['disk']['write_bytes']-a['disk']['write_bytes'])/elapsed/(1 << 20),
                device_read_mib_s=(b['disk']['read_bytes']-a['disk']['read_bytes'])/elapsed/(1 << 20),
                device_util=(b['disk']['io_ms']-a['disk']['io_ms'])/elapsed/1000.))
        cpu = [a['external_cpu_fraction'] for a in intervals]
        writes = [a['device_write_mib_s'] for a in intervals]
        numa = numa_guard.validate(observations, start, end)
        accepted = bool(len(intervals) >= 5 and numa['verified']
                        and np.mean(cpu) <= .10 and max(cpu) <= .35
                        and np.mean(writes) <= 8 and max(writes) <= 32)
        record = dict(command=command, environment=env, accepted=accepted, numa=numa,
                      memory_bind_3_verified=numa['verified'],
                      attempt=attempt, monitor_run_id=run_id, intervals=intervals,
                      mean_external_cpu_fraction=float(np.mean(cpu)) if cpu else None)
        B.save(ROOT/'analysis'/(stem+'.json'), record)
        if accepted:
            B.save(ROOT/'analysis'/(label+'-accepted.json'), record)
            clear_pending(label)
            save_progress(label)
            return point
        shutil.copyfile(str(result), str(result.with_suffix('.'+numa_guard.VERSION+'.'+run_id+'.rejected-{}.json'.format(attempt))))
        if not numa['verified']:
            raise RuntimeError('NUMA measurement guard failed: '+','.join(numa['rejection_reasons']))
        B.status('retrying_external_interference', task=label, attempt=attempt)
    raise queue_guard.DeferredMeasurement('three attempts rejected for external interference: '+label)


def query(arm, kind, medium, exact, filtering, L, r, cohort, label,
          timed=False, trace=False):
    result = ROOT/'results'/(label+'.json')
    if result.exists() and (not timed or (ROOT/'analysis'/(label+'-accepted.json')).exists()):
        if timed:
            clear_pending(label)
        return json.loads(result.read_text())[0]
    meta, topo, folder = layout(arm, kind)
    files = {k: v['path'] for k, v in SPLIT['cohorts'][cohort]['files'].items()}
    if arm == 'single':
        files['gt'] = str(ROOT/'scratch'/('single-'+cohort+'-gt.bin'))
    env = dict(EXTRA)
    if medium == 'DRAM':
        env['LAYOUT_DRAM'] = '1'
    if exact:
        env['METRIC_NAV_EXACT'] = '1'
    if arm == 'single':
        env['METRIC_CONCAT'] = '1'
    if arm == 'full' and not filtering:
        env['LAYOUT_BYPASS_ALPHA'] = '1'
    env['METRIC_RERANK'] = str(r)
    if not timed:
        env.update(LAYOUT_ONE_PASS='1', LAYOUT_GT_DIAGNOSTIC='1')
    if trace:
        env.update(LAYOUT_TRACE=str(ROOT/'analysis'/label), LAYOUT_DUMP=str(ROOT/'analysis'/label))
    args = [ROOT/'bin/metric-search', 'query', meta, topo, COORD, folder, kind, result, PQ,
            files['query1'], files['query2'], files['alpha'], files['gt'], 4, 10, L]
    if timed:
        point = monitored(args, label, env, result)
    else:
        # Recall/trajectory/work counters do not use this run's wall time.
        # Keep the four-core low-priority limit, but do not wait for a quiet
        # timing window just to compute deterministic tuning diagnostics.
        B.run(args, label, CPUS, env)
        point = json.loads(result.read_text())[0]
    if exact and (point['reranked_nodes'] or point['pq_space_evals'] or point['coordinate_submitted_pages']):
        raise RuntimeError('exact DRAM arm unexpectedly used PQ or rerank')
    if not timed:
        save_progress(label)
    return point


def timed_or_defer(args):
    label = args[-1]
    try:
        query(*args, timed=True)
        return True
    except queue_guard.DeferredMeasurement as error:
        pending = pending_points()
        previous = pending.get(label, {})
        pending[label] = dict(arguments=list(args), reason=str(error), updated_unix=time.time(),
                              deferred_passes=previous.get('deferred_passes', 0)+1)
        B.save(ROOT/'analysis/pending-measurements.json', pending)
        B.status('point_deferred', task=label, reason=str(error), pending_points=len(pending))
        return False


def measure(manifest, media):
    arm, kind, exact = manifest['arm'], manifest['layout'], manifest['exact']
    selected = manifest['selected']
    # Held-out diagnostics include every nondominated tuning point. None is
    # chosen or dropped based on its held-out recall or latency.
    for p in manifest['tuning_frontier']:
        label = 'report-work-{}-{}-{}-f{}-L{}-r{}'.format(
            arm, kind, 'exact' if exact else 'pq', int(p['filtering']), p['L'], p['r'])
        query(arm, kind, 'DRAM', exact, p['filtering'], p['L'], p['r'], 'report', label)
    deferred = []
    for repeat in range(3):
        order = selected[repeat % len(selected):]+selected[:repeat % len(selected)]
        if repeat % 2:
            order = list(reversed(order))
        for p in order:
            for medium in (media if repeat % 2 == 0 else list(reversed(media))):
                label = 'report-time-{}-{}-{}-{}-f{}-L{}-r{}-repeat{}'.format(
                    arm, kind, 'exact' if exact else 'pq', medium, int(p['filtering']), p['L'], p['r'], repeat+1)
                args = (arm, kind, medium, exact, p['filtering'], p['L'], p['r'], 'report', label)
                if not timed_or_defer(args):
                    deferred.append(args)
    # Give failed points one bounded recovery pass after the other points.
    # Unresolved gaps remain explicit while subsequent stages still run.
    for args in deferred:
        timed_or_defer(args)
    B.save(ROOT/'analysis'/('stage-{}-{}-{}-status.json'.format(arm, kind, 'exact' if exact else 'pq')),
           dict(completed_unix=time.time(), unresolved_points=[a[-1] for a in deferred if a[-1] in pending_points()]))

