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
        B.quiet(CPUS, disk=not env.get('LAYOUT_DRAM'))
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
                if time.monotonic()-child_start > 180:
                    queue_guard.stop_owned_group(child)
                    timed_out = True
                    break
                time.sleep(1)
                samples.append(sample(child.pid))
                observations.append(numa_guard.observe(child.pid))
            B.CHILD = None
        if timed_out:
            B.save(ROOT/'analysis'/(stem+'.json'), dict(accepted=False, timeout_seconds=180,
                command=command, environment=env, reason='measurement_process_timeout'))
            raise queue_guard.DeferredMeasurement('measurement process exceeded 180 seconds: '+label)
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


def parity():
    for filtering in (True, False):
        for medium in ('DRAM', 'SSD'):
            label = 'parity-{}-{}'.format('on' if filtering else 'off', medium)
            query('full', 'T_XY', medium, False, filtering, 60, 40, 'tune', label, trace=True)
        for suffix in ('.ids', '.dists'):
            a = ROOT/'analysis'/('parity-{}-DRAM.L60'.format('on' if filtering else 'off')+suffix)
            b = ROOT/'analysis'/('parity-{}-SSD.L60'.format('on' if filtering else 'off')+suffix)
            if a.read_bytes() != b.read_bytes():
                raise RuntimeError('DRAM/SSD result parity failed')
        # Trace contains only deterministic node IDs, no timing.
        a = ROOT/'analysis'/('parity-{}-DRAM.L60'.format('on' if filtering else 'off'))
        b = ROOT/'analysis'/('parity-{}-SSD.L60'.format('on' if filtering else 'off'))
        if a.read_bytes() != b.read_bytes():
            raise RuntimeError('DRAM/SSD trajectory parity failed')
    query('compact', 'T_XY', 'DRAM', False, False, 60, 40, 'tune', 'parity-compact-DRAM', trace=True)
    for suffix in ('', '.ids', '.dists'):
        a = ROOT/'analysis'/('parity-off-DRAM.L60'+suffix)
        b = ROOT/'analysis'/('parity-compact-DRAM.L60'+suffix)
        if a.read_bytes() != b.read_bytes():
            raise RuntimeError('Full-bypass/compact trajectory or result mismatch')
    # Independently recompute returned scores, so a wrong fusion formula in
    # both media cannot pass solely by agreeing with itself.
    from audit_objective import matrix
    files = SPLIT['cohorts']['tune']['files']
    qx, qy = matrix(files['query1']['path']), matrix(files['query2']['path'])
    alpha = matrix(files['alpha']['path'])[:, 0]
    coordinates = np.memmap(str(COORD), dtype=np.float32, mode='r', shape=(456699, 2048))
    relative_error = 0.
    for filtering in ('on', 'off'):
        prefix = ROOT/'analysis'/('parity-{}-DRAM.L60'.format(filtering))
        ids = np.fromfile(str(prefix)+'.ids', dtype=np.uint32).reshape(500, 10)
        scores = np.fromfile(str(prefix)+'.dists', dtype=np.float32).reshape(500, 10)
        for q in range(0, 500, 10):
            records = np.asarray(coordinates[ids[q]], dtype=np.float64)
            d1 = np.sqrt(np.sum((records[:, :768]-qx[q])**2, axis=1))
            d2 = np.sqrt(np.sum((records[:, 768:1536]-qy[q])**2, axis=1))
            expected = float(alpha[q])*d1+(1-float(alpha[q]))*d2
            error = float(np.max(np.abs(scores[q]-expected)/np.maximum(1e-8, np.abs(expected))))
            relative_error = max(relative_error, error)
    if relative_error > 5e-6:
        raise RuntimeError('returned distances disagree with exact root objective')
    B.save(ROOT/'analysis/query-parity.json', dict(status='PASS', queries=500, L=60, r=40,
        bitwise_equal_result_ids=True, bitwise_equal_distances=True, equal_trajectory=True,
        full_bypass_compact_parity=True, independent_score_max_relative_error=relative_error))


def screen(arm, kind, exact=False):
    key = '{}-{}-{}'.format(arm, kind, 'exact' if exact else 'pq')
    output = ROOT/'analysis'/('frozen-'+key+'.json')
    if output.exists():
        return json.loads(output.read_text())
    rows = []
    for L in GRID:
        depths = [L] if exact else sorted(set([r for r in RERANK if r <= L]+[L]))
        for r in depths:
            for filtering in ((True, False) if arm == 'full' else (False,)):
                label = 'tune-{}-f{}-L{}-r{}'.format(key, int(filtering), L, r)
                point = query(arm, kind, 'DRAM', exact, filtering, L, r, 'tune', label)
                rows.append(dict(filtering=filtering, L=L, r=r, recall=point['recall'],
                    pages=point['submitted_pages'], pq=point['pq_space_evals'], exact=point['navigation_exact_space_evals'],
                    source=str(ROOT/'results'/(label+'.json'))))
    # Select by deterministic work, not by a lucky latency. Keep the minimum
    # page and minimum compute representative at each prespecified recall band.
    # Preserve all nondominated tuning points in the manifest, including those
    # not selected for repeated timing. This is not a latency-optimal oracle.
    selected, frontier = [], []
    for filtering in ((True, False) if arm == 'full' else (False,)):
        candidates = [p for p in rows if p['filtering'] == filtering]
        for point in candidates:
            if not any(q['recall'] >= point['recall'] and q['pages'] <= point['pages'] and
                (q['pq']+q['exact']) <= (point['pq']+point['exact']) and
                (q['recall'] > point['recall'] or q['pages'] < point['pages'] or
                 (q['pq']+q['exact']) < (point['pq']+point['exact'])) for q in candidates):
                frontier.append(point)
        for target in (80, 85, 88, 90, 92, 94, 95, 96, 97, 98, 99):
            covering = [p for p in candidates if p['recall'] >= target]
            if not covering:
                continue
            for order in (lambda p:(p['pages'], p['pq']+p['exact'], p['L'], p['r']),
                          lambda p:(p['pq']+p['exact'], p['pages'], p['L'], p['r'])):
                point = min(covering, key=order)
                if point not in selected:
                    selected.append(point)
        if not any(p['filtering'] == filtering for p in selected):
            selected.append(max(candidates, key=lambda p:p['recall']))
    frozen = dict(arm=arm, layout=kind, exact=exact, selected=selected, tuning_frontier=frontier,
                  selection='minimum pages and minimum computations per prespecified recall band; no latency selection',
                  frozen_unix=time.time(), tune_queries=500, report_queries=500,
                  split_sha256=B.digest(ROOT/'analysis/tuning-split-v2.json'))
    B.save(output, frozen)
    return frozen


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


def verify_coordinates():
    from audit_objective import matrix, DATA
    x, y = matrix(DATA/'base_img_emb.fvecs'), matrix(DATA/'base_text_emb.fvecs')
    stored = np.memmap(str(COORD), dtype=np.float32, mode='r', shape=(len(x), 2048))
    for start in range(0, len(x), 4096):
        z = stored[start:start+4096]
        if not np.array_equal(z[:, :768], x[start:start+4096]) or not np.array_equal(z[:, 768:1536], y[start:start+4096]):
            raise RuntimeError('stored coordinates differ from GT/build inputs')
    B.save(ROOT/'analysis/coordinate-identity.json', dict(status='PASS', nodes=len(x), dimensions=1536,
        all_coordinate_values_compared=True, original_coordinate_file=str(COORD)))


def single_prepare():
    directory = Path('/mnt/nvme2/wz/hvs-metric-consistent-20261111')
    if not (directory/'concat-gt.bin').exists():
        B.quiet(B.CPUS)
        B.run([os.sys.executable, ROOT/'scripts/audit_objective.py', 'concat', directory],
              'prepare-single-base-gt', B.CPUS)
    from audit_objective import matrix
    gt = matrix(directory/'concat-gt.bin', np.uint32)
    for cohort in ('tune', 'report'):
        target = ROOT/'scratch'/('single-'+cohort+'-gt.bin')
        ids = SPLIT['cohorts'][cohort]['query_ids']
        with target.open('wb') as out:
            out.write(struct.pack('<II', len(ids), 10))
            np.asarray(gt[ids]).astype('<u4').tofile(out)
    graph = ROOT/'scratch/single'
    graph.mkdir(exist_ok=True)
    if not (ROOT/'analysis/single-validated.json').exists():
        B.quiet(B.CPUS)
        start = time.time()
        B.run([ROOT/'bin/build-graph', 'single', directory/'concat-base.bin', '-', 456699, 768, 768, 8, graph],
              'build-openimage-concat-single', B.CPUS)
        from validate_graph import validate
        proof = validate(graph, 456699, 768, 768)
        proof.update(wall_seconds=time.time()-start, concat_gt_sha256=B.digest(directory/'concat-gt.bin'),
                     concat_base_sha256=B.digest(directory/'concat-base.bin'))
        B.save(ROOT/'analysis/single-validated.json', proof)


def main(coordinator):
    global B, SPLIT
    B = coordinator
    SPLIT = json.loads((ROOT/'analysis/tuning-split-v2.json').read_text())
    ids = [v['query_ids'] for v in SPLIT['cohorts'].values()]
    if len(ids) != 2 or any(len(v) != 500 for v in ids) or set(ids[0]) & set(ids[1]):
        raise RuntimeError('frozen query split changed')
    for cohort in SPLIT['cohorts'].values():
        for file in cohort['files'].values():
            if B.digest(file['path']) != file['sha256']:
                raise RuntimeError('query/GT slice identity changed')
    identity = dict(binary_sha256=B.digest(ROOT/'bin/metric-search'),
                    phase_two_sha256=B.digest(Path(__file__)),
                    monitor_guard_sha256=B.digest(Path(numa_guard.__file__)),
                    queue_guard_sha256=B.digest(Path(queue_guard.__file__)),
                    pq_sha256={name:B.digest(Path(PQ)/name) for name in (
                        'emb_pq_compressed.bin', 'loc_pq_compressed.bin',
                        'emb_pq_pivots.bin', 'loc_pq_pivots.bin')},
                    coordinates_sha256=B.digest(COORD),
                    graph_sha256={p.name:B.digest(p) for p in (ROOT/'scratch/dual').iterdir()
                                  if p.name in ('full.meta', 'full.topo', 'compact.meta', 'compact.topo', 'float-graph.bin')},
                    split_sha256=B.digest(ROOT/'analysis/tuning-split-v2.json'))
    frozen_identity = ROOT/'analysis/measurement-inputs.json'
    if frozen_identity.exists() and json.loads(frozen_identity.read_text()) != identity:
        raise RuntimeError('measured executable or source graph changed; use a new result namespace')
    B.save(frozen_identity, identity)
    if not (ROOT/'analysis/coordinate-identity.json').exists():
        verify_coordinates()
    if not (ROOT/'analysis/query-parity.json').exists():
        parity()
    exact = screen('full', 'T_XY', exact=True)
    measure(exact, ['DRAM'])
    pq = screen('full', 'T_XY')
    measure(pq, ['DRAM', 'SSD'])
    # Stage four uses a genuinely independently built 1536D single graph.
    single_prepare()
    for arm in ('single', 'compact', 'full'):
        for kind in ('T_XY', 'TXY'):
            frozen = screen(arm, kind)
            measure(frozen, ['SSD'])
            if kind == 'TXY' and (ROOT/'scratch'/('{}-TXY'.format(arm))/'group-7.bin').exists():
                # Only this queue's generated coupled page image is removed.
                # Keep topology, float graph, validation and all result files.
                temporary = ROOT/'scratch'/('{}-TXY'.format(arm))/'group-7.bin'
                proof = temporary.parent/'identity.json'
                identity = json.loads(proof.read_text())
                B.save(ROOT/'analysis'/('{}-TXY-retired.json'.format(arm)), dict(
                    generated_path=str(temporary), sha256=B.digest(temporary), validation=identity,
                    reason='bounded storage, canonical source and reproduction code retained'))
                temporary.unlink()
                proof.unlink()
    pending = pending_points()
    B.status('complete_with_pending' if pending else 'complete', metric_correctness='PASS',
             experiment_scope='OpenImages four priority tasks', pending_points=list(pending))
