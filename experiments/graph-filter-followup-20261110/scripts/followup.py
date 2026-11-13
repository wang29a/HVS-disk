"""Fixed-layout alpha filter experiment, reusing the archived X endpoint."""
import fcntl
import os
from pathlib import Path
import signal
import statistics
import struct
import sys
import traceback


ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
ARCHIVE = EXP / 'openimage-matched-graphs-20261110'
sys.path.insert(0, str(ARCHIVE / 'scripts'))
import compare as prior
import numpy as np

tools, base, monitor = prior.tools, prior.base, prior.monitor
tools.ROOT = base.ROOT = monitor.ROOT = prior.ROOT = ROOT
base.ENGINE = ROOT / 'bin/endpoint-search'
TARGETS = (90, 95)
CACHE = 256 << 20
LEVELS = [10, 20, 24, 30, 34, 36, 40, 48, 60, 66, 80, 100, 120, 176]
FIELDS = ('qps', 'latency_us', 'expanded_nodes', 'pq_space_evals', 'inspected_edges',
          'alpha_rejected_edges', 'topology_submitted_pages', 'coordinate_submitted_pages',
          'topology_submit_wait_us', 'coordinate_submit_wait_us')



# Leave cores 80-93 and their SMT siblings for the two explicitly pinned jobs.
CPU_POOL = set(range(72,80)) | {94,95}


def quiet_cores(count=4):
    history = []
    while True:
        cpu, busy = base.disk_cpu_sample()
        rates = {a:max(cpu[a],cpu[b]) for a,b in base.CPU_PAIRS if a in CPU_POOL}
        ceiling = 10 if count==2 else 25
        quiet = {c for c, rate in rates.items() if rate<=15} if busy<=ceiling else set()
        history = (history+[quiet])[-3:]
        stable = set.intersection(*history)
        if len(history)==3 and len(stable)>=count:
            preferred = [74,75,76,77]
            if count==4 and set(preferred)<=stable:
                return preferred
            return sorted(sorted(stable,key=rates.get)[:count])
        base.status('waiting_for_quiet_cores',disk_busy_pct=busy,quiet_cores=len(quiet),
                    stable_cores=len(stable),sample_count=len(history),required=count)

def layout(c, name, kind, x_reference=None):
    folder = ROOT / 'scratch' / name
    n, dx, dy, degree, slots = struct.unpack_from('<5I', Path(c['meta']).read_bytes())
    widths = [4 + degree * (4 + 2 * slots), 4 * (dx + (dy if kind == 'T_XY' else 0))]
    sizes = [((n + 8192//w - 1) // (8192//w)) * 8192 for w in widths]
    fields = ('group-1.bin', 'group-6.bin' if kind == 'T_XY' else 'group-2.bin')
    proof = ROOT / 'analysis' / (name + '-layout.json')
    if proof.exists() and all((folder/f).is_file() and (folder/f).stat().st_size == s
                              for f, s in zip(fields, sizes)):
        if base.read(proof)['status'] == 'PASS':
            return folder
    base.require_space(sum(sizes))
    folder.mkdir(exist_ok=True)
    base.status('building_layout', task=name, expected_new_bytes=sum(sizes))
    base.run([base.ENGINE, 'build', c['meta'], c['topo'], c['coord'], folder, kind, proof],
             name+'-layout', {'LAYOUT_ORDER': c['order']})
    record = base.read(proof)
    if record['status'] != 'PASS' or record['record_mismatches'] or record['physical_bytes'] != sum(sizes):
        raise RuntimeError('layout verification failed: '+name)
    if x_reference:
        coord = folder / 'group-2.bin'
        if base.digest(coord) != base.digest(x_reference):
            raise RuntimeError('independent single graph uses different X bytes')
        link = folder / 'shared-x.bin'
        os.link(str(x_reference), str(link))
        os.replace(str(link), str(coord))
    return folder


def run_search(spec, levels, label, diagnostic=False, trace=False):
    c, folder, kind, prefix, truth, only_x, bypass = spec
    env = dict(LAYOUT_ORDER=c['order'], LAYOUT_SHARED_CACHE_BYTES=str(CACHE),
               LAYOUT_ENDPOINT_X_ONLY=str(int(only_x)))
    if bypass:
        env['LAYOUT_BYPASS_ALPHA'] = '1'
    if diagnostic:
        env.update(LAYOUT_ONE_PASS='1', LAYOUT_GT_DIAGNOSTIC='1',
                   LAYOUT_DUMP=str(ROOT/'analysis'/label))
    if trace:
        if not diagnostic:
            raise ValueError('edge audit requires one-pass diagnostics')
        env['LAYOUT_TRACE'] = str(ROOT/'analysis'/label)
    result = ROOT / 'results' / (label + '.json')
    cmd = [base.ENGINE, 'compare_x', c['meta'], c['topo'], c['coord'], folder, kind, result,
           prefix, c['query1'], c['query2'], 'unused-alpha', truth, 4, base.SECONDS] + levels
    base.status('diagnosing' if diagnostic else 'timing', task=label, levels=levels,
                cpus=monitor.CPUS)
    if diagnostic:
        base.run(cmd, label, env, monitor.CPUS)
    else:
        monitor.monitored_run(cmd, label, env, monitor.CPUS)
    rows = base.read(result)
    if [p['L'] for p in rows] != levels:
        raise RuntimeError('missing L points: '+label)
    for p in rows:
        base.check_point(p, True)
        if p['threads'] != 4 or p['queries'] != 1000 or (not diagnostic and p['seconds'] < base.SECONDS):
            raise RuntimeError('unexpected workload')
        if bypass and p['alpha_rejected_edges']:
            raise RuntimeError('alpha filter bypass was not applied')
        if diagnostic and not p['recall']-1e-8 <= p['gt_candidates_pct'] <= p['gt_discovered_pct']:
            raise RuntimeError('inconsistent ground-truth discovery counters')
    return rows


def levels_for(rows, targets=TARGETS):
    result = set()
    for t in targets:
        pair = base.bracket(rows, t)
        if pair:
            a, b = pair
            result.update((a['L'], b['L']))
            if a['L'] != b['L']:
                result.add(round(a['L']+(b['L']-a['L'])*(t-a['recall'])/(b['recall']-a['recall'])))
    return sorted(result)


def summarize(records, targets):
    matched = []
    for record in records:
        if not record['accepted']:
            continue
        for target in targets:
            pair = base.bracket(record['points'], target)
            if not pair:
                continue
            a, b = pair
            weight = 0 if a['L'] == b['L'] else (target-a['recall'])/(b['recall']-a['recall'])
            near = min(record['points'], key=lambda p: abs(p['recall']-target))
            matched.append(dict(variant=record['variant'], repeat=record['repeat'], target=target,
                                p95_us=near['p95_us'], p95_recall=near['recall'],
                                **{k:a[k]+weight*(b[k]-a[k]) for k in FIELDS+('L',)}))
    base.save(ROOT/'analysis/matched.json', matched)
    lines = ['# Matched Recall@10', '', 'Three accepted repetitions per row. p95 uses the closest measured recall.', '',
             '| Variant | Recall | L | QPS | Mean ms | p95 ms | p95 measured recall | Expanded nodes | PQ space evals | Topology pages | Coordinate pages |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for variant in sorted({r['variant'] for r in matched}):
        for target in targets:
            rows = [r for r in matched if r['variant'] == variant and r['target'] == target]
            if len(rows) != 3:
                continue
            mean = {k: statistics.mean(r[k] for r in rows) for k in rows[0]
                    if k != 'variant'}
            lines.append('| {} | {}% | {:.1f} | {:.0f} | {:.3f} | {:.3f} | {:.2f}% | {:.1f} | {:.1f} | {:.2f} | {:.2f} |'.format(
                variant,target,mean['L'],mean['qps'],mean['latency_us']/1000,mean['p95_us']/1000,
                mean['p95_recall'],mean['expanded_nodes'],mean['pq_space_evals'],
                mean['topology_submitted_pages'],mean['coordinate_submitted_pages']))
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def benchmark(specs, screens, targets=TARGETS):
    measurement = ROOT/'analysis/measurements.json'
    records = base.read(measurement) if measurement.exists() else []
    selected = {v:levels_for(points, targets) for v, points in screens.items()}
    for variant, levels in selected.items():
        if not levels:
            raise RuntimeError('no covered recall target for '+variant)
        missing = sorted(set(levels)-{p['L'] for p in screens[variant]})
        if missing:
            screens[variant].extend(run_search(specs[variant],missing,variant+'-refine',True))
    base.save(ROOT/'analysis/selected.json',selected)
    names = list(specs)
    for repeat in (1,2,3):
        rotation = (repeat-1)%len(names)
        for variant in names[rotation:] + names[:rotation]:
            earlier = [r for r in records if r['variant'] == variant and r['repeat'] == repeat]
            if any(r['accepted'] for r in earlier):
                continue
            for attempt in range(len(earlier)+1, 4):
                label = '{}-r{}-a{}'.format(variant,repeat,attempt)
                monitor.gate(label)
                pts = run_search(specs[variant],selected[variant],label)
                base.compare_logical(pts,{p['L']:p for p in screens[variant]},label)
                quality = [monitor.telemetry(label,p) for p in pts]
                accepted = all(t['quiet'] for t in quality)
                records.append(dict(variant=variant,repeat=repeat,attempt=attempt,accepted=accepted,
                                    points=pts,telemetry=quality))
                base.save(measurement,records)
                summarize(records,targets)
                base.save(ROOT/'analysis/progress.json',dict(accepted=sum(r['accepted'] for r in records),
                          total=len(names)*3,attempts=len(records),task=label,updated_at=base.now()))
                if accepted:
                    break
    return records


def openimages():
    c = next(c for c in base.read(monitor.OLD/'analysis/inputs.json') if c['name'] == 'openimage')
    monitor.verify_inputs(c)
    c = base.prepare_budget(c,'2')
    folder = layout(c,'openimage-m2-x','S_T_X')
    truth = ARCHIVE/'analysis/x-top10-ids.bin'
    archive_protocol = base.read(ARCHIVE/'analysis/protocol.json')
    if base.digest(truth) != archive_protocol['truth_sha256']:
        raise RuntimeError('archived X-only GT changed')
    expected = base.read(ARCHIVE/'analysis/validation.json')['output_hashes']['dual_x']
    specs = {variant:(c,folder,'S_T_X',c['prefix']+'/',truth,True,variant=='bypass')
             for variant in ('filter','bypass')}
    monitor.CPUS = quiet_cores()
    base.save(ROOT/'analysis/protocol.json',dict(dataset='openimage',alpha=1,threads=4,numa=3,
              cpus=monitor.CPUS,cache_bytes=CACHE,topology_sha256=base.digest(folder/'group-1.bin'),
              coordinate_sha256=base.digest(folder/'group-2.bin'),gt_sha256=base.digest(truth)))
    screens = {}
    for variant,spec in specs.items():
        grid = sorted(set(LEVELS)|{int(L) for L in expected})
        screens[variant] = run_search(spec,grid,variant+'-screen',True)
    for L,hashes in expected.items():
        for suffix in ('ids','dists'):
            if base.digest(ROOT/'analysis'/('filter-screen.L'+L+'.'+suffix)) != hashes[suffix]:
                raise RuntimeError('filter-on no longer reproduces archived endpoint')
    if not any(p['alpha_rejected_edges']>0 for p in screens['filter']):
        raise RuntimeError('filter-on did not exercise endpoint rejection')
    x = np.memmap(str(Path(c['query1']).with_name('base_img_emb.fvecs')),mode='r',
                  dtype='<f4',offset=8,shape=(prior.N,prior.DIM))
    q = np.memmap(c['query1'],mode='r',dtype='<f4',offset=8,shape=(1000,prior.DIM)).astype(np.float64)
    gt = np.fromfile(str(truth),dtype='<u4',offset=8).reshape(1000,10)
    bypass_proof = prior.validate_answers(screens['bypass'],'bypass-screen',x,q,gt)
    base.save(ROOT/'analysis/diagnostics.json',screens)
    base.save(ROOT/'analysis/validation.json',dict(status='PASS',archived_filter_output_hashes=expected,bypass_output_hashes=bypass_proof))
    return benchmark(specs,screens)


def main():
    for d in ('analysis','bin','results','scratch','logs'):
        (ROOT/d).mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        tools.build()
        tools.prepare_window()
        records = openimages()
    accepted = sum(r['accepted'] for r in records)
    base.status('complete' if accepted==6 else 'complete_with_flagged_points', accepted=accepted,total=6)


if __name__ == '__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as e:
        base.status('interrupted' if isinstance(e,InterruptedError) else 'failed',error=str(e))
        traceback.print_exc()
        sys.exit(1)
