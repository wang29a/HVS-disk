"""Serial NUMA 3 layout-cache ablation followed by the budget scan."""
import csv
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import subprocess
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
EXP = BUILD / 'experiments'
PREV = EXP / 'budget-reorder-20261103'
FIRST = EXP / 'layout-preexperiments-20261102'
ENGINE = ROOT / 'bin/layout-engine'
CONVERTER = ROOT / 'bin/convert-budget'
DATASETS = ('openimage', 'howto100m')
LAYOUTS = ('TXY', 'T_XY')
BUDGETS = ('0', '1', '2', '3', 'full')
TARGETS = (90, 95)
CACHE = 256 << 20
SECONDS = 10
CPU_PAIRS = [(cpu, cpu + 96) for cpu in range(72, 96)]
ENV = {k: v for k, v in os.environ.items() if not k.startswith('LAYOUT_')}
ENV.update(LC_ALL='C', OMP_NUM_THREADS='4', OMP_THREAD_LIMIT='4', OMP_DYNAMIC='FALSE',
           OMP_NESTED='FALSE', OMP_MAX_ACTIVE_LEVELS='1', OMP_PROC_BIND='close',
           OMP_PLACES='cores', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')
LOGICAL = ('recall', 'inspected_edges', 'alpha_rejected_edges', 'duplicate_rejected_edges',
           'pq_comparisons', 'expanded_nodes', 'candidate_insertions', 'reranked_nodes')
METRICS = ('qps', 'latency_us', 'p95_us', 'p99_us', 'submitted_pages',
           'topology_submitted_pages', 'coordinate_submitted_pages', 'topology_submit_wait_us',
           'coordinate_submit_wait_us', 'topology_shared_cache_us', 'coordinate_shared_cache_us',
           'topology_page_hits', 'coordinate_page_hits', 'topology_shared_hits',
           'coordinate_shared_hits', 'inspected_edges', 'alpha_rejected_edges', 'pq_comparisons')


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    os.replace(str(tmp), str(path))


def digest(path):
    h = hashlib.sha256()
    with open(str(path), 'rb') as f:
        for chunk in iter(lambda: f.read(8 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def status(state, **fields):
    save(ROOT / 'status.json', dict(status=state, updated_at=now(), **fields))
    print(now(), state, fields, flush=True)


def disk_cpu_sample():
    def counters():
        values = {}
        for line in Path('/proc/stat').read_text().splitlines():
            cols = line.split()
            if cols[0].startswith('cpu') and cols[0][3:].isdigit():
                values[int(cols[0][3:])] = list(map(int, cols[1:9]))
        disk = next(line.split() for line in Path('/proc/diskstats').read_text().splitlines()
                    if line.split()[2] == 'nvme3n1')
        return values, int(disk[12])
    start = time.monotonic()
    a, d0 = counters()
    time.sleep(5)
    b, d1 = counters()
    cpu = {}
    for pair in CPU_PAIRS:
        for cpu_id in pair:
            delta = [y-x for x,y in zip(a[cpu_id], b[cpu_id])]
            total = sum(delta)
            cpu[cpu_id] = 100 * (total - delta[3]) / total
    return cpu, (d1 - d0) / (10 * (time.monotonic() - start))


def timing_window(label, fixed_cores=None):
    # I/O wait counts as busy here. Check both SMT siblings for each core.
    while True:
        cpu, disk = disk_cpu_sample()
        loads = {a: max(cpu[a], cpu[b]) for a, b in CPU_PAIRS}
        ordered = sorted(loads, key=loads.get)
        candidate = list(fixed_cores) if fixed_cores else sorted(ordered[:4])
        sample = dict(at=now(), task=label, cpu_pair_max_pct=loads, nvme3_busy_pct=disk, cpus=candidate)
        with (ROOT / 'logs/entry-load.jsonl').open('a') as f:
            f.write(json.dumps(sample) + '\n')
        if disk <= 25 and max(loads[c] for c in candidate) <= 15:
            return candidate
        status('waiting_for_quiet_timing_window', task=label,
               nvme3_busy_pct=round(disk, 1), max_selected_cpu_pct=round(max(loads[c] for c in candidate), 1),
               candidate_cpus=candidate)
        time.sleep(25)


def run(cmd, label, env=None, cpus=None):
    cmd = [str(v) for v in cmd]
    if cpus:
        cmd = ['taskset', '-c', ','.join(map(str, cpus))] + cmd
    print(now(), 'RUN', label, flush=True)
    started = time.time()
    with (ROOT / 'logs' / (label + '.log')).open('w') as log:
        worker = subprocess.Popen(cmd, env=dict(ENV, **(env or {})), stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = worker.wait()
        except BaseException:
            try:
                os.killpg(worker.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            raise
    if code:
        raise RuntimeError('{} exit {}'.format(label, code))
    save(ROOT / 'analysis' / (label + '-run.json'),
         dict(command=cmd, environment=env or {}, started_unix=started, finished_unix=time.time()))
    print(now(), 'DONE', label, flush=True)


def build():
    status('building')
    spec = read(EXP / 'concurrency-alpha/analysis/build.json')['separated']
    if digest(spec['library']) != spec['library_sha256']:
        raise RuntimeError('native library hash changed')
    scripts = ROOT / 'scripts'
    cpp = sorted(p for p in scripts.iterdir() if p.suffix in ('.cpp', '.h'))
    run(['clang-format', '-i', '-style=LLVM'] + cpp, 'format-code')
    run(['g++-11'] + spec['flags'] + [scripts/'layout_engine.cpp', spec['library'],
         '-L/mnt/nvme2/wz/disk/local/openblas/lib', '-laio', '-ljemalloc', '-lopenblas', '-o', ENGINE],
        'build-layout-engine')
    run(['g++-11','-std=c++17','-O3',scripts/'convert_budget.cpp','-laio','-o',CONVERTER], 'build-converter')
    test = ROOT / 'bin/test-layout-io'
    run(['g++-11','-std=c++17','-O2','-fopenmp',scripts/'test_layout_io.cpp','-laio','-o',test], 'build-io-test')
    save(ROOT / 'analysis/build.json', dict(source_commit='d2a39e2', library=spec['library'],
         library_sha256=spec['library_sha256'], sources={p.name:digest(p) for p in scripts.iterdir() if p.is_file()},
         binaries={p.name:digest(p) for p in (ENGINE, CONVERTER, test)}))


def synthetic():
    folder = ROOT / 'scratch/synthetic'
    folder.mkdir()
    n, d1, d2, degree, slots = 509, 16, 32, 4, 4
    width = 4 + degree * (4 + 2*slots)
    meta = bytearray(8192)
    struct.pack_into('<IIIIIQII', meta, 0, n, d1, d2, degree, slots, 8192//width, 1, 0)
    tn, cn = 8192//width, 8192//(4*(d1+d2))
    topo = bytearray(((n+tn-1)//tn)*8192)
    coord = bytearray(((n+cn-1)//cn)*8192)
    for i in range(n):
        pos = (i//tn)*8192+(i%tn)*width
        struct.pack_into('<II', topo, pos, 1, (i+1)%n)
        topo[pos+8:pos+16] = bytes((0,10,20,30,40,50,70,100))
        struct.pack_into('<48f', coord, (i//cn)*8192+(i%cn)*192, *[float(i+j) for j in range(48)])
    for name, contents in (('meta',meta),('topo',topo),('coord',coord)):
        (folder/name).write_bytes(contents)
    # Give the fixture more than eight physical pages, including unequal vector dimensions.
    for budget in (1, 2, 3):
        result = folder / ('conversion-{}.json'.format(budget))
        files = [folder/('meta'+str(budget)), folder/('topo'+str(budget)), folder/'coord']
        run([CONVERTER, folder/'meta', folder/'topo', folder/'coord'] + files[:2] + [budget, result],
            'test-convert-{}'.format(budget))
        if read(result)['status'] != 'PASS':
            raise RuntimeError('fixture conversion failed')
        for layout in LAYOUTS:
            label = 'test-b{}-{}'.format(budget, layout)
            out = folder / label
            out.mkdir()
            run([ENGINE,'build'] + files + [out,layout,folder/'build.json'], label+'-build')
            run([ROOT/'bin/test-layout-io'] + files + [out,layout], label+'-shared-io')
    shutil.rmtree(str(folder))
    save(ROOT / 'analysis/tests.json', dict(status='PASS', budgets=[1,2,3], layouts=LAYOUTS,
         topology_only_population=True, shared_rerank_lookup=True, four_thread_reads=True))


def configs():
    selected = []
    for c in read(FIRST / 'analysis/inputs.json'):
        ds = c['name']
        if ds not in DATASETS:
            continue
        folder = PREV / 'inputs' / ds / 'b2'
        old = read(PREV / 'analysis' / (ds + '__b2-index-identity.json'))
        source = read(PREV / 'analysis' / (ds + '-source-identity.json'))['hashes']
        for name, value in dict(source, **c['input_sha256']).items():
            if digest(c[name]) != value:
                raise RuntimeError('source changed: ' + c[name])
        for name in ('meta', 'topo', 'order'):
            if digest(folder / (name + '.bin')) != old[name + '_sha256']:
                raise RuntimeError('archived b2 changed: ' + ds + ' ' + name)
        c.update(order=str(folder/'order.bin'), source_sha256=source, b2_identity=old)
        selected.append(c)
    if [c['name'] for c in selected] != list(DATASETS):
        raise RuntimeError('dataset inventory')
    save(ROOT / 'analysis/inputs.json', selected)
    return selected


def layout_bytes(c, layout):
    n, x, y, degree, slots = struct.unpack_from('<5I', Path(c['meta']).read_bytes())
    topo, coord = 4 + degree*(4+2*slots), 4*(x+y)
    widths = (topo+coord,) if layout == 'TXY' else (topo,coord)
    pages = [(n+(8192//w)-1)//(8192//w) for w in widths]
    return dict(nodes=n, degree=degree, slots=slots, topo_record_bytes=topo,
                group_bytes=[8192*p for p in pages], nodes_per_page=[8192//w for w in widths])


def require_space(extra):
    if shutil.disk_usage(str(ROOT)).free < extra + (24 << 30):
        raise RuntimeError('nvme3 space gate: less than 24 GiB headroom after expected writes')


def prepare_budget(base, budget):
    ds = base['name']
    c = dict(base, budget=budget)
    if budget == '2':
        folder = Path(base['order']).parent
        c.update(meta=str(folder/'meta.bin'), topo=str(folder/'topo.bin'))
    elif budget != 'full':
        folder = ROOT / 'scratch' / (ds + '-b' + budget + '-convert')
        folder.mkdir()
        out = ROOT / 'analysis' / (ds + '-b' + budget + '-conversion.json')
        n, _, _, degree, _ = struct.unpack_from('<5I', Path(base['meta']).read_bytes())
        npp = 8192//(4+degree*(4+2*int(budget)))
        require_space(((n+npp-1)//npp)*8192)
        run([CONVERTER, base['meta'], base['topo'], base['coord'], folder/'meta.bin',
             folder/'topo.bin', budget, out], ds+'-b'+budget+'-convert')
        result = read(out)
        if (result['status'] != 'PASS' or not result['neighbor_ids_and_order_preserved']
                or not result['original_activity_contained'] or result['budget'] != int(budget)):
            raise RuntimeError('conversion validation failed')
        c.update(meta=str(folder/'meta.bin'), topo=str(folder/'topo.bin'))
    return c


def build_layout(c, layout, coord_ref=None):
    label = '{}-b{}-{}'.format(c['name'], c['budget'], layout)
    folder = ROOT / 'scratch' / label
    identity = layout_bytes(c, layout)
    require_space(sum(identity['group_bytes'][:1]) if coord_ref else sum(identity['group_bytes']))
    folder.mkdir()
    env = {'LAYOUT_ORDER':c['order']}
    if coord_ref:
        os.link(str(coord_ref), str(folder/'group-6.bin'))
        env['LAYOUT_REUSE_COORD'] = '1'
    output = ROOT / 'analysis' / (label + '-layout.json')
    status('building_layout', task=label, expected_new_bytes=(identity['group_bytes'][0] if coord_ref
                                                               else sum(identity['group_bytes'])))
    run([ENGINE,'build',c['meta'],c['topo'],c['coord'],folder,layout,output], label+'-layout', env)
    result = read(output)
    if result['status'] != 'PASS' or result['record_mismatches'] or result['physical_bytes'] != sum(identity['group_bytes']):
        raise RuntimeError('layout validation failed')
    save(ROOT/'analysis'/(label+'-identity.json'), dict(identity, layout=layout, coord_reused=bool(coord_ref),
         order_sha256=c['b2_identity']['order_sha256'], metadata_sha256=digest(c['meta']), topology_sha256=digest(c['topo'])))
    return folder


def check_point(p, enabled):
    hits = p['topology_shared_hits'] + p['coordinate_shared_hits']
    if not enabled and hits:
        raise RuntimeError('shared hits with cache off')
    if abs(p['requested_pages'] - p['page_hits'] - hits - p['submitted_pages']) > 1e-6:
        raise RuntimeError('page accounting mismatch')
    if abs(p['submitted_pages'] - p['topology_submitted_pages'] - p['coordinate_submitted_pages']) > 1e-6:
        raise RuntimeError('phase page accounting mismatch')


def query(c, layout, folder, enabled, levels, label, one=False, dump=False, cpus=None):
    out = ROOT / 'results' / (label+'.json')
    env = dict(LAYOUT_ORDER=c['order'], LAYOUT_SHARED_CACHE_BYTES=str(CACHE if enabled else 0))
    if one:
        env['LAYOUT_ONE_PASS'] = '1'
    if dump:
        env['LAYOUT_DUMP'] = str(ROOT/'analysis'/label)
    status('screening' if one else 'timing', task=label, levels=levels, shared_cache=enabled, cpus=cpus)
    cmd = [ENGINE,'query',c['meta'],c['topo'],c['coord'],folder,layout,out,
           c['prefix']+'/',c['query1'],c['query2'],c['alpha'],c['gt'],4,SECONDS]+levels
    run(cmd,label,env,cpus)
    pts = read(out)
    if [p['L'] for p in pts] != levels:
        raise RuntimeError('missing L result')
    for p in pts:
        check_point(p, enabled)
        if layout=='T_XY' and p['coordinate_shared_hits']:
            raise RuntimeError('separated coordinates entered the shared cache')
        if p['threads'] != 4 or p['queries'] != 1000 or (not one and p['seconds'] < SECONDS):
            raise RuntimeError('wrong measurement duration or workload')
    return pts


def bracket(rows, target):
    rows = sorted(rows, key=lambda p:p['L'])
    for p in rows:
        if p['recall'] == target:
            return (p,p)
    for lo, hi in zip(rows, rows[1:]):
        if lo['recall'] < target < hi['recall']:
            return lo,hi
    return None


def screen(c, folder, enabled, label):
    archive = read(PREV/'results'/(c['name']+'__b2__starling__T_XY__r1__default__main__t1.json'))
    levels = sorted({p['L'] for target in TARGETS for p in bracket(archive, target)})
    rows = query(c, 'T_XY', folder, enabled, levels, label+'-seed', one=True)
    if not all(bracket(rows, t) for t in TARGETS):
        grid = [20,40,60,80,100,120,160,200,240,300,400,500,600,800,1200,1600]
        rows = query(c, 'T_XY', folder, enabled, grid, label+'-wide', one=True)
    pairs = [bracket(rows, t) for t in TARGETS]
    if not all(pairs):
        raise RuntimeError('target recall not covered: ' + label)
    initial = {p['L'] for pair in pairs for p in pair}
    refine = set()
    for t,(lo,hi) in zip(TARGETS,pairs):
        if lo['L'] != hi['L']:
            estimate = lo['L'] + (hi['L']-lo['L'])*(t-lo['recall'])/(hi['recall']-lo['recall'])
            refine.add(round(estimate))
    add = sorted(refine - initial)
    if add:
        rows.extend(query(c, 'T_XY', folder, enabled, add, label+'-refine', one=True))
    selected = sorted(initial | refine)
    points = {p['L']:p for p in rows if p['L'] in selected}
    save(ROOT/'analysis'/(label+'-levels.json'), dict(levels=selected, recall={L:p['recall'] for L,p in points.items()}))
    return selected, points


def compare_logical(points, expected, label):
    for p in points:
        for k in LOGICAL:
            if abs(p[k] - expected[p['L']][k]) > 1e-7:
                raise RuntimeError('{} L={} changed {}'.format(label, p['L'], k))


def parity(c, layouts, levels, expected):
    reference = None
    for layout in LAYOUTS:
        for enabled in (False,True):
            label = '{}-parity-{}-{}'.format(c['name'],layout,'on' if enabled else 'off')
            points = query(c, layout, layouts[layout], enabled, levels, label, one=True, dump=True)
            compare_logical(points, expected, label)
            if enabled and not any(p['topology_shared_hits'] > 0 for p in points):
                raise RuntimeError('shared cache never hit')
            outputs = {}
            for L in levels:
                for ext in ('ids','dists'):
                    path = ROOT/'analysis'/('{}.L{}.{}'.format(label,L,ext))
                    if path.stat().st_size != 40000:
                        raise RuntimeError('invalid parity dump size')
                    outputs['{}.{}'.format(L,ext)] = digest(path)
            if reference is None:
                reference = outputs
            elif reference != outputs:
                raise RuntimeError('IDs or distances differ across four cells')
    save(ROOT/'analysis'/(c['name']+'-four-cell-parity.json'),dict(status='PASS', levels=levels, hashes=reference))


def matched(points):
    values = {}
    for target in TARGETS:
        lo,hi = bracket(points,target)
        f = 0 if lo['L']==hi['L'] else (target-lo['recall'])/(hi['recall']-lo['recall'])
        near = min(points,key=lambda p:abs(p['recall']-target))
        values[target] = dict(L=lo['L']+f*(hi['L']-lo['L']), p95_L=near['L'],
            p95_measured_recall=near['recall'], p95_us=near['p95_us'],
            **{k:lo[k]+f*(hi[k]-lo[k]) for k in METRICS if k not in ('p95_us','p99_us')})
    return values


def report(records):
    rows = []
    for r in records:
        for target, values in matched(r['points']).items():
            rows.append(dict(dataset=r['dataset'], stage=r['stage'], budget=r['budget'], layout=r['layout'],
                             cache=r['cache'], repeat=r['repeat'], target=target, **values))
    save(ROOT/'analysis/summary.json', dict(runs=len(records), points=sum(len(r['points']) for r in records), matched=rows))
    with (ROOT/'analysis/matched.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = ['# Layout-cache and budget, four search threads', '',
             'QPS and reader metrics interpolate within each repeat. p95 uses the nearest measured L; consult matched.csv for the actual recall.',
             'Page counts are reader submissions. Only conditions with three completed repeats appear.', '',
             '| Dataset | Stage | Budget | Layout | Cache | Recall | QPS | p95 (ms) | Pages / query |',
             '|---|---|---|---|---|---:|---:|---:|---:|']
    keys = ('dataset','stage','budget','layout','cache','target')
    groups = {}
    for row in rows:
        key = tuple(row[k] for k in keys)
        groups.setdefault(key,[]).append(row)
    for key, group in sorted(groups.items()):
        if len(group)==3:
            qps = statistics.mean(p['qps'] for p in group)
            tail = statistics.mean(p['p95_us'] for p in group)/1000
            pages = statistics.mean(p['submitted_pages'] for p in group)
            lines.append('| {} | {} | {} | {} | {} | {}% | {:.1f} | {:.2f} | {:.2f} |'.format(
                *(list(key)+[qps,tail,pages])))
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def formal(c, layout, folder, enabled, levels, expected, stage, rep, cpus, records):
    ds, b = c['name'], c['budget']
    mode = 'on' if enabled else 'off'
    label = '{}-{}-b{}-{}-{}-r{}'.format(stage,ds,b,layout,mode,rep)
    cpus = timing_window(label, cpus)
    points = query(c,layout,folder,enabled,levels,label,cpus=cpus)
    compare_logical(points,expected,label)
    if enabled and not any(p['topology_shared_hits'] > 0 for p in points):
        raise RuntimeError('timed shared cache never hit')
    records.append(dict(dataset=ds,stage=stage,budget=b,layout=layout,cache=mode,repeat=rep,points=points))
    report(records)
    save(ROOT/'analysis/progress.json',dict(complete_curves=len(records),total_curves=54,task=label,cpus=cpus,updated_at=now()))
    return cpus


def experiment():
    c_list = configs()
    retained = {}
    records = []
    cpus = None
    orders = ((('TXY',False),('TXY',True),('T_XY',True),('T_XY',False)),
              (('T_XY',False),('T_XY',True),('TXY',True),('TXY',False)),
              (('TXY',True),('TXY',False),('T_XY',False),('T_XY',True)))
    for base in c_list:
        c = prepare_budget(base,'2')
        layouts = {layout:build_layout(c,layout) for layout in LAYOUTS}
        levels, expected = screen(c,layouts['T_XY'],False,c['name']+'-four-cell')
        parity(c,layouts,levels,expected)
        for rep, order in enumerate(orders,1):
            for layout, enabled in order:
                cpus = formal(c,layout,layouts[layout],enabled,levels,expected,'four-cell',rep,cpus,records)
        shutil.rmtree(str(layouts['TXY']))
        retained[c['name']] = (c,layouts['T_XY'],levels,expected)

    for base in c_list:
        c2, folder2, levels2, expected2 = retained[base['name']]
        prepared = {'2':(c2,folder2,levels2,expected2)}
        for b in BUDGETS:
            if b == '2':
                continue
            c = prepare_budget(base,b)
            folder = build_layout(c,'T_XY',coord_ref=folder2/'group-6.bin')
            levels, expected = screen(c,folder,True,'{}-budget-b{}'.format(c['name'],b))
            prepared[b] = c,folder,levels,expected
        for rep in range(1,4):
            shift = 2*(rep-1)
            for b in BUDGETS[shift:]+BUDGETS[:shift]:
                c, folder, levels, expected = prepared[b]
                cpus = formal(c,'T_XY',folder,True,levels,expected,'budget',rep,cpus,records)
        for _,folder,_,_ in prepared.values():
            shutil.rmtree(str(folder))
        for folder in (ROOT/'scratch').glob(base['name']+'-b*-convert'):
            shutil.rmtree(str(folder))
        for name, value in base['source_sha256'].items():
            if digest(base[name]) != value:
                raise RuntimeError('source changed during experiment: '+base[name])

    return len(records), sum(len(r['points']) for r in records)


def main():
    if set(os.sched_getaffinity(0)) != {cpu for pair in CPU_PAIRS for cpu in pair}:
        raise RuntimeError('expected NUMA 3 CPU mask')
    for directory in ('analysis','bin','results','scratch'):
        (ROOT/directory).mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock, (PREV/'queue.lock').open('r') as shared_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status('waiting_for_serial_lock')
        fcntl.flock(shared_lock,fcntl.LOCK_EX)
        save(ROOT/'analysis/protocol.json',dict(datasets=DATASETS,layouts=LAYOUTS,budgets=BUDGETS,
             threads=4,cpu_node=3,memory_node=3,cache_bytes=CACHE,repeat=3,seconds=SECONDS,
             shared_cache_policy='physical LRU, topology populate, co-located rerank lookup',
             cross_budget_order='fixed archived b2 Starling permutation', budget_cache='on',
             quiet_gate=dict(cpu_each_smt_sibling_max_pct=15,nvme3_max_pct=25,sample_seconds=5),
             warmup_queries=1000,targets=TARGETS,new_gt=False,curves=54))
        build()
        synthetic()
        with (ROOT/'logs/iostat.log').open('w') as log:
            monitor = subprocess.Popen(['iostat','-x','-t','-y','5'],env=ENV,stdout=log,stderr=subprocess.STDOUT)
            try:
                curves,points=experiment()
            finally:
                if monitor.poll() is None:
                    monitor.terminate()
                monitor.wait()
    status('complete', curves=curves, points=points, serial_lock_released=True)


def stop(signum, frame):
    raise InterruptedError('signal {}'.format(signum))


if __name__ == '__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,stop)
    try:
        main()
    except Exception as error:
        status('interrupted' if isinstance(error,InterruptedError) else 'failed',error=str(error))
        traceback.print_exc()
        raise SystemExit(1)
