"""Compare native Baseline and Merge with cache off/on in the same timing batch."""
import json
from pathlib import Path
import statistics
import struct

import followup as common

base, monitor = common.base, common.monitor
EXP, ROOT = common.EXP, common.ROOT/'native'
PHASE = EXP/'io-time-numa3-20261108'
CACHE = EXP/'shared-cache-t4-20261108'
VARIANTS = ('baseline','merge_off','merge_on')
TARGETS = (90,95)


def baseline_binary():
    manifest = base.read(PHASE/'analysis/build.json')
    path = PHASE/'scripts/phase_search.cpp'
    if base.digest(path)!=manifest['sources'][path.name]:
        raise RuntimeError('archived Baseline harness changed')
    source = path.read_text()
    old = 'const size_t pilot ='
    start = source.index(old)
    end = source.index(';',start)
    source = source[:start]+'const size_t pilot = q'+source[end:]
    old = 'latency[i] = s.total_us;'
    if source.count(old)!=1:
        raise RuntimeError('unexpected Baseline latency calculation')
    source = source.replace(old,'latency[i] = service[i];')
    patched = ROOT/'analysis/phase_search_external.cpp'
    patched.write_text(source)
    spec = manifest['commands'][0]
    library = next(a for a in spec if a.endswith('libpipeann.a'))
    if not any(v['path']==library and base.digest(library)==v['sha256']
               for v in manifest['libraries'].values()):
        raise RuntimeError('Baseline search library changed')
    binary = ROOT/'bin/phase-baseline'
    cmd = [str(patched) if a.endswith('/phase_search.cpp') else str(binary)
           if a.endswith('/phase-baseline') else a for a in spec]
    cmd.insert(1,'-I'+str(path.parent))
    base.status('building_baseline_measurement_driver')
    base.run(cmd,'build-baseline')
    base.save(ROOT/'analysis/build.json',dict(binary_sha256=base.digest(binary),
              library_sha256=base.digest(library),baseline_source_sha256=base.digest(patched),
              changes=['full-query warmup','external service percentiles']))
    return binary


def config(dataset, variant, binaries):
    condition = 'mixed' if dataset=='cc3m' else 'fixed-0.5'
    method = 'baseline' if variant=='baseline' else 'merge'
    path = EXP/'concurrency-alpha/results'/('{}__{}__T4__{}__rep1.json'.format(dataset,condition,method))
    raw = base.read(path)
    argv = raw['command'][4:17]
    if argv[:1]!=['float'] or argv[2:4]!=['4','4'] or argv[8:10]!=['10','l2']:
        raise RuntimeError('archived argument mismatch')
    if set(argv[4:8]) != set(raw['input_sha256']):
        raise RuntimeError('archived query manifest mismatch')
    for file,h in raw['input_sha256'].items():
        if base.digest(file)!=h:
            raise RuntimeError('query, alpha, or GT changed: '+file)
    if dataset=='laion':
        weights = Path(argv[6]).read_bytes()
        if struct.unpack_from('<II',weights)!=(1000,1) or set(struct.unpack_from('<1000f',weights,8))!={.5}:
            raise RuntimeError('Laion alpha is not fixed at 0.5')
    files = {}
    for file in Path(argv[1]).iterdir():
        if file.is_file():
            resolved = file.resolve()
            if not str(resolved).startswith('/mnt/nvme3/'):
                raise RuntimeError('index device changed: '+str(resolved))
            st = resolved.stat()
            files[str(file)] = dict(path=str(resolved),size=st.st_size,mtime_ns=st.st_mtime_ns)
    if not files:
        raise RuntimeError('missing native index')
    points = list(raw['points'])
    if dataset=='cc3m':
        high = EXP/'cc3m-highrecall-t4-20261104/results'
        existing = {p['L'] for p in points}
        points.extend(p for p in base.read(high/(method+'__r1__on.json'))['points'] if p['L'] not in existing)
    base.save(ROOT/'analysis'/('{}-{}-inputs.json'.format(dataset,variant)),
              dict(reference_sha256=base.digest(path), inputs=raw['input_sha256'], files=files, argv=argv))
    return dict(argv=argv, binary=binaries[method], points=points, baseline=method=='baseline',
                enabled=variant=='merge_on')


def invoke(case, levels, label, diagnostic=False):
    env = dict(HVS_POINT_SECONDS='10', HVS_SHARED_CACHE_BYTES=str(common.CACHE if case['enabled'] else 0))
    if diagnostic:
        env['HVS_ONE_PASS']='1'
        key = 'PHASE_DUMP_PREFIX' if case['baseline'] else 'HVS_DUMP_PREFIX'
        env[key] = str(ROOT/'analysis'/label)
    cmd = [case['binary']]+case['argv']+levels
    base.status('native_screening' if diagnostic else 'native_timing',task=label,levels=levels,
                cpus=monitor.CPUS)
    if diagnostic:
        base.run(cmd,label,env,monitor.CPUS)
    else:
        monitor.monitored_run(cmd,label,env,monitor.CPUS)
    log = (ROOT/'logs'/(label+'.log')).read_text(errors='replace').splitlines()
    done = 'HVS_IO_PHASE_COMPLETE' if case['baseline'] else 'HVS_CACHE_COMPLETE'
    points = [json.loads(line[10:]) for line in log if line.startswith('HVS_POINT ')]
    if done not in log or [p['L'] for p in points]!=levels:
        raise RuntimeError('native curve incomplete: '+label)
    expected = {p['L']:p for p in case['points']}
    for p in points:
        if p['unique_queries']!=1000 or p['threads']!=4 or (not diagnostic and p['seconds']<10):
            raise RuntimeError('native workload mismatch')
        for k in ('recall','comparisons','topology_ios','rerank_ios'):
            if p['L'] in expected and abs(p[k]-expected[p['L']][k])>1e-7:
                raise RuntimeError('{} changed at {} for {}'.format(k,p['L'],label))
        if case['baseline']:
            p.update(p95_service_us=p['p95_us'],coupled_pages=p['coupled_backend_requests'],
                     topology_pages=0,coord_pages=0,coupled_io_us=p['reader_coupled_us'],
                     topology_io_us=0,coord_io_us=0)
        else:
            if (p['cache_bytes']!=int(env['HVS_SHARED_CACHE_BYTES']) or
                    abs(p['shared_hits']+p['topology_backend_pages']-p['shared_requested'])>1e-7):
                raise RuntimeError('native cache instrumentation mismatch')
            p.update(p95_service_us=p['external_p95_us'],coupled_pages=0,
                     topology_pages=p['topology_backend_pages'],coord_pages=p['coordinate_backend_pages'],
                     coupled_io_us=0,topology_io_us=p['reader_topology_us'],coord_io_us=p['reader_coordinate_us'])
        p['backend_pages'] = p['coupled_pages']+p['topology_pages']+p['coord_pages']
    base.save(ROOT/'results'/(label+'.json'),points)
    return points


def report(records):
    fields = ('L','qps','external_service_us','backend_pages','coupled_pages','topology_pages',
              'coord_pages','coupled_io_us','topology_io_us','coord_io_us')
    matched = []
    for r in records:
        if not r['accepted']:
            continue
        for t in TARGETS:
            pair = base.bracket(r['points'],t)
            if pair is None:
                continue
            a,b = pair
            f = 0 if a['L']==b['L'] else (t-a['recall'])/(b['recall']-a['recall'])
            near = min(r['points'],key=lambda p:abs(p['recall']-t))
            matched.append(dict(dataset=r['dataset'],variant=r['variant'],repeat=r['repeat'],target=t,
                 p95_us=near['p95_service_us'],p95_recall=near['recall'],**{k:a[k]+f*(b[k]-a[k]) for k in fields}))
    base.save(ROOT/'analysis/matched.json',matched)
    lines = ['# Native Baseline and Merge at matched recall', '',
             'Three accepted repeats per row. Service p95 is from the nearest measured recall. Reader wall times and software backend pages.', '',
             '| Dataset | Variant | Recall | L | QPS | Mean ms | p95 ms | Actual p95 recall | Total pages | Topo pages | Coord pages | Coupled I/O ms | Topo I/O ms | Coord I/O ms |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ds in ('cc3m','laion'):
        for t in TARGETS:
            for v in VARIANTS:
                rows = [r for r in matched if (r['dataset'],r['variant'],r['target'])==(ds,v,t)]
                if len(rows)!=3:
                    continue
                def avg(k):
                    return statistics.mean(r[k] for r in rows)
                lines.append('| {} | {} | {}% | {:.1f} | {:.0f} | {:.3f} | {:.3f} | {:.2f}% | {:.2f} | {:.2f} | {:.2f} | {:.3f} | {:.3f} | {:.3f} |'.format(
                    ds,v,t,avg('L'),avg('qps'),avg('external_service_us')/1000,avg('p95_us')/1000,avg('p95_recall'),
                    avg('backend_pages'),avg('topology_pages'),avg('coord_pages'),avg('coupled_io_us')/1000,
                    avg('topology_io_us')/1000,avg('coord_io_us')/1000))
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n')


def experiment():
    for d in ('analysis','bin','results','logs'):
        (ROOT/d).mkdir(exist_ok=True)
    common.ROOT = common.tools.ROOT = base.ROOT = monitor.ROOT = ROOT
    binaries = {'baseline':baseline_binary(),'merge':CACHE/'bin/cache-search'}
    if base.digest(binaries['merge'])!=base.read(CACHE/'analysis/build.json')['binaries']['cache-search']:
        raise RuntimeError('archive cache binary changed')
    base.save(ROOT/'analysis/protocol.json',dict(threads=4,numa=3,repeat=3,beam=4,top_k=10,
              targets=TARGETS,cache_bytes=common.CACHE,merge_binary_sha256=base.digest(binaries['merge'])))
    monitor.CPUS = common.quiet_cores()
    measurement = ROOT/'analysis/measurements.json'
    records = base.read(measurement) if measurement.exists() else []
    for ds in ('cc3m','laion'):
        cases = {v:config(ds,v,binaries) for v in VARIANTS}
        if len({tuple(c['argv'][4:8]) for c in cases.values()})!=1:
            raise RuntimeError('native variants have different query/GT inputs')
        levels = {v:common.levels_for(c['points']) for v,c in cases.items()}
        expected = {}
        parity = {}
        for v in VARIANTS:
            monitor.gate(ds+'-'+v+'-parity')
            label = ds+'-'+v+'-parity'
            expected[v] = invoke(cases[v],levels[v],label,True)
            if not all(base.bracket(expected[v],t) for t in TARGETS):
                raise RuntimeError('archived grid no longer brackets both recalls')
            if v!='baseline':
                parity[v] = {str(L):[base.digest(ROOT/'analysis'/('{}.L{}.{}'.format(label,L,k)))
                                      for k in ('ids','dists')] for L in levels[v]}
        if parity['merge_off']!=parity['merge_on'] or not any(p['shared_hits']>0 for p in expected['merge_on']):
            raise RuntimeError('Merge cache parity or usage check failed')
        base.save(ROOT/'analysis'/(ds+'-parity.json'),dict(status='PASS',merge_output_hashes=parity['merge_on']))
        for rep in (1,2,3):
            rotation = rep-1
            for v in VARIANTS[rotation:]+VARIANTS[:rotation]:
                prev = [r for r in records if (r['dataset'],r['variant'],r['repeat'])==(ds,v,rep)]
                if any(r['accepted'] for r in prev):
                    continue
                for attempt in range(len(prev)+1,4):
                    label = '{}-{}-r{}-a{}'.format(ds,v,rep,attempt)
                    monitor.gate(label)
                    pts = invoke(cases[v],levels[v],label)
                    for p,q in zip(pts,expected[v]):
                        if any(abs(p[k]-q[k])>1e-7 for k in ('recall','comparisons','topology_ios','rerank_ios')):
                            raise RuntimeError('native formal run changed logical work')
                    quality = [monitor.telemetry(label,p) for p in pts]
                    accepted = all(x['quiet'] for x in quality)
                    records.append(dict(dataset=ds,variant=v,repeat=rep,attempt=attempt,
                                   accepted=accepted,points=pts,telemetry=quality))
                    base.save(measurement,records)
                    report(records)
                    base.save(ROOT/'analysis/progress.json',dict(accepted=sum(r['accepted'] for r in records),
                              total=18,attempts=len(records),task=label,updated_at=base.now()))
                    if accepted:
                        break
    accepted = sum(r['accepted'] for r in records)
    base.status('complete' if accepted==18 else 'complete_with_flagged_points',accepted=accepted,total=18)
