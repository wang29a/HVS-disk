"""Serial OpenImages alpha filter, disk format, and backing-medium experiment."""
import fcntl
import os
from pathlib import Path
import shutil
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent/'graph-filter-followup-20261110/scripts'))
import followup as common
import numpy as np

base, monitor, tools = common.base, common.monitor, common.tools
base.ROOT = monitor.ROOT = common.ROOT = tools.ROOT = ROOT
base.ENGINE = ROOT/'bin/factorial-search'
TARGETS = (90,95)
GRID = [20,40,60,80,100,120,160,200,240,320,400,600,800]
PAGE = 8192
TOTAL_CACHE = 256 << 20
common.FIELDS += ('topology_distinct_pages','coordinate_distinct_pages','exact_space_evals',
                  'pq_lookup_us','exact_us','topology_shared_hits')


def build_layout(config, coord_ref=None):
    label = 'openimage-b{}-T_XY'.format(config['budget'])
    folder = ROOT/'scratch'/label
    proof = ROOT/'analysis'/(label+'-identity.json')
    if proof.is_file():
        identity = base.read(proof)
        files = [folder/'group-1.bin',folder/'group-6.bin']
        if ([p.stat().st_size for p in files]!=identity['group_bytes'] or
                identity['metadata_sha256']!=base.digest(config['meta']) or
                identity['topology_sha256']!=base.digest(config['topo']) or
                identity['order_sha256']!=base.digest(config['order']) or
                (coord_ref and files[1].stat().st_ino!=coord_ref.stat().st_ino)):
            raise RuntimeError('existing layout identity mismatch')
        return folder
    if folder.exists():
        shutil.rmtree(str(folder))
    return base.build_layout(config,'T_XY',coord_ref)


def prepare():
    config = next(c for c in base.read(monitor.OLD/'analysis/inputs.json') if c['name']=='openimage')
    monitor.verify_inputs(config)
    for key,expected in config['source_sha256'].items():
        if base.digest(config[key])!=expected:
            raise RuntimeError('Full input changed: '+key)
    full = base.prepare_budget(config,'full')
    tmp = ROOT/'scratch/openimage-b0-convert'
    if tmp.exists():
        conversion = base.read(ROOT/'analysis/openimage-b0-conversion.json')
        if conversion['status']!='PASS' or not conversion['neighbor_ids_and_order_preserved']:
            raise RuntimeError('archived compact conversion was not validated')
        compact = dict(config,budget='0',meta=str(tmp/'meta.bin'),topo=str(tmp/'topo.bin'))
    else:
        compact = base.prepare_budget(config,'0')
    full_layout = build_layout(full)
    compact_layout = build_layout(compact,full_layout/'group-6.bin')
    n, _, _, degree, slots, npp = struct.unpack_from('<5IQ',Path(full['meta']).read_bytes())
    source = np.memmap(full['topo'],mode='r',dtype=np.uint8)
    width = 4+degree*(4+2*slots)
    sidecar = ROOT/'scratch/full-alpha-sidecar.bin'
    sidecar_tmp = sidecar.with_suffix('.partial')
    groups = (n+npp-1)//npp
    # Row order follows original node ID. Page padding is omitted.
    paged = np.ndarray((groups,npp,degree,2*slots),dtype=np.uint8,buffer=source,offset=8,
                       strides=(PAGE,width,4+2*slots,1))
    with sidecar_tmp.open('wb') as out:
        for start in range(0,groups,2048):
            batch = paged[start:start+2048].reshape(-1,degree,2*slots)
            batch[:n-start*npp].tofile(out)
        out.flush()
        os.fsync(out.fileno())
    os.replace(str(sidecar_tmp),str(sidecar))
    expected_size = n*degree*2*slots
    if sidecar.stat().st_size!=expected_size:
        raise RuntimeError('wrong sidecar width')
    base.save(ROOT/'analysis/protocol.json',dict(
        dataset='OpenImages', layout='T_XY',alpha='mixed',n=n,degree=degree,full_slots=slots,
        threads=4,repeats=3,seconds_per_point=base.SECONDS,cache_off=0,
        equal_extra_memory_bytes=TOTAL_CACHE,sidecar_bytes=expected_size,
        sidecar_sha256=base.digest(sidecar),source=config,
        compact_identity_sha256=base.digest(compact['topo']),shared_coordinate_file=True))
    return full,compact,full_layout,compact_layout,sidecar,slots


def run_case(spec,levels,label,diagnostic=False,trace=False,screening=False):
    config, folder, bypass, sidecar, slots, dram, cache = spec
    env = dict(LAYOUT_ORDER=config['order'],LAYOUT_SHARED_CACHE_BYTES=str(cache))
    if bypass:
        env['LAYOUT_BYPASS_ALPHA']='1'
    if sidecar:
        env.update(LAYOUT_ALPHA_SIDECAR=str(sidecar),LAYOUT_ALPHA_SLOTS=str(slots))
    if dram:
        env['LAYOUT_DRAM']='1'
    if screening and (not diagnostic or trace):
        raise ValueError('screening requires untraced one-pass diagnostics')
    if diagnostic:
        env.update(LAYOUT_ONE_PASS='1',LAYOUT_GT_DIAGNOSTIC='1')
    if trace:
        env.update(LAYOUT_TRACE=str(ROOT/'analysis'/(label+'.trace')),
                   LAYOUT_DUMP=str(ROOT/'analysis'/label))
    result = ROOT/'results'/(label+'.json')
    command = [base.ENGINE,'query',config['meta'],config['topo'],config['coord'],folder,'T_XY',result,
               config['prefix']+'/',config['query1'],config['query2'],config['alpha'],config['gt'],
               4,base.SECONDS]+levels
    base.status('diagnosing' if diagnostic else 'timing',task=label,levels=levels,cpus=monitor.CPUS)
    if screening and result.is_file():
        saved = base.read(ROOT/'analysis'/(label+'-run.json'))
        if (saved['command'][-len(command):]!=list(map(str,command))
                or saved['environment']!=env):
            raise RuntimeError('archived screening inputs changed')
    elif diagnostic:
        base.run(command,label,env,monitor.CPUS)
    else:
        monitor.monitored_run(command,label,env,monitor.CPUS)
    points = base.read(result)
    if [p['L'] for p in points]!=levels:
        raise RuntimeError('incomplete search curve')
    for p in points:
        base.check_point(p,bool(cache))
        if (p['queries']!=1000 or p['threads']!=4 or (p['cache_evictions'] and not screening)
                or (not diagnostic and p['seconds']<base.SECONDS)):
            raise RuntimeError('unexpected workload or query cache eviction')
        if diagnostic and not p['recall']-1e-8 <= p['gt_candidates_pct'] <= p['gt_discovered_pct']:
            raise RuntimeError('ground truth counters disagree')
        if (bypass or (not sidecar and 'b0' in folder.name)) and p['alpha_rejected_edges']:
            raise RuntimeError('filter not bypassed')
    if screening:
        base.save(ROOT/'analysis'/(label+'-excluded.json'),
                  [dict(L=p['L'],cache_evictions=p['cache_evictions'])
                   for p in points if p['cache_evictions']])
        points = [p for p in points if not p['cache_evictions']]
    return points


def output_hashes(label,levels):
    paths = {}
    for level in levels:
        prefix = ROOT/'analysis'/label
        for suffix in ('ids','dists'):
            paths[(level,suffix)] = Path(str(prefix)+'.L{}.{}'.format(level,suffix))
        paths[(level,'trace')] = Path(str(prefix)+'.trace.L{}'.format(level))
    return {'{}:{}'.format(*key):base.digest(path) for key,path in paths.items()}


def run():
    full,compact,ff,cf,sidecar,slots = prepare()
    specs = {
        'A': (full,ff,False,None,0,False,0),
        'B': (full,ff,True,None,0,False,0),
        'C': (compact,cf,False,None,0,False,0),
        'D': (compact,cf,False,sidecar,slots,False,0),
        'A_dram': (full,ff,False,None,0,True,0),
        'B_dram': (full,ff,True,None,0,True,0),
    }
    extra = TOTAL_CACHE-sidecar.stat().st_size
    for variant in ('A','B','C','D'):
        cache = extra//PAGE*PAGE if variant=='D' else TOTAL_CACHE
        specs[variant+'_cache'] = specs[variant][:-1]+(cache,)
    monitor.CPUS = common.quiet_cores()
    proof, expected = {}, {}
    if '--resume' in sys.argv:
        levels = base.read(ROOT/'analysis/selected.json')
        archived = base.read(ROOT/'analysis/parity.json')
        if archived['status']!='PASS':
            raise RuntimeError('resume requires validated parity')
        for variant in specs:
            points = base.read(ROOT/'results'/(variant+'-parity.json'))
            expected[variant] = {p['L']:p for p in points}
            proof[variant] = output_hashes(variant+'-parity',levels[variant])
        if proof!=archived['hashes']:
            raise RuntimeError('archived parity files changed')
    else:
        selected = {}
        for variant in ('A','B'):
            points = run_case(specs[variant],GRID,variant+'-screen',True)
            if not all(base.bracket(points,t) for t in TARGETS):
                raise RuntimeError('OpenImages target recall uncovered on '+variant)
            selected[variant] = common.levels_for(points,TARGETS)
        levels = {variant:selected['A' if variant[0] in ('A','D') else 'B'] for variant in specs}
        base.save(ROOT/'analysis/selected.json',levels)
        for variant,spec in specs.items():
            label = variant+'-parity'
            points = run_case(spec,levels[variant],label,True,True)
            expected[variant] = {p['L']:p for p in points}
            proof[variant] = output_hashes(label,levels[variant])
    for a,b in (('A','D'),('B','C'),('A','A_dram'),('B','B_dram')) + tuple(
            (v,v+'_cache') for v in ('A','B','C','D')):
        if proof[a]!=proof[b]:
            raise RuntimeError('result or expansion trace parity failed: '+a+'/'+b)
        base.compare_logical(list(expected[a].values()),expected[b],a+'/'+b)
    if not any(p['alpha_rejected_edges'] for p in expected['A'].values()):
        raise RuntimeError('filter-on never rejected an edge')
    base.save(ROOT/'analysis/parity.json',dict(status='PASS',hashes=proof))

    measurement = ROOT/'analysis/measurements.json'
    records = base.read(measurement) if measurement.exists() else []
    groups = [('A','B','C','D','A_dram','B_dram'),
              ('A_cache','B_cache','C_cache','D_cache')]
    for group in groups:
        for rep in (1,2,3):
            rotation = rep-1
            for v in group[rotation:]+group[:rotation]:
                old = [r for r in records if r['variant']==v and r['repeat']==rep]
                if any(r['accepted'] for r in old):
                    continue
                for attempt in range(len(old)+1,len(old)+4):
                    label = '{}-r{}-a{}'.format(v,rep,attempt)
                    monitor.gate(label)
                    pts = run_case(specs[v],levels[v],label)
                    base.compare_logical(pts,expected[v],label)
                    quality = [monitor.telemetry(label,p) for p in pts]
                    accepted = all(t['quiet'] for t in quality)
                    records.append(dict(variant=v,repeat=rep,attempt=attempt,accepted=accepted,
                                        points=pts,telemetry=quality))
                    base.save(measurement,records)
                    common.summarize(records,TARGETS)
                    base.save(ROOT/'analysis/progress.json',dict(accepted=sum(r['accepted'] for r in records),
                              total=len(specs)*3,attempts=len(records),task=label,updated_at=base.now()))
                    if accepted:
                        break
    accepted = sum(r['accepted'] for r in records)
    base.status('complete' if accepted==len(specs)*3 else 'complete_with_flagged_points',
                accepted=accepted,total=len(specs)*3)


def main():
    for name in ('analysis','bin','results','scratch','logs'):
        (ROOT/name).mkdir(exist_ok=True)
    base.status('waiting_for_serial_lock')
    with (ROOT/'queue.lock').open('a') as own, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(shared,fcntl.LOCK_EX)
        if '--resume' in sys.argv:
            if base.digest(base.ENGINE)!=base.read(ROOT/'analysis/build.json')['binary_sha256']:
                raise RuntimeError('resume binary changed')
        else:
            tools.build()
        tools.prepare_window()
        run()


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        base.status('failed',error=str(exc))
        traceback.print_exc()
        sys.exit(1)
