"""Fixed-output decomposition of second-space compute and coordinate storage."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
RETEST = ROOT.parent/'layout-cache-retest-20261110'
sys.path.insert(0, str(RETEST/'scripts'))
import retest as monitor
base = monitor.base
monitor.ROOT = base.ROOT = ROOT
base.ENGINE = ROOT/'bin/endpoint-search'
LEVELS = [40,80,176]
CACHE = 256 << 20
THREADS = 4
VARIANTS = {'A':('T_XY',False), 'B':('T_XY',True), 'C':('S_T_X',True)}
METRICS = ('qps','latency_us','p95_us','prep_us','pq_lookup_us','exact_us','pq_space_evals',
           'exact_space_evals','coordinate_submitted_pages','coordinate_submit_wait_us',
           'topology_submitted_pages','topology_submit_wait_us','topology_shared_hits')


def build():
    base.status('building')
    spec = base.read(base.EXP/'concurrency-alpha/analysis/build.json')['separated']
    if base.digest(spec['library']) != spec['library_sha256']:
        raise RuntimeError('native library hash mismatch')
    sources = [ROOT/'scripts'/name for name in ('layout_engine.cpp','layout_store.h')]
    base.run(['clang-format','-i','-style=LLVM']+sources,'format')
    base.run(['g++-11']+spec['flags']+[sources[0],spec['library'],
             '-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas',
             '-o',base.ENGINE],'compile')
    base.save(ROOT/'analysis/build.json',dict(library_sha256=spec['library_sha256'],
              binary_sha256=base.digest(base.ENGINE), sources={p.name:base.digest(p) for p in sources}))


def choose_cpus():
    previous = [74,75,76,77]
    last = set()
    while True:
        cpu,disk_pct = base.disk_cpu_sample()
        loads={a:max(cpu[a],cpu[b]) for a,b in base.CPU_PAIRS}
        quiet={c for c in loads if loads[c]<=15} if disk_pct<=25 else set()
        stable=sorted(quiet & last,key=loads.get)
        if len(stable)>=4:
            return previous if all(c in stable for c in previous) else sorted(stable[:4])
        base.status('waiting_for_quiet_cores',disk_busy_pct=disk_pct,quiet_cores=len(quiet),stable_cores=len(stable))
        last=quiet
        if len(quiet)<4:
            time.sleep(10)


def prepare_window():
    while True:
        before=monitor.disk()
        start=time.monotonic()
        time.sleep(4)
        after=monitor.disk()
        seconds=time.monotonic()-start
        busy=(after[2]-before[2])/(10*seconds)
        writes=(after[1]-before[1])/(monitor.MIB*seconds)
        if busy<=25 and writes<=1:
            return
        base.status('waiting_to_prepare_layouts',disk_busy_pct=busy,write_mib_s=writes)
        time.sleep(15)


def layouts(c):
    xy = base.build_layout(c,'T_XY')
    x = ROOT/'scratch/openimage-b2-x-only'
    x.mkdir()
    n, dx, dy, _, _ = struct.unpack_from('<5I', Path(c['meta']).read_bytes())
    coord_pp = 8192//(4*dx)
    coord_bytes = ((n+coord_pp-1)//coord_pp)*8192
    base.require_space(coord_bytes+(xy/'group-1.bin').stat().st_size)
    output=ROOT/'analysis/x-only-layout.json'
    base.status('building_x_only_coordinates')
    base.run([base.ENGINE,'build',c['meta'],c['topo'],c['coord'],x,'S_T_X',output],
             'build-x-only',dict(LAYOUT_ORDER=c['order']))
    if base.read(output)['status'] != 'PASS' or (x/'group-2.bin').stat().st_size!=coord_bytes:
        raise RuntimeError('X-only coordinate layout validation')
    xy_topo, x_topo = xy/'group-1.bin',x/'group-1.bin'
    if base.digest(xy_topo)!=base.digest(x_topo):
        raise RuntimeError('layout changed the topology')
    alias=x/'group-1.shared'
    os.link(str(xy_topo),str(alias))
    os.replace(str(alias),str(x_topo))
    base.save(ROOT/'analysis/layout-identity.json',dict(shared_topology_inode=xy_topo.stat().st_ino,
              original_dims=[dx,dy],xy_coord_bytes=(xy/'group-6.bin').stat().st_size,
              x_coord_bytes=coord_bytes,xy_nodes_per_page=8192//(4*(dx+dy)),x_nodes_per_page=coord_pp,
              same_topology_sha256=base.digest(xy_topo),same_order_sha256=c['b2_identity']['order_sha256']))
    return {'T_XY':xy,'S_T_X':x}


def query(c, folders, variant, label, parity=False):
    layout, only_x = VARIANTS[variant]
    out=ROOT/'results'/(label+'.json')
    env=dict(LAYOUT_ORDER=c['order'],LAYOUT_SHARED_CACHE_BYTES=str(CACHE),
             LAYOUT_ENDPOINT_X_ONLY='1' if only_x else '0')
    if parity:
        prefix=ROOT/'analysis'/label
        env.update(LAYOUT_ONE_PASS='1',LAYOUT_TRACE=str(prefix)+'.trace',LAYOUT_DUMP=str(prefix))
    base.status('parity' if parity else 'timing',task=label,variant=variant,levels=LEVELS,cpus=monitor.CPUS)
    cmd=[base.ENGINE,'endpoint',c['meta'],c['topo'],c['coord'],folders[layout],layout,out,
         c['prefix']+'/',c['query1'],c['query2'],'unused-alpha','unused-gt',THREADS,base.SECONDS]+LEVELS
    if parity:
        base.run(cmd,label,env)
    else:
        monitor.monitored_run(cmd,label,env,monitor.CPUS)
    pts=base.read(out)
    if [p['L'] for p in pts]!=LEVELS:
        raise RuntimeError('incomplete L grid')
    for p in pts:
        base.check_point(p,True)
        if p['recall']!=-1 or p['queries']!=1000 or p['threads']!=THREADS or (not parity and p['seconds']<base.SECONDS):
            raise RuntimeError('endpoint workload validation')
        factor = 1 if only_x else 2
        if abs(p['exact_space_evals']-factor*p['reranked_nodes'])>1e-7:
            raise RuntimeError('exact distance count mismatch')
    return pts


def parity(c, folders):
    reference = None
    expected = None
    for variant in VARIANTS:
        label='parity-'+variant
        pts=query(c,folders,variant,label,parity=True)
        values={}
        for L in LEVELS:
            for ext in ('ids','dists'):
                file=ROOT/'analysis'/('{}.L{}.{}'.format(label,L,ext))
                if file.stat().st_size!=40000:
                    raise RuntimeError('output dump size')
                values['{}.{}'.format(L,ext)]=base.digest(file)
            values['{}.trace'.format(L)]=base.digest(ROOT/'analysis'/('{}.trace.L{}'.format(label,L)))
        if reference is None:
            reference = values
            expected = {p['L']:p for p in pts}
        elif reference != values:
            raise RuntimeError('IDs, distances or traversal trace differ')
        base.compare_logical(pts,expected,label)
        if variant=='B':
            for p in pts:
                a=expected[p['L']]
                if (abs(a['pq_space_evals']-2*p['pq_space_evals'])>1e-7 or
                    abs(a['coordinate_submitted_pages']-p['coordinate_submitted_pages'])>1e-7):
                    raise RuntimeError('A/B differing physical coordinates or incorrect PQ count')
    base.save(ROOT/'analysis/parity.json',dict(status='PASS',queries=1000,levels=LEVELS,
              ids_distances_and_traces_equal=True,hashes=reference))
    return expected


def resume_prepared(c):
    build_record = base.read(ROOT/'analysis/build.json')
    layout_record = base.read(ROOT/'analysis/layout-identity.json')
    proof = base.read(ROOT/'analysis/parity.json')
    folders = {'T_XY':ROOT/'scratch/openimage-b2-T_XY',
               'S_T_X':ROOT/'scratch/openimage-b2-x-only'}
    xy, x = folders['T_XY'], folders['S_T_X']
    if ((ROOT/'analysis/progress.json').exists()
            or base.digest(base.ENGINE)!=build_record['binary_sha256']
            or proof['status']!='PASS' or proof['levels']!=LEVELS
            or not proof['ids_distances_and_traces_equal']
            or layout_record['same_order_sha256']!=c['b2_identity']['order_sha256']
            or (xy/'group-1.bin').stat().st_ino!=(x/'group-1.bin').stat().st_ino
            or base.digest(xy/'group-1.bin')!=layout_record['same_topology_sha256']
            or (xy/'group-6.bin').stat().st_size!=layout_record['xy_coord_bytes']
            or (x/'group-2.bin').stat().st_size!=layout_record['x_coord_bytes']):
        raise RuntimeError('refusing to resume an unverified or partially timed experiment')
    for name,value in build_record['sources'].items():
        if base.digest(ROOT/'scripts'/name)!=value:
            raise RuntimeError('prepared binary source mismatch')
    base.status('restored_validated_layouts',levels=LEVELS)
    return folders,{p['L']:p for p in base.read(ROOT/'results/parity-A.json')}


def report(records):
    base.save(ROOT/'analysis/measurements.json',records)
    lines=['# Second-space cost at a fixed answer','',
           'A computes and stores XY. B computes X and stores XY. C computes and stores X.',
           'All three use the same M=2 graph, alpha=1, identical answers, four threads and 256 MiB topology cache.',
           'Rows show the mean of three accepted curves. Reader timing is submit and wait wall time.',
           '',
           '| L | Variant | QPS | Mean ms | p95 ms | PQ gather + lookup ms | Exact ms | Coordinate pages | Coordinate I/O ms |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---:|']
    means=[]
    for L in LEVELS:
        for variant in VARIANTS:
            group=[p for r in records if r['variant']==variant and r['accepted']
                   for p in r['points'] if p['L']==L]
            if len(group)!=3:
                continue
            row={k:statistics.mean(p[k] for p in group) for k in METRICS}
            means.append(dict(L=L,variant=variant,**row))
            lines.append('| {} | {} | {:.0f} | {:.3f} | {:.3f} | {:.3f} | {:.3f} | {:.2f} | {:.3f} |'.format(
                L,variant,row['qps'],row['latency_us']/1000,row['p95_us']/1000,row['pq_lookup_us']/1000,
                row['exact_us']/1000,row['coordinate_submitted_pages'],row['coordinate_submit_wait_us']/1000))
    base.save(ROOT/'analysis/means.json',means)
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    for d in ('analysis','bin','results','scratch'):
        (ROOT/d).mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        c=next(c for c in base.read(monitor.OLD/'analysis/inputs.json') if c['name']=='openimage')
        if os.getenv('HVS_RESUME_PREPARED')=='1':
            c=base.prepare_budget(c,'2')
            folders,expected=resume_prepared(c)
        else:
            build()
            prepare_window()
            monitor.verify_inputs(c)
            c=base.prepare_budget(c,'2')
            folders=layouts(c)
            expected=parity(c,folders)
        monitor.CPUS = choose_cpus()
        base.save(ROOT/'analysis/protocol.json',dict(dataset='openimage',alpha=1,budget=2,threads=THREADS,
                  cpus=monitor.CPUS,numa=3,cache_bytes=CACHE,levels=LEVELS,repeats=3,seconds=base.SECONDS,
                  gt_used=False,variants=VARIANTS,raw_telemetry=True))
        records=[]
        for rep, order in enumerate(('ABC','BCA','CAB'),1):
            for variant in order:
                for attempt in (1,2,3):
                    label='{}-r{}-a{}'.format(variant,rep,attempt)
                    monitor.gate(label)
                    pts=query(c,folders,variant,label)
                    base.compare_logical(pts,expected,label)
                    quality=[monitor.telemetry(label,p) for p in pts]
                    accepted=all(p['quiet'] for p in quality)
                    records.append(dict(label=label,variant=variant,repeat=rep,attempt=attempt,
                                        accepted=accepted,points=pts,telemetry=quality))
                    report(records)
                    base.save(ROOT/'analysis/progress.json',dict(accepted=sum(r['accepted'] for r in records),
                              total=9,attempts=len(records),updated_at=base.now(),task=label))
                    if accepted:break
        for folder in folders.values():
            shutil.rmtree(str(folder))
    accepted=sum(r['accepted'] for r in records)
    base.status('complete' if accepted==9 else 'complete_with_flagged_points',
                accepted=accepted,total=9,points=accepted*len(LEVELS),attempts=len(records),serial_lock_released=True)


if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as e:
        base.status('interrupted' if isinstance(e,InterruptedError) else 'failed',error=str(e))
        traceback.print_exc()
        sys.exit(1)
