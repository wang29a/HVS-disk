"""Targeted repeat of the November 9 comparisons at their original L points."""
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent.parent
OLD = ROOT.parent / 'layout-cache-budget-20261109'
sys.path.insert(0, str(OLD / 'scripts'))
import run as base

ORIGINAL_RUN = base.run
CPUS = [74,75,76,77]
WR_LIMIT = 5
READ_RESIDUAL_LIMIT = 100
base.ROOT = ROOT
base.ENGINE = OLD/'bin/layout-engine'
base.CONVERTER = OLD/'bin/convert-budget'
DS = ('openimage','howto100m')
MIB = 1 << 20


def disk():
    row = next(line.split() for line in Path('/proc/diskstats').read_text().splitlines()
               if line.split()[2] == 'nvme3n1')
    return (int(row[5])*512, int(row[9])*512, int(row[12]))


def cpu():
    out = {}
    for line in Path('/proc/stat').read_text().splitlines():
        cols = line.split()
        if cols[0].startswith('cpu') and cols[0][3:].isdigit():
            out[int(cols[0][3:])] = list(map(int,cols[1:9]))
    return out


def gate(label):
    consecutive = 0
    while consecutive < 2:
        before, ticks, start = disk(), cpu(), time.monotonic()
        time.sleep(3)
        after, end_ticks = disk(), cpu()
        elapsed = time.monotonic()-start
        maximum = 0
        for c in CPUS + [c+96 for c in CPUS]:
            diff = [a-b for a,b in zip(end_ticks[c],ticks[c])]
            maximum = max(maximum, 100*(1-diff[3]/sum(diff)))
        sample = dict(task=label, at=base.now(), max_cpu_pct=maximum,
                      disk_busy_pct=(after[2]-before[2])/(10*elapsed),
                      write_mib_s=(after[1]-before[1])/(MIB*elapsed))
        with (ROOT/'logs/entry-load.jsonl').open('a') as f:
            f.write(json.dumps(sample)+'\n')
        if maximum <= 15 and sample['disk_busy_pct'] <= 25 and sample['write_mib_s'] <= 1:
            consecutive += 1
        else:
            consecutive = 0
            base.status('waiting_for_quiet_window', **sample)
            time.sleep(15)


def read_bytes(pid):
    try:
        for line in Path('/proc/{}/io'.format(pid)).read_text().splitlines():
            if line.startswith('read_bytes:'):
                return int(line.split(':',1)[1])
    except (OSError, ValueError):
        return None
    return None


def monitored_run(cmd, label, env=None, cpus=None):
    if not cpus:
        return ORIGINAL_RUN(cmd,label,env,cpus)
    args = ['taskset','-c',','.join(map(str,cpus))] + [str(x) for x in cmd]
    started = time.time()
    samples = []
    with (ROOT/'logs'/(label+'.log')).open('w') as log:
        worker = subprocess.Popen(args,env=dict(base.ENV,**(env or {})),stdout=log,
                                  stderr=subprocess.STDOUT,start_new_session=True)
        try:
            previous = time.time(), disk(), read_bytes(worker.pid)
            while True:
                try:
                    code = worker.wait(timeout=1)
                    break
                except subprocess.TimeoutExpired:
                    current = time.time(), disk(), read_bytes(worker.pid)
                    elapsed = current[0]-previous[0]
                    if elapsed > 0:
                        dr = (current[1][0]-previous[1][0])/(MIB*elapsed)
                        wr = (current[1][1]-previous[1][1])/(MIB*elapsed)
                        own = ((current[2]-previous[2])/(MIB*elapsed)
                               if current[2] is not None and previous[2] is not None else None)
                        samples.append(dict(start=previous[0],end=current[0],device_read_mib_s=dr,
                                            device_write_mib_s=wr,process_read_mib_s=own))
                    previous = current
        except BaseException:
            try:
                os.killpg(worker.pid,signal.SIGKILL)
            except ProcessLookupError:
                pass
            worker.wait()
            raise
    base.save(ROOT/'analysis'/(label+'-telemetry.json'),samples)
    if code:
        raise RuntimeError('{} exit {}'.format(label,code))
    base.save(ROOT/'analysis'/(label+'-run.json'),dict(command=args,started_unix=started,
              finished_unix=time.time(),environment=env or {}))


def telemetry(label, point):
    raw = base.read(ROOT/'analysis'/(label+'-telemetry.json'))
    start = point['start_unix']
    end = start+point['seconds']
    aligned = [s for s in raw if s['start']>=start and s['end']<=end]
    own = [s for s in aligned if s['process_read_mib_s'] is not None]
    writes = max((s['device_write_mib_s'] for s in aligned),default=0)
    outside = sum(s['device_read_mib_s']-s['process_read_mib_s'] for s in own)/max(1,len(own))
    # Also retain all raw intervals. Small differences can arise from delayed accounting.
    return dict(aligned_samples=len(aligned), max_device_write_mib_s=writes,
                mean_device_minus_process_read_mib_s=outside,
                quiet=len(own)>=5 and writes<=WR_LIMIT and outside<=READ_RESIDUAL_LIMIT)


def verify_inputs(c):
    for key, spec in c['input_stats'].items():
        stat = Path(c[key]).stat()
        if stat.st_size != spec['bytes'] or stat.st_mtime_ns != spec['mtime_ns']:
            raise RuntimeError('original input changed: ' + c[key])
    for key, value in c['input_sha256'].items():
        if base.digest(c[key]) != value:
            raise RuntimeError('original query or metadata hash changed')
    folder = Path(c['order']).parent
    for key in ('meta','topo','order'):
        if base.digest(folder/(key+'.bin')) != c['b2_identity'][key+'_sha256']:
            raise RuntimeError('archived budget 2 identity changed')


def old_case(ds, layout, enabled, budget, L):
    stage = 'four-cell' if ds=='openimage' else 'budget'
    mode = 'on' if enabled else 'off'
    values = []
    for rep in (1,2,3):
        name = '{}-{}-b{}-{}-{}-r{}.json'.format(stage,ds,budget,layout,mode,rep)
        rows = base.read(OLD/'results'/name)
        values.append(next(p for p in rows if p['L']==L))
    return values


def openimage_layout(c, layout):
    label = 'openimage-b2-'+layout
    folder = ROOT/'scratch'/label
    if not folder.exists():
        return base.build_layout(c,layout)
    record = base.read(ROOT/'analysis'/(label+'-layout.json'))
    identity = base.read(ROOT/'analysis'/(label+'-identity.json'))
    if (record['status'] != 'PASS' or record['record_mismatches']
            or identity['order_sha256'] != c['b2_identity']['order_sha256']
            or identity['metadata_sha256'] != c['b2_identity']['meta_sha256']
            or identity['topology_sha256'] != c['b2_identity']['topo_sha256']
            or sorted(f.stat().st_size for f in folder.glob('group-*.bin')) != sorted(identity['group_bytes'])):
        raise RuntimeError('existing validated layout identity mismatch')
    return folder


def verify_openimage(c, folders):
    ref = base.read(OLD/'analysis/openimage-four-cell-parity.json')['hashes']
    for layout in base.LAYOUTS:
        for mode in (False,True):
            label = 'parity-openimage-{}-{}'.format(layout,'on' if mode else 'off')
            point = base.query(c,layout,folders[layout],mode,[176],label,one=True,dump=True)[0]
            old = old_case('openimage',layout,mode,'2',176)
            base.compare_logical([point],{176:old[0]},label)
            for ext in ('ids','dists'):
                if base.digest(ROOT/'analysis'/('{}.L176.{}'.format(label,ext))) != ref['176.'+ext]:
                    raise RuntimeError('original output mismatch: '+label)
    base.save(ROOT/'analysis/openimage-parity.json',dict(status='PASS',L=176))


def report(records):
    base.save(ROOT/'analysis/retest-results.json',records)
    lines = ['# Targeted quiet-window retest', '',
             '| Dataset | Budget | Layout | Cache | Recall | Original QPS range | Retest QPS (quiet) | Retest p95 ms (quiet) |',
             '|---|---|---|---|---:|---:|---|---|']
    groups = {}
    for r in records:
        key = tuple(r[k] for k in ('dataset','budget','layout','cache'))
        groups.setdefault(key,[]).append(r)
    for key, group in sorted(groups.items()):
        example=group[0]
        keep=[r['point'] for r in group if r['telemetry']['quiet']]
        old=example['old_qps']
        lines.append('| {} | {} | {} | {} | {:.2f}% | {:.0f}–{:.0f} | {} | {} |'.format(
            *(list(key)+[example['point']['recall'],min(old),max(old),
             ', '.join('{:.0f}'.format(p['qps']) for p in keep) or 'pending',
             ', '.join('{:.2f}'.format(p['p95_us']/1000) for p in keep) or 'pending'])))
    lines.extend(['','Flagged timing attempts are preserved in the JSON and are absent from the quiet columns.'])
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def run_cases(ds, cases, records):
    count = len(cases)
    orders = ((0,1,3,2),(2,3,1,0),(1,0,2,3)) if count==4 else ((0,1),(1,0),(0,1))
    for rep,order in enumerate(orders,1):
        for i in order:
            c,layout,folder,enabled,L = cases[i]
            mode = 'on' if enabled else 'off'
            budget = c['budget']
            name = '{}-b{}-{}-{}-r{}'.format(ds,budget,layout,mode,rep)
            old = old_case(ds,layout,enabled,budget,L)
            earlier = [r for r in records if (r['dataset'],r['budget'],r['layout'],r['cache'],r['repeat'])
                       == (ds,budget,layout,mode,rep)]
            if any(r['telemetry']['quiet'] for r in earlier):
                continue
            for attempt in range(1+len(earlier),4):
                label = '{}-a{}'.format(name,attempt)
                gate(label)
                point = base.query(c,layout,folder,enabled,[L],label,cpus=CPUS)[0]
                base.compare_logical([point],{L:old[0]},label)
                quality = telemetry(label,point)
                records.append(dict(dataset=ds,budget=budget,layout=layout,cache=mode,
                                    repeat=rep,attempt=attempt,point=point,telemetry=quality,
                                    old_qps=[p['qps'] for p in old],old_p95_us=[p['p95_us'] for p in old]))
                report(records)
                if quality['quiet']:
                    break
            base.save(ROOT/'analysis/progress.json',dict(
                accepted=sum(r['telemetry']['quiet'] for r in records),total=18,
                attempts=len(records),updated_at=base.now(),last=label))


def main():
    for directory in ('analysis','results','scratch'):
        (ROOT/directory).mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        spec=base.read(OLD/'analysis/build.json')
        for binary in (base.ENGINE,base.CONVERTER):
            if base.digest(binary)!=spec['binaries'][binary.name]:
                raise RuntimeError('original binary hash mismatch')
        if base.digest(spec['library'])!=spec['library_sha256']:
            raise RuntimeError('original library hash mismatch')
        configs={c['name']:c for c in base.read(OLD/'analysis/inputs.json')}
        base.save(ROOT/'analysis/protocol.json',dict(cpus=CPUS,numa_node=3,shared_cache_mib=256,
                  binary_sha256=spec['binaries'],old=str(OLD),independent_query_count=1000,seconds=base.SECONDS,
                  datasets=dict(openimage=[176],howto100m=[314,316]),repeat=3,
                  write_limit_mib_s=WR_LIMIT,read_residual_limit_mib_s=READ_RESIDUAL_LIMIT))
        base.run=monitored_run
        previous = ROOT/'analysis/retest-results.json'
        records=base.read(previous) if previous.exists() else []
        if records:
            report(records)
        for ds in DS:
            gate(ds+'-prepare')
            base.status('verifying_inputs',dataset=ds)
            c=configs[ds]
            verify_inputs(c)
            c2=base.prepare_budget(c,'2')
            if ds=='openimage':
                folders={layout:openimage_layout(c2,layout) for layout in base.LAYOUTS}
                verify_openimage(c2,folders)
                cases=[(c2,layout,folders[layout],mode,176) for layout in base.LAYOUTS for mode in (False,True)]
            else:
                f2=base.build_layout(c2,'T_XY')
                c3=base.prepare_budget(c,'3')
                f3=base.build_layout(c3,'T_XY',coord_ref=f2/'group-6.bin')
                cases=[(c2,'T_XY',f2,True,314),(c3,'T_XY',f3,True,316)]
            run_cases(ds,cases,records)
            for folder in (ROOT/'scratch').glob(ds+'-*'):
                shutil.rmtree(str(folder))
    accepted=sum(r['telemetry']['quiet'] for r in records)
    base.status('complete' if accepted==18 else 'complete_with_flagged_points', accepted=accepted,
                total=18,attempts=len(records),serial_lock_released=True)


if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as e:
        base.status('interrupted' if isinstance(e,InterruptedError) else 'failed',error=str(e))
        traceback.print_exc()
        sys.exit(1)
