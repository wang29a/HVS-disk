"""Resumable, owned serial queue. Every long task is restricted to NUMA 3."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import struct
import subprocess
import sys
import time
from queue_guard import stop_owned_group

ROOT = Path(__file__).resolve().parent.parent
EXPERIMENTS = ROOT.parent
CPUS = list(range(72, 80))
ENV = dict(os.environ, LC_ALL='C', OMP_NUM_THREADS='8', OMP_THREAD_LIMIT='8',
           OPENBLAS_NUM_THREADS='4', MKL_NUM_THREADS='4')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')
CHILD = None


def save(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(data, indent=2)+'\n')
    os.replace(str(temp), str(path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def status(stage, **extra):
    save(ROOT/'status.json', dict(stage=stage, updated_unix=time.time(), pid=os.getpid(), **extra))


def cpu_ticks(cpus):
    result = {}
    with open('/proc/stat') as stream:
        for line in stream:
            parts = line.split()
            if parts and parts[0].startswith('cpu') and parts[0][3:].isdigit():
                cpu = int(parts[0][3:])
                if cpu in cpus:
                    v = list(map(int, parts[1:9]))
                    result[cpu] = (sum(v), sum(v)-v[3]-v[4])
    return result


def quiet(cpus, disk=False):
    siblings = cpus + [c+96 for c in cpus]
    waiting_since = time.monotonic()
    while True:
        before = cpu_ticks(siblings)
        disks = Path('/sys/block/nvme3n1/stat').read_text().split()
        start = time.monotonic()
        time.sleep(2)
        after = cpu_ticks(siblings)
        elapsed = time.monotonic()-start
        busy = sum(after[c][1]-before[c][1] for c in siblings)
        capacity = os.sysconf('SC_CLK_TCK')*elapsed*len(cpus)
        fraction = busy/capacity
        disk_after = Path('/sys/block/nvme3n1/stat').read_text().split()
        util = (int(disk_after[9])-int(disks[9]))/(elapsed*1000.)
        if fraction < .10 and (not disk or util < .10):
            return
        status('waiting_for_quiet_resources', cpus=cpus, external_cpu_fraction=fraction,
               nvme3_busy_fraction=util, wait_seconds=time.monotonic()-waiting_since)
        time.sleep(20)


def run(args, label, cpus=None, extra=None, expected=0, timeout=None):
    global CHILD
    env = dict(ENV)
    env.update(extra or {})
    command = list(map(str, args))
    if cpus:
        command = ['nice', '-n', '10', 'ionice', '-c', '2', '-n', '7',
                   'numactl', '--all', '--physcpubind='+','.join(map(str, cpus)), '--membind=3']+command
    if timeout is None:
        timeout = 14400 if label.startswith(('build-openimage', 'prepare-single')) else 1200
    save(ROOT/'analysis'/(label+'-command.json'), dict(command=command, overrides=extra or {}, start=time.time(),
         timeout_seconds=timeout))
    with (ROOT/'logs'/(label+'.log')).open('w') as output:
        CHILD = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        status('running', task=label, child_pid=CHILD.pid, cpus=cpus)
        try:
            code = CHILD.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_owned_group(CHILD)
            CHILD = None
            raise RuntimeError('{} exceeded {} second timeout'.format(label, timeout))
        CHILD = None
    if code != expected:
        raise RuntimeError('{} exited {}, expected {}'.format(label, code, expected))


def terminate(signum, frame):
    if CHILD and CHILD.poll() is None:
        os.killpg(CHILD.pid, signal.SIGTERM)
    status('interrupted', signal=signum)
    sys.exit(128+signum)


def compile_all():
    run([sys.executable, ROOT/'scripts/prepare_variant.py'], 'prepare-variant')
    spec = json.loads((EXPERIMENTS/'concurrency-alpha/analysis/build.json').read_text())['separated']
    if digest(spec['library']) != spec['library_sha256']:
        raise RuntimeError('old reference archive changed')
    flags = ['-I'+str(ROOT/'variant'), '-I'+str(ROOT/'scripts')]+spec['flags']
    link = ['-L/mnt/nvme2/wz/disk/local/openblas/lib', '-laio', '-ljemalloc', '-lopenblas']
    objects = [ROOT/'bin/hybrid_index.cpp.o', ROOT/'bin/aux_utils.cpp.o']
    # Sequential compilation keeps process ownership and cancellation simple.
    for source, obj in zip(['hybrid_index.cpp', 'aux_utils.cpp'], objects):
        run(['g++-11']+flags+['-c', ROOT/'variant'/source, '-o', obj], 'compile-'+source, CPUS[:2])
    library = ROOT/'bin/libpipeann-metric.a'
    shutil.copyfile(spec['library'], str(library))
    members = subprocess.check_output(['ar', 't', str(library)], env=ENV).decode().splitlines()
    if not all(members.count(p.name) == 1 for p in objects):
        raise RuntimeError('archive object identity')
    run(['ar', 'r', library]+objects, 'replace-isolated-objects')
    for name, source in [('build-graph', 'build_graph.cpp'), ('metric-search', 'layout_engine.cpp'),
                         ('check-metric', 'check_metric.cpp'), ('check-intervals', 'check_merge_intervals.cpp')]:
        run(['g++-11']+flags+[ROOT/'scripts'/source, library]+link+['-o', ROOT/'bin'/name], 'link-'+name, CPUS[:2])
    run([ROOT/'bin/check-metric'], 'metric-regression', CPUS[:2])
    run([ROOT/'bin/check-intervals'], 'interval-regression', CPUS[:2])
    run([ROOT/'bin/metric-search', 'selftest'], 'search-metric-regression', CPUS[:2])
    save(ROOT/'analysis/build.json', dict(status='PASS', flags=flags, link=link,
         reference_library_sha256=spec['library_sha256'], library_sha256=digest(library),
         engine_source_sha256=digest(ROOT/'scripts/layout_engine.cpp'),
         binaries={p.name:digest(p) for p in (ROOT/'bin').iterdir() if p.is_file() and not p.suffix}))


def refresh_search():
    proof = json.loads((ROOT/'analysis/build.json').read_text())
    source = ROOT/'scripts/layout_engine.cpp'
    if proof.get('engine_source_sha256') == digest(source):
        return
    if (ROOT/'analysis/measurement-inputs.json').exists():
        raise RuntimeError('search source changed after measurement freeze; use a new namespace')
    run(['g++-11']+proof['flags']+[source, ROOT/'bin/libpipeann-metric.a']+proof['link']+
        ['-o', ROOT/'bin/metric-search'], 'refresh-metric-search', CPUS[:2])
    run([ROOT/'bin/metric-search', 'selftest'], 'search-metric-regression', CPUS[:2])
    proof['engine_source_sha256'] = digest(source)
    proof['binaries']['metric-search'] = digest(ROOT/'bin/metric-search')
    save(ROOT/'analysis/build.json', proof)


def preflight_fixture():
    import numpy as np
    target = ROOT/'scratch/fixture'
    target.mkdir(exist_ok=True)
    random = np.random.RandomState(20261111)
    for name, dim in [('x', 8), ('y', 16)]:
        with (target/(name+'.bin')).open('wb') as out:
            out.write(struct.pack('<II', 512, dim))
            random.normal(size=(512, dim)).astype('<f4').tofile(out)
    run([ROOT/'bin/build-graph', 'dual', target/'x.bin', target/'y.bin', 512, 8, 16, 2, target],
        'fixture-build', CPUS[:2], dict(OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2'))
    from validate_graph import validate
    save(ROOT/'analysis/fixture-validated.json', validate(target, 512, 8, 16))


def main():
    for name in ('analysis', 'logs', 'bin', 'scratch', 'results'):
        (ROOT/name).mkdir(exist_ok=True)
    lock = open(str(EXPERIMENTS/'budget-reorder-20261103/queue.lock'), 'a+')
    status('waiting_for_serial_lock')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if not (ROOT/'analysis/build.json').is_file():
        compile_all()
    refresh_search()
    if not (ROOT/'analysis/fixture-validated.json').is_file():
        preflight_fixture()
    smoke = ROOT/'analysis/engine-smoke.json'
    if not smoke.exists() or json.loads(smoke.read_text()).get('engine_sha256') != digest(ROOT/'bin/metric-search'):
        run([sys.executable, ROOT/'scripts/preflight_engine.py'], 'eight-query-functional-smoke', CPUS[:2],
            dict(OMP_NUM_THREADS='2', OMP_THREAD_LIMIT='2', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1'))
    if not (ROOT/'analysis/objective-audit.json').is_file():
        quiet(CPUS)
        run([sys.executable, ROOT/'scripts/audit_objective.py'], 'audit-objective', CPUS)
    if json.loads((ROOT/'analysis/objective-audit.json').read_text())['status'] != 'PASS':
        raise RuntimeError('GT objective gate failed')
    graph = ROOT/'scratch/dual'
    graph.mkdir(exist_ok=True)
    if not (ROOT/'analysis/dual-validated.json').is_file():
        if shutil.disk_usage(str(ROOT)).free < 6*(1 << 30):
            raise RuntimeError('insufficient free space for source graph and one coupled layout')
        quiet(CPUS)
        data = Path('/mnt/nvme3/wz/dataset/Openimage_disk')
        started = time.time()
        audit = graph/'graph-audit.json'
        ready = audit.exists()
        if ready:
            try:
                ready = json.loads(audit.read_text())['status'] == 'PASS'
            except (ValueError, KeyError):
                ready = False
        if not ready:
            run([ROOT/'bin/build-graph', 'dual', data/'base_img_emb.fvecs', data/'base_text_emb.fvecs',
                 456699, 768, 768, 8, graph], 'build-openimage-dual', CPUS)
        status('validating_new_dual_graph', cpus=[72], resumed_completed_build=ready)
        from validate_graph import validate
        proof = validate(graph, 456699, 768, 768)
        proof['wall_seconds'] = time.time()-started
        proof['build_reused_after_resume'] = ready
        save(ROOT/'analysis/dual-validated.json', proof)
    # Phase two is run in the same lock-owning process.
    import phase_two
    phase_two.main(sys.modules[__name__])


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        main()
    except Exception as error:
        status('failed', error=str(error))
        raise
