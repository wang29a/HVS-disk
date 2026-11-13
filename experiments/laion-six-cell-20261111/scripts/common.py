"""Owned serial Laion runner; search four cores, construction at most 24."""
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from queue_guard import stop_owned_group

ROOT = Path(__file__).resolve().parent.parent
PREV = ROOT.parent/'metric-consistent-20261111'
DATA = Path('/mnt/nvme2/wz/disk/dataset/Laion')
STORE = Path('/mnt/nvme2/wz/hvs-laion-six-cell-20261111')
COORD = Path('/mnt/nvme3/wz/Index/Laion-shard/disk_index_data')
CPUS = [72, 73, 74, 75]
BUILD_CPUS = list(range(72, 96))
ACTIVE = ROOT
CHILD = None
ENV = dict(os.environ, LC_ALL='C', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='4', OMP_THREAD_LIMIT='4', HVS_FULL_ONLY='1')
for key in list(ENV):
    if key.startswith(('LAYOUT_', 'METRIC_')):
        del ENV[key]
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(str(temp), str(path))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for data in iter(lambda: source.read(8 << 20), b''):
            h.update(data)
    return h.hexdigest()


def status(stage, **extra):
    save(ROOT/'status.json', dict(stage=stage, pid=os.getpid(), updated_unix=time.time(),
                                 workload=str(ACTIVE.relative_to(ROOT)), **extra))


def init():
    for name in ('analysis', 'logs', 'bin', 'scratch', 'workloads'):
        (ROOT/name).mkdir(exist_ok=True)
    STORE.mkdir(exist_ok=True)
    for name in ('metric-search', 'libpipeann-metric.a'):
        link = ROOT/'bin'/name
        if not link.exists():
            link.symlink_to(PREV/'bin'/name)


def run(args, label, cpus=None, extra=None, expected=0, timeout=21600):
    global CHILD
    cpus = CPUS if cpus is None else cpus
    env = dict(ENV)
    env.update(extra or {})
    (ACTIVE/'logs').mkdir(parents=True, exist_ok=True)
    (ACTIVE/'analysis').mkdir(parents=True, exist_ok=True)
    command = ['nice', '-n', '10', 'ionice', '-c', '2', '-n', '7', 'numactl', '--all',
               '--physcpubind='+','.join(map(str, cpus)), '--membind=3']+list(map(str, args))
    timing = ACTIVE/'logs'/(label+'.time')
    command = ['/usr/bin/time', '-v', '-o', str(timing)]+command
    save(ACTIVE/'analysis'/(label+'-command.json'), dict(command=command, overrides=extra or {},
         started_unix=time.time(), timeout_seconds=timeout))
    with (ACTIVE/'logs'/(label+'.log')).open('w') as output:
        CHILD = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        status('running', task=label, child_pid=CHILD.pid, cpus=cpus)
        try:
            code = CHILD.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            stop_owned_group(CHILD)
            CHILD = None
            raise RuntimeError(label+' exceeded timeout')
        CHILD = None
    if code != expected:
        raise RuntimeError('{} exited {}, expected {}'.format(label, code, expected))


def cpu_ticks(cpus):
    result = {}
    for line in Path('/proc/stat').read_text().splitlines():
        parts = line.split()
        if parts[0].startswith('cpu') and parts[0][3:].isdigit():
            cpu = int(parts[0][3:])
            if cpu in cpus:
                values = list(map(int, parts[1:9]))
                result[cpu] = (sum(values), sum(values)-values[3]-values[4])
    return result


def quiet(cpus, disk=False, timeout=None):
    siblings = cpus+[c+96 for c in cpus]
    started = time.monotonic()
    while True:
        before = cpu_ticks(siblings)
        d = list(map(int, Path('/sys/block/nvme3n1/stat').read_text().split()))
        tick = time.monotonic()
        time.sleep(2)
        elapsed = time.monotonic()-tick
        after = cpu_ticks(siblings)
        cpu = sum(after[c][1]-before[c][1] for c in siblings)/(elapsed*os.sysconf('SC_CLK_TCK')*len(cpus))
        dd = list(map(int, Path('/sys/block/nvme3n1/stat').read_text().split()))
        util = (dd[9]-d[9])/(elapsed*1000.)
        if cpu < .10 and (not disk or util < .10):
            return
        status('waiting_for_measurement_window', cpus=cpus, external_cpu_fraction=cpu,
               device_busy_fraction=util, wait_seconds=time.monotonic()-started)
        if timeout is not None and time.monotonic()-started >= timeout:
            raise TimeoutError('measurement resources remain busy')
        time.sleep(20)


def node_memory():
    result = {}
    for line in Path('/sys/devices/system/node/node3/meminfo').read_text().splitlines():
        parts = line.split()
        if len(parts) >= 5:
            result[parts[2].rstrip(':')] = int(parts[3])*1024
    return result


def stop(signum, frame):
    if CHILD and CHILD.poll() is None:
        stop_owned_group(CHILD)
    status('interrupted', signal=signum)
    sys.exit(128+signum)


for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sig, stop)
