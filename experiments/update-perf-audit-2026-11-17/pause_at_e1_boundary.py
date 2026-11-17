"""User-authorized pause after exact-r1, without interrupting that trajectory."""
from pathlib import Path
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import time

ROOT=Path(__file__).resolve().parent
E1=Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected-v2')
E=Path('/mnt/nvme3/wz/hvs-disk-build-20261027/experiments')
OUT=ROOT/'runtime';OUT.mkdir(exist_ok=True)


def save(name,value):
    p=OUT/name;q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(value,indent=2)+'\n');q.replace(p)


def coordinator():
    matches=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():continue
        try:
            if proc.stat().st_uid!=os.getuid():continue
            cmd=(proc/'cmdline').read_bytes()
            if cmd!=b'python3\0resume_degree_e1.py\0':continue
            env=(proc/'environ').read_bytes().split(b'\0')
            if ('HVS_E1_RUN_ROOT='+str(E1)).encode() in env:matches.append(int(proc.name))
        except OSError:pass
    assert len(matches)==1,'expected one live E1 v2 coordinator: '+str(matches)
    return matches[0]


def main():
    pid=coordinator();proc=Path('/proc')/str(pid)
    command=(proc/'cmdline').read_bytes()
    start_id=(proc/'stat').read_text().rsplit(')',1)[1].split()[19]
    save('pause-monitor.json',dict(stage='WAITING_FOR_EXACT_R1_ARCHIVE',pid=pid,
        authorization='user chose diagnostics after current exact first trajectory; later repetitions and ablations paused',
        cmdline_sha256=hashlib.sha256(command).hexdigest()))
    archive=E1/'variants/exact-t4-r1/archive-identity.json'
    while not archive.exists():
        state=json.loads((E1/'status.json').read_text())
        if state['stage'].startswith('STOPPED'):
            save('pause-monitor.json',dict(stage='E1_STOPPED_BEFORE_BOUNDARY',e1=state));return
        if not proc.exists():
            save('pause-monitor.json',dict(stage='COORDINATOR_EXITED_BEFORE_BOUNDARY'));return
        time.sleep(5)
    points=json.loads((E1/'variants/exact-t4-r1/checkpoints.json').read_text())
    assert [v['checkpoint'] for v in points]==list(range(5))
    assert points[-1]['n']==2000000
    assert (proc/'cmdline').read_bytes()==command
    assert (proc/'stat').read_text().rsplit(')',1)[1].split()[19]==start_id
    os.kill(pid,signal.SIGTERM)
    # The following repetition can begin preparing files between two checks.
    # Stop only a verified owned driver from a later repetition, never exact-r1.
    stopped=[]
    for p in (E1/'variants').glob('*/driver-command.json'):
        if p.parent.name in ('pq-t4-r1','exact-t4-r1'):continue
        v=json.loads(p.read_text());child=int(v['pid']);cp=Path('/proc')/str(child)
        if cp.exists():
            cmd=(cp/'cmdline').read_bytes()
            assert b'e1_driver' in cmd and str(p.parent).encode() in cmd
            os.kill(child,signal.SIGTERM);stopped.append(child)
    lock=(E/'budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    record=dict(stage='PAUSED_AFTER_EXACT_R1_USER_REQUEST',updated_unix=time.time(),
        completed=['pq-t4-r1','exact-t4-r1'],coordinator_pid=pid,late_following_driver_pids=stopped,
        reason='prioritize1000-insert profile and upstream OdinANN same-device reference',
        archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    save('pause-monitor.json',record)
    q=E1/'status.json';temp=q.with_name(q.name+'.tmp');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(q)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    while not (ROOT/'READY').exists():time.sleep(5)
    log=(OUT/'suite-launch.log').open('w')
    child=subprocess.Popen(['python3',str(ROOT/'run_suite.py')],stdout=log,stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,start_new_session=True,cwd=str(ROOT))
    save('suite-pid.json',dict(pid=child.pid,launched_unix=time.time()))


if __name__=='__main__':main()
