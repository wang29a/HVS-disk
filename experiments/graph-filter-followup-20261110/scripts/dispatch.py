"""Run the two downstream stages in order after the OpenImages batch."""
import fcntl
import os
import subprocess
import sys
import time
import traceback

import followup as common

ROOT, base = common.ROOT, common.base
PARENT = ROOT/'status.json'
RETRY = sys.argv[1:] == ['--cc3m-only']


def status(state, **fields):
    base.save(ROOT/('cc3m-retry.json' if RETRY else 'downstream.json'),dict(status=state, updated_at=base.now(),**fields))
    print(base.now(),state,fields,flush=True)


def stage(module, function, cores=None):
    command = [sys.executable,'-u','-c','from {} import {}; {}()'.format(module,function,function)]
    if cores:
        command = ['numactl','--physcpubind='+','.join(map(str,cores)),'--membind=3']+command
    subprocess.run(command,cwd=str(ROOT/'scripts'),env=dict(base.ENV,PYTHONDONTWRITEBYTECODE='1'),check=True)


def cc3m_stages():
    original_status = base.status
    try:
        base.status = status
        cores = common.quiet_cores(2)
    finally:
        base.status = original_status
    status('preparing_cc3m_x',cpus=cores)
    stage('cc3m','prepare',cores)
    status('comparing_cc3m_endpoint')
    stage('cc3m','compare')


def main():
    if RETRY:
        status('waiting_for_serial_lock')
        with (ROOT/'downstream.lock').open('a') as own, (base.PREV/'queue.lock').open('r') as shared:
            fcntl.flock(own,fcntl.LOCK_EX)
            fcntl.flock(shared,fcntl.LOCK_EX)
            cc3m_stages()
        state = base.read(ROOT/'cc3m/status.json')
        status(state['status'], stage=state)
        return

    status('waiting_for_openimage')
    while True:
        prior = base.read(PARENT)
        if prior['status'] in ('complete','complete_with_flagged_points'):
            break
        if prior['status'] in ('failed','interrupted'):
            raise RuntimeError('OpenImages stage ended before completion: '+str(prior))
        time.sleep(15)
    with (ROOT/'downstream.lock').open('a') as own, (base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        failures = {}
        try:
            cc3m_stages()
        except subprocess.CalledProcessError as exc:
            failures['cc3m'] = str(exc)
            traceback.print_exc()
        try:
            status('comparing_native_cgl',previous_errors=failures)
            stage('native','experiment')
        except subprocess.CalledProcessError as exc:
            failures['native'] = str(exc)
            traceback.print_exc()
    stages = {name:base.read(ROOT/name/'status.json') for name in ('cc3m','native')
              if (ROOT/name/'status.json').exists()}
    if failures:
        status('failed',errors=failures,stages=stages)
        sys.exit(1)
    status('complete' if all(s['status']=='complete' for s in stages.values()) else 'complete_with_flagged_points',
           stages=stages)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        status('failed',error=str(exc))
        traceback.print_exc()
        sys.exit(1)
