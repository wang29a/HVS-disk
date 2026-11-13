#!/usr/bin/env python3
"""Import the completed NUMA 3 I/O batch into the existing experiment summary."""
import fcntl
import importlib.util
from pathlib import Path
import time


ROOT = Path(__file__).resolve().parent.parent
IMPORTER = ROOT.parent / 'cc3m-highrecall-t4-20261104/scripts/sync_results.py'


def main():
    spec = importlib.util.spec_from_file_location('experiment_importer', IMPORTER)
    importer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(importer)
    importer.IO = 'io-time-numa3-20261108'
    importer.LOCAL_IO = ROOT
    with (importer.ROOT / 'sync.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                states = importer.remote_states()
                status = states[importer.IO]
                importer.write_json(ROOT / 'remote-status.json', dict(checked_at=importer.now(), status=status))
                if status['status'] == 'complete':
                    importer.append_completed(states)
                    importer.write_json(ROOT / 'local-sync-status.json', dict(status='complete', finished_at=importer.now()))
                    importer.refresh_checksums(ROOT)
                    print(importer.now(), 'NUMA3_IO_IMPORTED', flush=True)
                    return
                if status['status'] in {'failed', 'paused_by_user'}:
                    print(importer.now(), status, flush=True)
                    return
            except Exception as error:
                print(importer.now(), 'SYNC_RETRY', type(error).__name__, str(error), flush=True)
            time.sleep(60)


if __name__ == '__main__':
    main()
