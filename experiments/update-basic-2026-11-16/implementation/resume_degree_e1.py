"""Wait for this revision's validation, then restart E1 in a fresh namespace."""
import json
import os
from pathlib import Path
import shutil
import time
import run_g0 as c
import run_e1 as e
from run_degree_validation import setup, OUT


def main():
    assert e.COLD.name.startswith('e1-degree-protected')
    waiting=c.ROOT/'runtime/e1-degree-protected-wait.json'
    while not (OUT/'result.json').exists():
        if (OUT/'failure.json').exists():
            raise RuntimeError('degree validation failed; E1 remains unstarted')
        e.save(waiting,dict(stage='WAITING_FOR_DEGREE_VALIDATION',updated_unix=time.time()))
        time.sleep(15)
    result=json.loads((OUT/'result.json').read_text())
    assert result['status']=='G1_AND_1850K_REPLAY_PASSED'
    for name,digest in result['sources'].items():assert c.sha(c.ROOT/'implementation'/name)==digest
    setup(e.COLD)
    # Keep the old failed graph and all its evidence untouched. Only the new
    # trajectory archives use this fresh PMEM1 directory.
    archive=Path('/mnt/pmem1/hvs-update-basic-2026-11-16')/(e.COLD.name+'-variants')
    assert shutil.disk_usage(str(archive.parent)).free>60*2**30,'insufficient new archive capacity'
    archive.mkdir(exist_ok=False)
    (e.COLD/'variants').rmdir();(e.COLD/'variants').symlink_to(archive,target_is_directory=True)
    (c.ROOT/'runtime'/e.COLD.name).symlink_to(e.COLD,target_is_directory=True)
    e.save(waiting,dict(stage='VALIDATION_PASSED_STARTING_FRESH_E1',updated_unix=time.time(),root=str(e.COLD)))
    e.main()


if __name__=='__main__':main()
