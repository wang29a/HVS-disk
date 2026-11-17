from pathlib import Path
import fcntl
import json
import shutil
import time
import run_g0 as c
import extract_prune


def main():
    c.RUN = c.ROOT/'runtime/prune-gate'
    c.RUN.mkdir(parents=True, exist_ok=True)
    lock = (c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = dict(stage='running', updated_unix=time.time())
    c.save('status.json', status)
    try:
        g0 = c.ROOT/'runtime/g0-attempt2'
        assert json.loads((g0/'status.json').read_text())['stage'] == 'G0_PASSED'
        extract_prune.main()
        source = c.ROOT/'implementation'
        ab = c.HOST_ROOT/'experiments/backlink-ab-20261115'
        shutil.copyfile(ab/'scripts/function_io.h',source/'function_io.h')
        flags = json.loads((g0/'build-identity.json').read_text())['flags']
        c.command(['clang-format','-style=LLVM','-i', source/'deg_prune.h',
                   source/'deg_prune.cpp',source/'prune_replay.cpp'], 'format-prune')
        c.command(['g++-11']+flags+[source/'deg_prune.cpp',source/'prune_replay.cpp',
                   '-o',c.RUN/'prune_replay'], 'compile-prune')
        c.command([c.RUN/'prune_replay',ab/'runtime/function-v2/capture.bin'], 'semantic-replay', 120)
        c.save('build-identity.json',dict(flags=flags,
            sources={p.name:c.sha(p) for p in [source/'deg_prune.h',source/'deg_prune.cpp',source/'pruning_semantics.h',source/'prune_replay.cpp']},
            binary=c.sha(c.RUN/'prune_replay')))
        c.save('status.json',dict(stage='PRUNE_SEMANTICS_PASSED',updated_unix=time.time()))
    except Exception as e:
        c.save('status.json',dict(stage='STOPPED',reason=str(e),updated_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
