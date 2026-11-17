"""Validate actual inserted codes against the upstream online encoder."""
import fcntl
import json
from pathlib import Path
import run_g0 as c
import run_e1 as e
from e1_evaluate import snapshot_codes

OUT=c.ROOT/'runtime/online-pq-audit-v1'
OLD=Path('/mnt/pmem0/hvs-update-basic-2026-11-16/e1-degree-protected')


def main():
    OUT.mkdir(exist_ok=False)
    e.COLD=OUT
    lock=(e.E/'budget-reorder-20261103/queue.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX)
    src=c.ROOT/'implementation';gate=json.loads((c.ROOT/'runtime/g1-in-degree-protection-v2/build-identity.json').read_text())
    e.run(['clang-format','-style=LLVM','-i',src/'online_pq_audit.cpp'],'format-online-audit',out=OUT)
    command=['g++-11']+gate['flags']+[src/'online_pq_audit.cpp',c.BUILD/'src/libpipeann.a',
        '-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-pthread','-o',OUT/'encoder']
    e.run(command,'compile-online-audit',out=OUT)
    e.run([OUT/'encoder',OLD/'inputs',e.MG/'data',OUT],'encode-all-200k-insertions',out=OUT)
    checks=[]
    for label,root in [('formal',OLD/'variants/pq-t4-r1/checkpoint-1'),
                       ('replay',Path('/mnt/pmem0/hvs-update-basic-2026-11-16/degree-replay/variants/pq-counterexample-replay/checkpoint-1'))]:
        dest=OUT/(label+'-snapshot-check');dest.mkdir()
        snapshot_codes(root/'normal-exit.snapshot',dest,e.MG/'data',OUT)
        proof=json.loads((dest/'pq-code-audit.json').read_text());checks.append(dict(label=label,proof=proof))
        e.save(dest/'actual-derivatives-sha256.json',{p.name:c.sha(p) for p in dest.glob('*_compressed.bin')})
        for p in dest.glob('*_compressed.bin'):p.unlink()
    e.save(OUT/'identity.json',dict(status='ONLINE_CODES_MATCH_SCALAR_ENCODER',checks=checks,
        command=list(map(str,command)),flags=gate['flags'],source_sha256=c.sha(src/'online_pq_audit.cpp'),
        upstream_encoder_header=c.sha(c.SOURCE/'include/pq_table.h'),binary=c.sha(OUT/'encoder'),
        files={p.name:c.sha(p) for p in OUT.glob('*_compressed.bin')},
        inputs={str(OLD/'inputs'/name):c.sha(OLD/'inputs'/name) for name in ('insert-x.bin','insert-y.bin')},
        pivots={space:c.sha(e.MG/'data'/(space+'_pq_pivots.bin')) for space in ('emb','loc')},
        purpose='audit reference only; update executor still computes every inserted code online'))


if __name__=='__main__':main()
