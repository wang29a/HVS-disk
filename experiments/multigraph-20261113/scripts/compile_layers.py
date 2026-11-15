"""Compile an isolated diagnostic, preserving the original executor binary."""
from pathlib import Path
import common as c

OUT=c.RUN/'three-layer-20261114'
def main():
    for sub in ('bin','build','validation','candidates','logs'):(OUT/sub).mkdir(parents=True,exist_ok=True)
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');lib=c.E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if c.sha(lib)!=cfg['library_sha256']:raise RuntimeError('base library changed')
    source=c.ROOT/'scripts/pq_layers.cpp'
    c.run(['clang-format','-style=LLVM','-i',source],'format-pq-layers',cpus='80,81',timeout=30)
    cmd=['g++-11']+cfg['flags']+['-I'+str(c.ROOT/'scripts'),source,lib]+cfg['link']+['-o',OUT/'bin/pq-layers']
    c.run(cmd,'compile-pq-layers',cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    c.run([OUT/'bin/pq-layers','--self-test'],'self-test-pq-layers',cpus='80,81',timeout=30)
    c.save(OUT/'validation/build.json',dict(status='PASS',source_sha256=c.sha(source),included_executor_sha256=c.sha(c.ROOT/'scripts/multigraph.cpp'),
        binary_sha256=c.sha(OUT/'bin/pq-layers'),original_binary_sha256=c.sha(c.RUN/'bin/multigraph'),library_sha256=c.sha(lib),command=list(map(str,cmd))))
if __name__=='__main__':main()
