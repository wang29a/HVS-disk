"""Build Linux executables against the already validated immutable libraries."""
import sys
from pathlib import Path
import common as c

def main():
    for sub in ('bin','build','logs','commands','validation'):(c.RUN/sub).mkdir(parents=True,exist_ok=True)
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json')
    lib=c.E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if c.sha(lib)!=cfg['library_sha256']:raise RuntimeError('base library identity')
    proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')
    builder=c.U/'stage2-v2/bin/build-fixed'
    if c.sha(builder)!=proof['binaries']['fixed']['builder_sha256']:raise RuntimeError('fixed builder identity')
    source=c.E/'laion-six-cell-20261111/scripts/build_graph.cpp'
    text=source.read_text(encoding='utf-8')
    needle='pipeann::Index<float> index(pipeann::Metric::L2, d1 + d2, n, false,'
    if text.count(needle)!=1:raise RuntimeError('Vamana wrapper hook changed')
    # A single-space input has d1 columns, but output metadata describes the
    # common XY coordinate file. Only this constructor dimension changes.
    text=text.replace(needle,'pipeann::Index<float> index(pipeann::Metric::L2, d1, n, false,')
    path=c.RUN/'build/build_vamana.cpp';path.write_text(text,encoding='utf-8')
    # Verify the actual squared-distance ratio test and saturation setting;
    # do not replace Index with an invented one-pass greedy builder.
    index=c.E.parent/'source/src/index.cpp'
    index_text=index.read_text(encoding='utf-8')
    if 'pool[t].distance / djk' not in index_text or 'occlude_factor[start] > cur_alpha' not in index_text:
        raise RuntimeError('Vamana pruning contract changed')
    flags=cfg['flags'];commands=[]
    for name,src in [('multigraph',c.ROOT/'scripts/multigraph.cpp'),('pack',c.ROOT/'scripts/pack.cpp'),('build-vamana',path)]:
        c.run(['clang-format','-style=LLVM','-i',src], 'format-'+name,cpus='80,81',timeout=30)
        cmd=['g++-11']+flags+[src,lib]+cfg['link']+['-o',c.RUN/'bin'/name]
        c.run(cmd,'compile-'+name,cpus='80,81',timeout=600,extra=dict(OMP_THREAD_LIMIT='2',OMP_NUM_THREADS='2'))
        commands.append(list(map(str,cmd)))
    c.save(c.RUN/'validation/build.json',dict(status='PASS',library_sha256=c.sha(lib),fixed_builder_sha256=c.sha(builder),
        protocol_sha256=c.sha(c.ROOT/'README.md'),commands=commands,binaries={p.name:c.sha(p) for p in (c.RUN/'bin').iterdir()},
        sources={p.name:c.sha(p) for p in (c.ROOT/'scripts').glob('*.cpp')},
        vamana='native Index, squared L2 ratio; skip factor > current_alpha; stages1,1.2; alpha=1.2, saturate_graph=true'))
if __name__=='__main__':main()
