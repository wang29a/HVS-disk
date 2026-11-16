"""Isolated corrected unified pruner with a relaxed pair-distance predicate."""
import shutil,re
import common as c
OUT=c.RUN/'relaxed'
def transform(source):
    start=source.index('void HybridIndex<T, TagT>::prune_neighbors_no_opt(')
    begin=source.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    body=source[begin:end]
    needle='float diff3 = xq_e_dist - cur_emb_dist + cur_geo_dist - xq_s_dist;'
    if body.count(needle)!=1:raise RuntimeError('corrected unified hook changed')
    body=body.replace(needle,'xq_e_dist *= hvs_unified_rho();\n        xq_s_dist *= hvs_unified_rho();\n        '+needle)
    return '#include "relaxation.h"\n'+source[:begin]+body+source[end:]
def main():
    for n in ('bin','build','validation'):(OUT/n).mkdir(parents=True,exist_ok=True)
    proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')['binaries']['unified']
    src=c.U/'stage2-v2/variant/unified/hybrid_index.cpp';base=c.U/'stage2-v2/bin/libpipeann-unified.a'
    if c.sha(src)!=proof['source_sha256'] or c.sha(base)!=proof['library_sha256']:raise RuntimeError('unified immutable inputs changed')
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json')
    flags=cfg['flags']+['-I'+str(c.U/'scripts'),'-I'+str(c.ROOT/'scripts'),'-fno-fast-math','-ffp-contract=off']
    path=OUT/'build/hybrid_index.cpp';path.write_bytes(transform(src.read_bytes().decode('utf-8')).encode('utf-8'))
    test=(c.U/'scripts/test_actual_pruner.cpp').read_text()
    test=test.replace('#include "index.h"','#include "index.h"\n#include "relaxation.h"')
    if test.count('dist(p, q) <= dist(0, q)')!=1:raise RuntimeError('reference hook changed')
    test=test.replace('dist(p, q) <= dist(0, q)','hvs_unified_rho() * dist(p, q) <= dist(0, q)')
    test=test.replace('dist(got[j].id_, got[i].id_) <= dist(0, got[i].id_)','hvs_unified_rho() * dist(got[j].id_, got[i].id_) <= dist(0, got[i].id_)')
    testpath=OUT/'build/test_member.cpp';testpath.write_text(test)
    def run(cmd,name):return c.run(cmd,name,cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    run(['clang-format','-style=LLVM','-i',path,testpath,c.ROOT/'scripts/relaxation.h'],'format-relaxed')
    lib=OUT/'bin/libpipeann-relaxed.a';shutil.copyfile(base,lib);obj=OUT/'build/hybrid_index.cpp.o'
    run(['g++-11']+flags+['-c',path,'-o',obj],'compile-relaxed-object');run(['ar','r',lib,obj],'archive-relaxed')
    run(['g++-11']+flags+[c.U/'stage2-v2/variant/build_graph.cpp',lib]+cfg['link']+['-o',OUT/'bin/build-relaxed'],'compile-relaxed-builder')
    run(['g++-11']+flags+['-UNDEBUG',testpath,lib]+cfg['link']+['-o',OUT/'bin/test-member'],'compile-relaxed-member')
    for rho in (1,1.025,1.05):
        c.run([OUT/'bin/test-member','unified','.5'],'test-relaxed-member-'+str(rho),cpus='80,81',timeout=120,extra=dict(HVS_UNIFIED_RHO=str(rho)))
    c.save(OUT/'validation/build.json',dict(status='PASS',source_sha256=c.sha(src),variant_sha256=c.sha(path),base_library_sha256=c.sha(base),builder_sha256=c.sha(OUT/'bin/build-relaxed'),member_cases=3000,rho=[1,1.025,1.05],scope='actual unified member versus independent scalar reference on identical-space geometry; not final graph identity'))
if __name__=='__main__':main()
