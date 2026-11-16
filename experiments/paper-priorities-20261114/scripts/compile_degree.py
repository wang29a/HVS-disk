"""Rebuild only an isolated HybridIndex object, never the original builder."""
import shutil
import common as c
OUT=c.RUN/'degree'
def main():
    for n in ('bin','build','validation'):(OUT/n).mkdir(parents=True,exist_ok=True)
    proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')['binaries']['fixed']
    src=c.U/'stage2-v2/variant/fixed/hybrid_index.cpp';base=c.U/'stage2-v2/bin/libpipeann-fixed.a'
    if c.sha(src)!=proof['source_sha256'] or c.sha(base)!=proof['library_sha256']:raise RuntimeError('v2 fixed implementation changed')
    target=OUT/'build/hybrid_index.cpp';shutil.copyfile(src,target);shutil.copyfile(c.ROOT/'scripts/fixed_pruning.h',OUT/'build/fixed_pruning.h')
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json')
    flags=cfg['flags']+['-I'+str(c.U/'scripts'),'-I'+str(c.ROOT/'scripts'),'-fno-fast-math','-ffp-contract=off']
    lib=OUT/'bin/libpipeann-strength.a';shutil.copyfile(base,lib);obj=OUT/'build/hybrid_index.cpp.o'
    def run(cmd,name):return c.run(cmd,name,cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    run(['clang-format','-style=LLVM','-i',c.ROOT/'scripts/fixed_pruning.h',c.ROOT/'scripts/test_strength.cpp'],'format-strength')
    shutil.copyfile(c.ROOT/'scripts/fixed_pruning.h',OUT/'build/fixed_pruning.h')
    run(['g++-11']+flags+['-UNDEBUG',c.ROOT/'scripts/test_strength.cpp','-o',OUT/'bin/test-strength'],'compile-strength-test')
    for rho in (1,1.025,1.05,1.1,1.2,1.4,1.8,2):
        c.run([OUT/'bin/test-strength'],'test-strength-'+str(rho),cpus='80,81',timeout=60,extra=dict(HVS_PRUNE_RHO=str(rho)))
    run(['g++-11']+flags+['-c',target,'-o',obj],'compile-strength-object');run(['ar','r',lib,obj],'link-strength-library')
    builder=c.U/'stage2-v2/variant/build_graph.cpp'
    run(['g++-11']+flags+[builder,lib]+cfg['link']+['-o',OUT/'bin/build-strength'],'compile-strength-builder')
    # The original member test has an independently coded scalar reference.
    # At rho=1 the new member must match it including tie cases.
    run(['g++-11']+flags+['-UNDEBUG',c.U/'scripts/test_actual_pruner.cpp',lib]+cfg['link']+['-o',OUT/'bin/test-strength-member'],'compile-strength-member-test')
    c.run([OUT/'bin/test-strength-member','fixed','0.5'],'test-strength-member-rho1',cpus='80,81',timeout=120,extra=dict(HVS_FIXED_ALPHA='0.5',HVS_PRUNE_RHO='1'))
    c.save(OUT/'validation/build.json',dict(status='PASS',base_fixed_source_sha256=c.sha(src),header_sha256=c.sha(c.ROOT/'scripts/fixed_pruning.h'),
        builder_sha256=c.sha(OUT/'bin/build-strength'),library_sha256=c.sha(lib),original_fixed_builder_sha256=c.sha(c.U/'stage2-v2/bin/build-fixed'),
        geometry_cases=4000,rho1_actual_member_test=True,rho_grid=[1,1.025,1.05,1.1,1.2,1.4,1.8,2],metric='fixed .5 weighted Euclidean; LE; rho multiplies fused distance'))
if __name__=='__main__':main()
