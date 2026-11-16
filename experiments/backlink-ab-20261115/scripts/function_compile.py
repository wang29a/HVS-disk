"""One capture-only B, actual B replay, and eight minimal author fix masks."""
import hashlib,re,shutil
import common as c
import compile_ab
F=c.RUN/'function-v2'
FIXES={1:'strict_degree_cap',2:'merge_last_interval',4:'zero_slope_and_rhs'}
ZERO='''
inline std::pair<float,float> hvs_fixed_prune_solve(float slope,float rhs) {
  if(slope==0)return rhs>=0?std::make_pair(0.f,1.f):std::make_pair(0.f,0.f);
  float cut=rhs/slope;
  float lo=slope>0?0.f:std::max(0.f,cut);
  float hi=slope>0?std::min(1.f,cut):1.f;
  return lo>hi?std::make_pair(0.f,0.f):std::make_pair(lo,hi);
}
'''

def author_variant(mask):
    src=c.RUN/'author-original';out=F/'author'/str(mask)
    if out.exists():shutil.rmtree(out)
    out.mkdir(parents=True);shutil.copytree(src/'include',out/'include');shutil.copytree(src/'src',out/'src')
    header=out/'include/component.h';ht=header.read_bytes().decode('utf8')
    start=ht.index('class ComponentDEGPruneHeuristic');end=ht.find('\n    class ',start+10)
    if end<0:end=len(ht)
    scope=ht[start:end]
    if mask&2:
        old='for (size_t i = 1; i < intervals.size() - 1; i++)'
        if scope.count(old)!=1:raise RuntimeError('author merge patch scope')
        scope=scope.replace(old,'for (size_t i = 1; i < intervals.size(); i++)')
    header.write_bytes((ht[:start]+scope+ht[end:]).encode('utf8'))
    source=out/'src/component_prune.cpp';t=source.read_bytes().decode('utf8')
    if mask&1:
        old='for (int i = 0; i < candidate.size(); i++)'
        if t.count(old)!=1:raise RuntimeError('author cap patch scope')
        t=t.replace(old,'for (int i = 0; i < candidate.size() && picked.size() < range; i++)')
    if mask&4:
        for which,stop in [(1,'// now for equation 2'),(2,'std::pair<float, float> tmp_prune_range;')]:
            begin=t.index('std::pair<float, float> tmp_prune_range_'+str(which)+';');finish=t.index(stop,begin)
            args=('diff1','diff2') if which==1 else ('diff3','diff4')
            t=t[:begin]+f'auto tmp_prune_range_{which} = hvs_fixed_prune_solve({args[0]}, {args[1]});\n                    '+t[finish:]
        t=t.replace('#include "component.h"','#include "component.h"\n'+ZERO,1)
    source.write_bytes(t.encode('utf8'))
    # Audit-only distance injection. Whole-graph binaries do not define it.
    distance=out/'include/distance.h';dt=distance.read_bytes().decode('utf8')
    old='T emb_distance = sqr_dist(a, b, length);'
    if dt.count(old)!=1:raise RuntimeError('author distance hook')
    dt=dt.replace(old,'#ifdef HVS_FUNCTION_ORACLE\n            return static_cast<T>(hvs_author_oracle(reinterpret_cast<const float*>(a),reinterpret_cast<const float*>(b),length));\n#endif\n            '+old)
    dt='float hvs_author_oracle(const float *,const float *,unsigned);\n'+dt
    distance.write_bytes(dt.encode('utf8'))
    c.save(out/'adaptation.json',dict(mask=mask,fixes=[v for k,v in FIXES.items() if mask&k],scope='only3 algorithm fixes; distance provider replaced only under HVS_FUNCTION_ORACLE',files={str(p.relative_to(out)):c.sha(p) for p in (header,source,distance)}))
    return out

def main():
    for name in ('bin','build','results','validation'):(F/name).mkdir(parents=True,exist_ok=True)
    upstream=c.load(c.ROOT/'AUTHOR-UPSTREAM.json')
    for name,h in upstream['files'].items():
        if c.sha(c.RUN/'author-original'/name)!=h:raise RuntimeError('pinned author source changed '+name)
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');old=c.load(c.RUN/'validation/build.json')
    flags=old['binaries']['B']['flags'];base=c.U/'stage2-v2/bin/libpipeann-unified.a'
    source=c.U/'stage2-v2/variant/unified/hybrid_index.cpp'
    if c.sha(source)!=old['formal_source'] or c.sha(base)!=old['base_library']:raise RuntimeError('formal identities changed')
    t,_=compile_ab.variant(source.read_bytes().decode('utf8'))
    hook='prune_neighbors_no_opt(candidates,parameter,selected);'
    if t.count(hook)!=1:raise RuntimeError('B capture hook changed')
    replacement='''auto function_input=candidates;
    prune_neighbors_no_opt(candidates,parameter,selected);
    hvs_capture::record(des.id_,n,function_input,selected,[&](unsigned a,unsigned b,unsigned space) {
      auto *data=space?_data2:_data1;auto dim=space?_aligned_dim2:_aligned_dim1;
      return _distance->compare(data+size_t(a)*dim,data+size_t(b)*dim,unsigned(dim));
    });'''
    t='#include "trace_capture.h"\n'+t.replace(hook,replacement)
    cap=F/'build/capture.cpp';cap.write_bytes(t.encode('utf8'));obj=F/'build/hybrid_index.cpp.o';lib=F/'build/libpipeann-capture.a';shutil.copyfile(base,lib)
    for cmd,tag in [(['clang-format','-style=LLVM','-i',cap],'format'),(['g++-11']+flags+['-c',cap,'-o',obj],'object'),(['ar','r',lib,obj],'archive'),(['g++-11']+flags+[c.RUN/'build/build_graph.cpp',lib]+cfg['link']+['-o',F/'bin/build-B-capture'],'link')]:c.run(cmd,'fn-capture-'+tag,cpus='80,81',timeout=600)
    b_lib=c.RUN/'build/B/libpipeann.a'
    if c.sha(b_lib)!=old['binaries']['B']['library']:raise RuntimeError('original B library identity changed')
    c.run(['g++-11']+flags+[c.ROOT/'scripts/replay_b.cpp',b_lib]+cfg['link']+['-o',F/'bin/replay-B'],'fn-compile-replay-B',cpus='80,81',timeout=600)
    af=[v for v in cfg['flags'] if not v.startswith('-I')]+['-I'+str(c.ROOT/'scripts'),'-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections']
    binaries={};sources={}
    for mask in range(8):
        folder=author_variant(mask);opts=af+['-I'+str(folder/'include'),'-DHVS_FUNCTION_ORACLE'];o=F/'build'/('author-'+str(mask)+'.o');binary=F/'bin'/('replay-author-'+str(mask))
        c.run(['g++-11']+opts+['-c',folder/'src/component_prune.cpp','-o',o],'fn-author-'+str(mask)+'-object',cpus='80,81',timeout=600)
        c.run(['g++-11']+opts+[c.ROOT/'scripts/replay_author.cpp',o,'-Wl,--gc-sections','-o',binary],'fn-author-'+str(mask)+'-link',cpus='80,81',timeout=600)
        binaries[str(mask)]=c.sha(binary);sources[str(mask)]=c.load(folder/'adaptation.json')
    # A corrected-author whole-graph reference with original vector distances.
    folder=F/'author/7';objects=[];wholeopts=af+['-I'+str(folder/'include')]
    for unit in ('component_init.cpp','component_prune.cpp'):
        o=F/'build'/('whole-'+unit+'.o');c.run(['g++-11']+wholeopts+['-c',folder/'src'/unit,'-o',o],'fn-whole-'+unit,cpus='80,81',timeout=600);objects.append(o)
    c.run(['g++-11']+wholeopts+[c.ROOT/'scripts/author_driver.cpp']+objects+['-Wl,--gc-sections','-o',F/'bin/build-author-fixed'],'fn-whole-link',cpus='80,81',timeout=600)
    c.save(F/'validation/build.json',dict(status='COMPILED_GATES_PENDING',formal_source=c.sha(source),original_B_library=c.sha(b_lib),capture_source=c.sha(cap),capture_binary=c.sha(F/'bin/build-B-capture'),replay_B=c.sha(F/'bin/replay-B'),author_binaries=binaries,author_sources=sources,B_flags=flags,author_flags=af,author_whole_flags=wholeopts,whole_author=c.sha(F/'bin/build-author-fixed'),source_scripts={p.name:c.sha(p) for p in (c.ROOT/'scripts').iterdir() if p.suffix in ('.py','.cpp','.h')}))

if __name__=='__main__':main()
