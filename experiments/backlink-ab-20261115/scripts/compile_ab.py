"""Isolated policy switch; default A body is kept verbatim."""
import shutil
import common as c
OUT=c.RUN
B_BODY='''
  const auto range=parameter.Get<unsigned>("R");
  for(const auto &des:pruned_list) {
    // The lock covers read, pruning and replacement. Distances are immutable
    // stored distances to d, including the symmetric pair from the source edge.
    v2::LockGuard guard(_locks->wrlock(des.id_));
    auto &neighbors=_final_graph[des.id_];
    std::vector<NeighborH> candidates;
    tsl::robin_set<unsigned> seen;
    auto add=[&](unsigned id,float e,float g) {
      if(id!=des.id_ && seen.insert(id).second)
        candidates.emplace_back(id,e,g,true,0);
    };
    add(n,des.emb_distance_,des.geo_distance_);
    for(const auto &old:neighbors)add(old.id_,old.emb_distance_,old.geo_distance_);
    std::vector<DEGNeighbor> selected;
    ++hvs_ab::reprune_calls;
    prune_neighbors_no_opt(candidates,parameter,selected);
    neighbors.assign(selected.begin(),selected.end());
  }
'''
def variant(text):
    start=text.index('void HybridIndex<T, TagT>::inter_insert(');begin=text.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    original=text[begin+1:end-1]
    wrapped='''{
  ++hvs_ab::insert_calls;
  hvs_ab::reverse_targets.fetch_add(pruned_list.size(),std::memory_order_relaxed);
#if HVS_REVERSE_ALWAYS_PRUNE
'''+B_BODY+'\n#else\n'+original+'\n#endif\n}'
    out=text[:begin]+wrapped+text[end:]
    needle='if (_final_graph[node].size() > range) {'
    if out.count(needle)!=1:raise RuntimeError('final cleanup observation hook changed')
    out=out.replace(needle,needle+'\n      ++hvs_ab::cleanup_changes;')
    return '#include "ab_stats.h"\n'+out,original

def main():
    for name in ('bin','build','validation','logs','commands'):(OUT/name).mkdir(parents=True,exist_ok=True)
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json');proof=c.load(c.U/'stage2-v2/validation/semantic-proof.json')['binaries']['unified'];source=c.U/'stage2-v2/variant/unified/hybrid_index.cpp';base=c.U/'stage2-v2/bin/libpipeann-unified.a'
    if c.sha(source)!=proof['source_sha256'] or c.sha(base)!=proof['library_sha256']:raise RuntimeError('formal source/library identity changed')
    text,body=variant(source.read_bytes().decode('utf-8'));path=OUT/'build/hybrid_index.cpp';path.write_bytes(text.encode('utf-8'))
    bsrc=c.U/'stage2-v2/variant/build_graph.cpp';bt=bsrc.read_bytes().decode('utf-8');needle='hvs_write_pruning_stats((std::string(argv[8])+"/pruning-counters.json").c_str());'
    # Formatting of the immutable wrapper can span multiple lines.
    import re
    pat=r'hvs_write_pruning_stats\(\s*\(std::string\(argv\[8\]\)\s*\+\s*"/pruning-counters.json"\)\.c_str\(\)\);'
    if len(re.findall(pat,bt))!=1:raise RuntimeError('builder stats hook changed')
    bt='extern "C" void hvs_ab_write_stats(const char *);\n'+re.sub(pat,lambda m:m.group(0)+'\n    hvs_ab_write_stats((std::string(argv[8])+"/reverse-insert-stats.json").c_str());',bt)
    builder=OUT/'build/build_graph.cpp';builder.write_bytes(bt.encode('utf-8'))
    flags=cfg['flags']+['-I'+str(c.U/'scripts'),'-I'+str(c.ROOT/'scripts'),'-fno-fast-math','-ffp-contract=off'];binaries={}
    c.run(['clang-format','-style=LLVM','-i',c.ROOT/'scripts/ab_stats.h',builder],'format-ab-wrapper',cpus='80,81',timeout=60)
    # Do not reformat this TU: keeping original A body byte-exact is auditable.
    if body not in path.read_bytes().decode('utf-8'):raise RuntimeError('A body changed')
    for mode in ('A','B'):
        root=OUT/'build'/mode;root.mkdir(exist_ok=True);lib=root/'libpipeann.a';shutil.copyfile(base,lib);obj=root/'hybrid_index.cpp.o'
        options=flags+['-DHVS_REVERSE_ALWAYS_PRUNE='+str(int(mode=='B'))]
        for cmd,tag in [(['g++-11']+options+['-c',path,'-o',obj],'object'),(['ar','r',lib,obj],'archive'),(['g++-11']+options+[builder,lib]+cfg['link']+['-o',OUT/'bin'/('build-'+mode)],'builder')]:
            c.run(cmd,'compile-'+mode+'-'+tag,cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
        binaries[mode]=dict(binary=c.sha(OUT/'bin'/('build-'+mode)),library=c.sha(lib),flags=options)
    c.save(OUT/'validation/build.json',dict(status='COMPILED_GATES_PENDING',formal_source=c.sha(source),base_library=c.sha(base),variant=c.sha(path),builder=c.sha(builder),header=c.sha(c.ROOT/'scripts/ab_stats.h'),binaries=binaries,A_body_verbatim=True,B_lock='same d write lock from read through prune and assignment',B_distances='stored distances only; no pair distances recomputed in inter_insert; pruning uses unchanged formal pair checks',link=cfg['link']))
if __name__=='__main__':main()
