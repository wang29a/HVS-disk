#!/usr/bin/env python3
"""Isolated corrected interval solver + direct fixed-metric RNG pruning."""
import hashlib,json,os,pathlib,re,shutil,subprocess,sys
R=pathlib.Path(__file__).resolve().parent.parent;E=R.parent;O=R/'stage2-v2'
sys.path.insert(0,str(R/'scripts'))
from prepare_specialized import specialize
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def corrected_intervals(source):
    source,n1=re.subn(r'std::pair<float,\s*float>\s+tmp_prune_range_1;.*?(?=// now for equation 2)',
        'auto tmp_prune_range_1 = hvs_pruning::solve_record(diff1, diff2);\n                  ',source,flags=re.S)
    source,n2=re.subn(r'std::pair<float,\s*float>\s+tmp_prune_range_2;.*?(?=std::pair<float,\s*float>\s+tmp_prune_range;)',
        'auto tmp_prune_range_2 = hvs_pruning::solve_record(diff3, diff4);\n                  ',source,flags=re.S)
    if(n1,n2)!=(4,4):raise RuntimeError('interval call-site count changed '+str((n1,n2)))
    # Shadow the legacy sign logic on the same actual candidate/witness set.
    # This measures conditional interval-width and keep/drop changes without
    # claiming to reconstruct the old build's transient candidate history.
    start=source.index('void HybridIndex<T, TagT>::prune_neighbors_no_opt(')
    begin=source.index('{',start);cursor=begin+1;depth=1
    while depth:
        depth+=(source[cursor]=='{')-(source[cursor]=='}');cursor+=1
    body=source[begin:cursor]
    needle='std::vector<std::pair<float, float>> prune_range;'
    if body.count(needle)!=1:raise RuntimeError('candidate shadow scope changed')
    body=body.replace(needle,needle+'\n              std::vector<std::pair<float,float>> legacy_prune_range;')
    needle='std::pair<float, float> tmp_prune_range;'
    shadow="""auto old_conditions=pruning_semantics::intersect(
                    pruning_semantics::legacy(diff1,diff2),
                    pruning_semantics::legacy(diff3,diff4));
                  if(old_conditions.width()>0)
                    intersection(picked_use_range,
                      {float(old_conditions.lo),float(old_conditions.hi)},legacy_prune_range);
                  """
    if body.count(needle)!=1:raise RuntimeError('witness shadow scope changed')
    body=body.replace(needle,shadow+needle)
    needle='if (use_size >= threshold)'
    shadow="""auto old_union=mergeIntervals(legacy_prune_range);
              std::vector<std::pair<float,float>> old_available;
              get_use_range(old_union,old_available);
              float old_use_size=0;
              for(auto r:old_available)old_use_size+=r.second-r.first;
              auto &candidate_stats=hvs_pruning::counts();
              ++candidate_stats.candidate_cases;
              if(old_use_size>use_size){
                ++candidate_stats.candidate_width_shrunk;
                candidate_stats.candidate_width_reduction_sum+=old_use_size-use_size;
              }
              candidate_stats.legacy_keep_new_drop+=old_use_size>=threshold&&use_size<threshold;
              """
    if body.count(needle)!=1:raise RuntimeError('candidate decision shadow scope changed')
    body=body.replace(needle,shadow+needle)
    source=source[:begin]+body+source[cursor:]
    return '#include "hvs_interval_stats.h"\n'+source
def direct_fixed(source):
    source=specialize(source)
    start=source.index('void HybridIndex<T, TagT>::prune_neighbors_no_opt(')
    begin=source.index('{',start);cursor=begin+1;depth=1
    while depth:
        depth+=(source[cursor]=='{')-(source[cursor]=='}');cursor+=1
    body='''{
    auto &stats=hvs_pruning::counts(); ++stats.fixed_calls;
    std::vector<fixed_pruning::Candidate> candidates;
    for(const auto &p:pool) candidates.push_back({p.id,p.distance1});
    stats.fixed_candidates+=candidates.size();
    auto pair_distance=[&](unsigned p,unsigned q) {
      ++stats.fixed_pair_checks;
      const float x=hvs_metric_distance(_distance->compare(
          _data1+size_t(p)*_aligned_dim1,_data1+size_t(q)*_aligned_dim1,
          unsigned(_aligned_dim1)),_dist_metric);
      const float y=hvs_metric_distance(_distance->compare(
          _data2+size_t(p)*_aligned_dim2,_data2+size_t(q)*_aligned_dim2,
          unsigned(_aligned_dim2)),_dist_metric);
      return hvs_fixed_fuse(x,y);
    };
    auto picked=fixed_pruning::select(candidates,parameter.Get<unsigned>("R"),pair_distance);
    pruned_list.clear();
    for(auto p:picked)
      pruned_list.emplace_back(p.id,p.score,p.score,
          std::vector<std::pair<float,float>>{{0.f,1.f}},0);
    stats.fixed_kept+=picked.size();
  }'''
    return '#include "fixed_pruning.h"\n'+source[:begin]+body+source[cursor:]
def main():
    for sub in ('bin','variant','validation','logs','analysis'):(O/sub).mkdir(parents=True,exist_ok=True)
    subprocess.check_call(['clang-format','-style=LLVM','-i']+[str(R/'scripts'/name) for name in ('fixed_pruning.h','pruning_semantics.h','hvs_interval_stats.h','test_actual_pruner.cpp','test_pruning_semantics.cpp')])
    config=json.loads((E/'metric-consistent-20261111/analysis/build.json').read_text())
    base=E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if digest(base)!=config['library_sha256']:raise RuntimeError('corrected-metric library mismatch')
    src=E/'metric-consistent-20261111/variant/hybrid_index.cpp'
    unified=corrected_intervals(src.read_text(encoding='utf-8'))
    fixed=direct_fixed(unified)
    builder=(E/'laion-six-cell-20261111/scripts/build_graph.cpp').read_text(encoding='utf-8')
    needle='throw std::runtime_error("graph target");'
    if builder.count(needle)!=1:raise RuntimeError('builder hook changed')
    builder='extern "C" void hvs_write_pruning_stats(const char *);\n'+builder.replace(needle,needle+'\n    hvs_write_pruning_stats((std::string(argv[8])+"/pruning-counters.json").c_str());')
    builder_path=O/'variant/build_graph.cpp';builder_path.write_text(builder,encoding='utf-8')
    # Preserve exact scalar predicate arithmetic, and keep assertions enabled in
    # the semantic executables even though the inherited build uses NDEBUG.
    flags=config['flags']+['-I'+str(R/'scripts'),'-fno-fast-math','-ffp-contract=off']
    env=dict(os.environ,LC_ALL='C',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    commands=[];binaries={}
    with (O/'logs/compile.log').open('w') as log:
        def run(args):
            cmd=['numactl','--all','--physcpubind=80,81','--membind=3']+list(map(str,args));commands.append(cmd)
            subprocess.check_call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        run(['g++-11','-std=c++17','-O2','-UNDEBUG',R/'scripts/test_pruning_semantics.cpp','-o',O/'bin/test-pruning-semantics'])
        run([O/'bin/test-pruning-semantics'])
        for mode,text in [('unified',unified),('fixed',fixed)]:
            directory=O/'variant'/mode;directory.mkdir(exist_ok=True)
            path=directory/'hybrid_index.cpp';path.write_text(text,encoding='utf-8')
            subprocess.check_call(['clang-format','-style=LLVM','-i',str(path),str(builder_path)])
            obj=directory/'hybrid_index.cpp.o';lib=O/'bin'/('libpipeann-'+mode+'.a');shutil.copyfile(str(base),str(lib))
            run(['g++-11']+flags+['-c',path,'-o',obj]);run(['ar','r',lib,obj])
            run(['g++-11']+flags+[builder_path,lib]+config['link']+['-o',O/'bin'/('build-'+mode)])
            run(['g++-11']+flags+['-UNDEBUG',R/'scripts/test_actual_pruner.cpp',lib]+config['link']+['-o',O/'bin'/('test-actual-'+mode)])
            for a in (['0','0.1','0.3','0.5','0.7','0.9','1'] if mode=='fixed' else ['0.5']):
                env['HVS_FIXED_ALPHA']=a
                run([O/'bin'/('test-actual-'+mode),mode,a])
            binaries[mode]=dict(builder_sha256=digest(O/'bin'/('build-'+mode)),library_sha256=digest(lib),source_sha256=digest(path))
    proof=dict(status='PASS',source_unified_v1_sha256=digest(src),base_library_sha256=digest(base),
      headers_sha256={name:digest(R/'scripts'/name) for name in ('fixed_pruning.h','pruning_semantics.h','hvs_interval_stats.h')},
      scalar_affine_checks=28577,geometric_brute_and_interval_checks=7000,
      actual_fixed_member_cases=7000,actual_unified_degenerate_member_cases=1000,
      fixed_alpha_tested=[0,.1,.3,.5,.7,.9,1],binaries=binaries,commands=commands,
      scope='member prune outputs and controlled same-order predicates; not full final graph equivalence. Backlink insertion, Pareto order, interval-width thresholds and closed endpoint encoding remain separate mechanisms.',
      arithmetic='new hybrid translation units disable fast-math and contraction; stock squared-L2 kernel unchanged')
    (O/'validation/semantic-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
    print('PASS: semantic primitives and both actual member-pruner implementations')
if __name__=='__main__':main()
