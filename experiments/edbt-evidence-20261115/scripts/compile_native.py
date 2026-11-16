"""Audit native count/objective, compile isolated plain and counted beam objects."""
import re,shlex,shutil
import common as c
OUT=c.RUN/'native'
def transform(source,count=True):
    matches=list(re.finditer(r'void SSDIndex<T, TagT>::do_beam_search\(',source))
    if len(matches)!=2:raise RuntimeError('dual beam scope changed')
    start=matches[1].start()
    begin=source.index('{',start);end=begin+1;depth=1
    while depth:
        depth+=(source[end]=='{')-(source[end]=='}');end+=1
    body=source[start:end]
    one='alpha*dist_scratch[i]+ (1-alpha)*distl_scratch[i]'
    two='alpha*nbor_diste + (1-alpha)*nbor_distl'
    if body.count(one)!=1 or body.count(two)!=1:raise RuntimeError('native PQ objective sites changed')
    body=body.replace(one,'alpha*std::sqrt(std::max(0.f,dist_scratch[i]))+(1-alpha)*std::sqrt(std::max(0.f,distl_scratch[i]))')
    body=body.replace(two,'alpha*std::sqrt(std::max(0.f,nbor_diste))+(1-alpha)*std::sqrt(std::max(0.f,nbor_distl))')
    if count:
        def replace(old,new,expected=1):
            nonlocal body
            if body.count(old)!=expected:raise RuntimeError('count hook changed '+old[:50]+' '+str(body.count(old)))
            body=body.replace(old,new)
        replace('uint32_t original_l_search = l_search;','native_measure::counts = {};\n    uint32_t original_l_search = l_search;')
        replace('::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);','native_measure::counts.pq_candidates += n_ids;\n      native_measure::counts.pq_spaces += n_ids;\n      ::aggregate_coords(ids, n_ids, this->data.data(), this->n_chunks, pq_coord_scratch);')
        replace('::aggregate_coords(ids, n_ids, this->datal.data(), this->n_chunks, pq_coordl_scratch);','native_measure::counts.pq_spaces += n_ids;\n      ::aggregate_coords(ids, n_ids, this->datal.data(), this->n_chunks, pq_coordl_scratch);')
        replace('num_ios++;','native_measure::counts.requested++;\n          native_measure::counts.requested_bytes += size_per_io;\n          num_ios++;')
        replace('float cur_expanded_dist1 =','native_measure::counts.expanded++;\n        native_measure::counts.exact_spaces += 2;\n        float cur_expanded_dist1 =')
        # n_cmps now explicitly means per-space PQ evaluations, entries included.
        replace('stats->total_us = (double) query_timer.elapsed();','stats->n_cmps = native_measure::counts.pq_spaces;\n      stats->read_size = native_measure::counts.requested_bytes;\n      stats->total_us = (double) query_timer.elapsed();')
    return '#include "native_counts.h"\n'+source[:start]+body+source[end:]

def main():
    for n in ('build','bin','validation'):(OUT/n).mkdir(parents=True,exist_ok=True)
    flagsfile=c.E.parent/'build-single-baseline/tests/CMakeFiles/search_disk_index.dir/flags.make'
    flags=[];text=flagsfile.read_text()
    for key in ('CXX_DEFINES','CXX_INCLUDES','CXX_FLAGS'):flags+=shlex.split(re.search(r'^'+key+r' = (.*)$',text,re.M).group(1))
    if '-DNO_MAPPING' not in flags or '-DUSE_TOPO_DISK' in flags:raise RuntimeError('native layout compilation flags')
    flags+=['-I'+str(c.ROOT/'scripts'),'-I'+str(c.E/'io-time-numa3-20261108/scripts'),'-fno-fast-math','-ffp-contract=off']
    base=c.E.parent/'build-single-baseline/src/libpipeann.a';src=c.E.parent/'source/src/search/beam_search.cpp'
    # Existing immutable source identity provides a human-checkable anchor.
    original=src.read_bytes().decode('utf-8');link=['-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas'];binaries={}
    for name,count in [('plain',False),('counted',True)]:
        folder=OUT/'build'/name;folder.mkdir(exist_ok=True);p=folder/'beam_search.cpp';p.write_bytes(transform(original,count).encode('utf-8'))
        obj=folder/'beam_search.cpp.o';lib=folder/'libpipeann.a';shutil.copyfile(base,lib)
        for cmd,tag in [(['clang-format','-style=LLVM','-i',p],'format'),(['g++-11']+flags+['-c',p,'-o',obj],'object'),(['ar','r',lib,obj],'archive'),(['g++-11']+flags+[c.ROOT/'scripts/native_search.cpp',lib]+link+['-o',OUT/'bin'/('native-'+name)],'driver')]:
            c.run(cmd,'native-compile-'+name+'-'+tag,cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
        binaries[name]=c.sha(OUT/'bin'/('native-'+name))
    c.save(OUT/'validation/build.json',dict(status='COMPILED_GATES_PENDING',source_sha256=c.sha(src),library_sha256=c.sha(base),binaries=binaries,objective_change='two dual-PQ fusion sites now sqrt-per-space to match corrected build/GT/exact rerank; native beam/reader preserved',counting='PQ candidates once and spaces twice incl entries, expanded/exact separately, n_cmps assigned spaces, read_size in bytes'))
if __name__=='__main__':main()
