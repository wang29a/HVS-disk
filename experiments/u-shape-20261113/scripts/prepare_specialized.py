#!/usr/bin/env python3
"""Create an isolated fixed-objective HybridIndex library, never patch main."""
import hashlib, json, pathlib, re, shutil, subprocess
R=pathlib.Path(__file__).resolve().parent.parent
E=R.parent
O=R/'stage2'
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def specialize(source):
    replacements=[]
    token='hvs_metric_distance('
    for match in re.finditer(re.escape(token),source):
        start=match.start(); depth=1; end=match.end()
        while depth:
            c=source[end]; depth+=(c=='(')-(c==')'); end+=1
        expr=source[start:end]
        one=('_data1' in expr or 'node_coords1' in expr or 'emb_center' in expr)
        two=('_data2' in expr or 'node_coords2' in expr or 'loc_center' in expr)
        if one==two: raise ValueError('ambiguous distance channels')
        a,b=expr,expr
        pairs=[('_data1','_data2'),('_aligned_dim1','_aligned_dim2'),('node_coords1','node_coords2'),('emb_center','loc_center')]
        for x,y in pairs:
            if one: b=re.sub(r'\b'+x+r'\b',y,b)
            else: a=re.sub(r'\b'+y+r'\b',x,a)
        replacements.append((start,end,'hvs_fixed_fuse('+a+', '+b+')'))
    if len(replacements)!=14: raise ValueError('unexpected distance site count')
    # Do not leave any raw squared-distance compare outside a wrapped call.
    check=source
    for start,end,_ in reversed(replacements): check=check[:start]+'FIXED_DISTANCE'+check[end:]
    uncommented=re.sub(r'//[^\n]*|/\*.*?\*/','',check,flags=re.S)
    if '_distance->compare' in uncommented: raise ValueError('unhandled build distance')
    for start,end,value in reversed(replacements): source=source[:start]+value+source[end:]
    helper='''#include <cstdlib>
#include <cmath>
#include <stdexcept>
// Both HybridIndex channels receive this same fixed scalar metric.
// This specializes candidate navigation, ordering, centroid entries and pruning.
static inline float hvs_fixed_fuse(float x, float y) {
  static const float a=[] {
    const char *s=std::getenv("HVS_FIXED_ALPHA");
    if(!s) throw std::runtime_error("HVS_FIXED_ALPHA required");
    char *end=nullptr; float v=std::strtof(s,&end);
    if(!end||*end||!std::isfinite(v)||v<0||v>1)
      throw std::runtime_error("HVS_FIXED_ALPHA invalid");
    return v;
  }();
  return a*x+(1-a)*y;
}
'''
    return helper+source
def main():
    O.mkdir(exist_ok=True); (O/'variant').mkdir(exist_ok=True); (O/'bin').mkdir(exist_ok=True)
    b=json.loads((E/'metric-consistent-20261111/analysis/build.json').read_text())
    lib=E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if digest(lib)!=b['library_sha256']: raise RuntimeError('corrected library identity')
    src=E/'metric-consistent-20261111/variant/hybrid_index.cpp'
    transformed=specialize(src.read_text(encoding='utf-8'))
    target=O/'variant/hybrid_index.cpp'; target.write_text(transformed,encoding='utf-8')
    # Keep inherited source formatting; only the generated helper is new code.
    obj=O/'variant/hybrid_index.cpp.o'
    flags=b['flags']
    commands=[['g++-11']+flags+['-c',str(target),'-o',str(obj)]]
    fixed=O/'bin/libpipeann-fixed.a'; shutil.copyfile(str(lib),str(fixed))
    commands.append(['ar','r',str(fixed),str(obj)])
    builder=E/'laion-six-cell-20261111/scripts/build_graph.cpp'
    commands.append(['g++-11']+flags+[str(builder),str(fixed)]+b['link']+['-o',str(O/'bin/build-fixed')])
    with (O/'specialized-compile.log').open('w') as f:
        for cmd in commands:
            subprocess.check_call(['numactl','--all','--physcpubind=80,81','--membind=3']+cmd,stdout=f,stderr=subprocess.STDOUT)
    proof=dict(status='COMPILED_NOT_YET_VALIDATED',distance_sites=14,corrected_library_sha256=digest(lib),source_sha256=digest(src),variant_sha256=digest(target),builder_sha256=digest(O/'bin/build-fixed'),commands=commands,method='duplicate fixed Euclidean blend into both internal channels; no final-pruning-only substitution')
    (O/'specialized-build.json').write_text(json.dumps(proof,indent=2)+'\n')
if __name__=='__main__': main()
