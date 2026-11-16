"""Build a pinned independent author checkout without replacing its algorithms."""
import shutil
import common as c
SHA='728c4128e74039ce212c8c22a42a5185abd68c23'
def main():
    root=c.RUN/'author-original';adapt=c.RUN/'author-adapted'
    manifest=c.load(c.ROOT/'AUTHOR-UPSTREAM.json')
    if manifest['commit']!=SHA:raise RuntimeError('author commit pin mismatch')
    for name,h in manifest['files'].items():
        if c.sha(root/name)!=h:raise RuntimeError('author source differs from official pinned checkout: '+name)
    if adapt.exists():
        if not (adapt/'adaptation.json').exists():raise RuntimeError('incomplete old author adaptation')
    else:
        adapt.mkdir();shutil.copytree(root/'include',adapt/'include');shutil.copytree(root/'src',adapt/'src')
        # The pinned author Index already uses E_Distance for both spaces.
        index=(adapt/'include/index.h').read_bytes().decode('utf-8')
        if 's_dist_ = new E_Distance(max_spatial_dist);' not in index:raise RuntimeError('author second metric is not full-dimensional sqrt L2; explicit adaptation required')
        c.save(adapt/'adaptation.json',dict(commit=SHA,changes='none in author algorithms/distance: both channels already E_Distance; wrapper sets both normalisers1',source={str(p.relative_to(adapt)):c.sha(p) for p in adapt.rglob('*') if p.is_file()}))
    cfg=c.load(c.E/'metric-consistent-20261111/analysis/build.json')
    # The author relies on ordinary C++ includes through its original headers.
    # Use the same formal optimisation/FP flags, plus its include directory.
    flags=[v for v in cfg['flags'] if not v.startswith('-I')]+['-I'+str(adapt/'include'),'-fno-fast-math','-ffp-contract=off','-ffunction-sections','-fdata-sections']
    objects=[]
    for src in ('component_init.cpp','component_prune.cpp'):
        obj=c.RUN/'build'/('author-'+src+'.o');c.run(['g++-11']+flags+['-c',adapt/'src'/src,'-o',obj],'compile-author-'+src,cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'));objects.append(obj)
    binary=c.RUN/'bin/build-author'
    c.run(['g++-11']+flags+[c.ROOT/'scripts/author_driver.cpp']+objects+['-Wl,--gc-sections','-o',binary],'link-author-driver',cpus='80,81',timeout=600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    c.save(c.RUN/'validation/author-build.json',dict(status='COMPILED_NOT_GATE_PASSED',commit=SHA,adaptation=c.load(adapt/'adaptation.json'),driver=c.sha(c.ROOT/'scripts/author_driver.cpp'),binary=c.sha(binary),flags=flags,normalisers=[1,1],metric='both full-vector sqrt L2, already in official Index constructor'))
if __name__=='__main__':main()
