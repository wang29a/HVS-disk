#!/usr/bin/env python3
"""Build and validate U-shape tools against the verified corrected library."""
import hashlib,json,os,pathlib,subprocess,sys
R=pathlib.Path(__file__).resolve().parent.parent; E=R.parent; O=R/'stage2'
def main():
    (O/'bin').mkdir(parents=True,exist_ok=True)
    config=json.loads((E/'metric-consistent-20261111/analysis/build.json').read_text())
    library=E/'metric-consistent-20261111/bin/libpipeann-metric.a'
    if hashlib.sha256(library.read_bytes()).hexdigest()!=config['library_sha256']: raise RuntimeError('corrected library mismatch')
    env=dict(os.environ,LC_ALL='C',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH']='/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:'+env.get('LD_LIBRARY_PATH','')
    cmd=['numactl','--all','--physcpubind=80,81','--membind=3','g++-11']+config['flags']+[str(R/'scripts/pointer_search.cpp'),str(library)]+config['link']+['-o',str(O/'bin/pointer-search')]
    with (O/'pointer-compile.log').open('w') as log: subprocess.check_call(cmd,env=env,stdout=log,stderr=subprocess.STDOUT)
    (O/'pointer-build.json').write_text(json.dumps(dict(status='PASS',command=cmd,source_sha256=hashlib.sha256((R/'scripts/pointer_search.cpp').read_bytes()).hexdigest()),indent=2)+'\n')
    for script in ('prepare_specialized.py','check_pointer.py','check_specialized.py'):
        subprocess.check_call([sys.executable,str(R/'scripts'/script)],env=env)
if __name__=='__main__': main()
