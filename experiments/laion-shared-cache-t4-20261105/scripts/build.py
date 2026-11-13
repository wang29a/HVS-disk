import hashlib
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent.parent
BUILD = Path('/mnt/nvme3/wz/hvs-disk-build-20261027')
ENV = dict(os.environ, LC_ALL='C', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
           OMP_NUM_THREADS='4', OMP_THREAD_LIMIT='4')
ENV['LD_LIBRARY_PATH'] = '/opt/gcc-11.4/lib64:/mnt/nvme2/wz/disk/local/openblas/lib:' + ENV.get('LD_LIBRARY_PATH', '')
def sha(p):
    h = hashlib.sha256()
    with open(str(p), 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''): h.update(b)
    return h.hexdigest()
def main():
    for d in ['bin', 'analysis', 'logs', 'results']: (ROOT/d).mkdir(exist_ok=True)
    sources = ['shared_topo_reader.h', 'cache_search.cpp', 'test_cache.cpp']
    subprocess.run(['clang-format', '-i', '-style=LLVM'] + [str(ROOT/'scripts'/s) for s in sources], check=True)
    spec = json.loads((BUILD/'experiments/concurrency-alpha/analysis/build.json').read_text())['separated']
    if sha(spec['library']) != spec['library_sha256']: raise RuntimeError('native library changed')
    commands = []
    for source, name, extra in [('cache_search.cpp', 'cache-search', []),
                                 ('test_cache.cpp', 'test-cache', []),
                                 ('test_cache.cpp', 'test-cache-asan', ['-fsanitize=address', '-fno-omit-frame-pointer'])]:
        cmd = ['g++-11'] + spec['flags'] + extra + [str(ROOT/'scripts'/source), spec['library'],
            '-L/mnt/nvme2/wz/disk/local/openblas/lib', '-laio', '-ljemalloc', '-lopenblas', '-o', str(ROOT/'bin'/name)]
        commands.append(cmd)
        with (ROOT/'logs'/('build-'+name+'.log')).open('w') as f:
            subprocess.run(cmd, env=ENV, stdout=f, stderr=subprocess.STDOUT, check=True)
    for name in ['test-cache', 'test-cache-asan']:
        fixture = '/mnt/nvme2/wz/laion-cache-fixture-20261105-{}-{}'.format(os.getpid(), name)
        e = dict(ENV, ASAN_OPTIONS='detect_leaks=0')
        with (ROOT/'logs'/(name+'.log')).open('w') as f:
            subprocess.run([str(ROOT/'bin'/name), fixture], env=e, stdout=f, stderr=subprocess.STDOUT, check=True)
        os.unlink(fixture)
    result = dict(source_commit='d2a39e2', native_library=spec['library'], native_library_sha256=spec['library_sha256'],
        commands=commands, sources={s:sha(ROOT/'scripts'/s) for s in sources},
        binaries={n:sha(ROOT/'bin'/n) for n in ['cache-search','test-cache','test-cache-asan']},
        synthetic_tests='PASS: release and ASan; 8 KiB, eviction, four threads, cache on/off',
        tests_do_not_search_datasets=True)
    (ROOT/'analysis/build.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))
if __name__ == '__main__': main()
