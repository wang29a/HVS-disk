"""Run the native exact-distance OpenImage DEG active-range ablation."""
import fcntl
import io
import os
from pathlib import Path
import re
import signal
import subprocess
import tarfile
import time
import traceback

import factorial as f

ROOT = f.ROOT
WORK = ROOT / 'native-deg'
SOURCE = Path('/mnt/nvme3/wz/DEG')
COMMIT = '728c4128e74039ce212c8c22a42a5185abd68c23'
DATA = Path('/mnt/nvme3/wz/dataset/OpenImage')
ARCHIVED_GRAPH = Path('/mnt/nvme3/wz/Index/Openimage/deg.index')
ARCHIVED_GRAPH_SHA256 = 'f42624628afadcb810d872b52cb2345b96c1bf986b45e1db2e26ece896879d6e'
BUILD_THREADS = 8
base, monitor = f.base, f.monitor
base.ROOT = monitor.ROOT = WORK
N, Q, D = 507444, 1000, 768
ALPHAS = ('0.2', '0.1', '0.3', '0.5', '0.7', '0.9')
LS = list(range(10, 201, 10)) + [240, 320, 400, 600, 800]


def replace(path, old, new):
    source = path.read_text(encoding="utf-8")
    if source.count(old) != 1:
        raise RuntimeError('patch no longer matches ' + str(path))
    path.write_text(source.replace(old, new), encoding="utf-8")


def prepare_data():
    np = f.np
    target = WORK/'dataset/OpenImage'
    target.mkdir(parents=True, exist_ok=True)
    for kind, count in (('base', N), ('query', Q)):
        for space in ('img','text'):
            name = '{}_{}_emb.fvecs'.format(kind,space)
            path = DATA/name
            vectors = np.memmap(str(path), dtype='<u4', mode='r', shape=(count,D+1))
            if path.stat().st_size != count*(D+1)*4 or not np.all(vectors[:,0]==D):
                raise RuntimeError('native vector shape: '+name)
            (target/name).symlink_to(path)
    for index in range(7):
        for suffix, width, dtype in (('query_alpha.fvecs',1,'<f4'),('top10_results.ivecs',10,'<u4')):
            name = 'range_{}_{}'.format(index,suffix)
            path = DATA/name
            raw = np.fromfile(str(path), dtype='<u4').reshape(Q,width+1)
            values = raw[:,1:].view(dtype)
            if np.any(raw[:,0]!=width) or (width==10 and np.any(values>=N)) or (
                    width==1 and (not np.all(np.isfinite(values)) or np.any(values<0) or np.any(values>1))):
                raise RuntimeError('invalid native alpha or GT: '+name)
            (target/name).symlink_to(path)
    for suffix, width in (('query_alpha.fvecs',1),('top10_results.ivecs',10)):
        path = DATA/'base_gt_round_00'/suffix
        flat = np.fromfile(str(path), dtype='<u4')
        if flat.size != Q*width+2 or list(flat[:2])!=[Q,width] or (width==10 and np.any(flat[2:]>=N)):
            raise RuntimeError('invalid overall query or GT: '+suffix)
        weights = flat[2:].view('<f4')
        if width==1 and (not np.all(np.isfinite(weights)) or np.any(weights<0) or np.any(weights>1)):
            raise RuntimeError('invalid overall weights')
        rows = np.empty((Q,width+1), dtype='<u4')
        rows[:,0] = width
        rows[:,1:] = flat[2:].reshape(Q,width)
        rows.tofile(str(target/('range_7_'+suffix)))
    return target


def prepare_code():
    source = WORK/'source'
    source.mkdir()
    archive = subprocess.check_output(['git','archive',COMMIT,'CMakeLists.txt','src','test','include'],
                                      cwd=str(SOURCE))
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(str(source))
    replace(source/'test/main.cpp', '/mnt/hdd/yinziqi/yinziqi/graphann-tkq/dataset/',
            str(WORK/'dataset')+'/')
    replace(source/'test/main.cpp', '/mnt/hdd/yinziqi/yinziqi/graphann-tkq/saved_index/',
            str(WORK/'index')+'/')
    replace(source/'test/main.cpp', 'parameters.set<unsigned>("n_threads", 8);',
            'parameters.set<unsigned>("n_threads", {});'.format(BUILD_THREADS))
    replace(source/'include/set_para.h', 'std::fabs(alpha - 0) < epsilon',
            'std::fabs(alpha - 0.2f) < epsilon) { range = 7; }\n    else if (std::fabs(alpha - 0) < epsilon')
    route = source/'src/component_route.cpp'
    signature = 'void ComponentSearchRouteDEG::SearchAtLayer('
    body = route.read_text(encoding="utf-8")
    if body.count(signature)!=1:
        raise RuntimeError('could not locate native traversal')
    at = body.index('{',body.index(signature))+1
    route.write_text('#include <cstdlib>\n'+body[:at]+
                     '\n        static const bool bypass_alpha = std::getenv("DEG_BYPASS_ALPHA") != nullptr;'+body[at:], encoding="utf-8")
    replace(route, 'bool search_flag = false;\n                for (int i = 0; i < use_range.size(); i++)',
            'bool search_flag = bypass_alpha;\n                for (int i = 0; !bypass_alpha && i < use_range.size(); i++)')
    builder = source/'src/builder.cpp'
    replace(builder, 'for (unsigned t = 0; t < 20; t++)\n            {\n\n                L = L + K;',
            'for (unsigned depth : {'+', '.join(map(str,LS))+'})\n            {\n                L = depth;')
    return source


def curve(path):
    rows = re.findall(r'SEARCH_L : (\d+)\s+search time: ([0-9.eE+-]+)\s+DistCount: (\d+)\s+HopCount: (\d+)\s+10 NN accuracy: ([0-9.eE+-]+)',path.read_text(encoding="utf-8"))
    if [int(row[0]) for row in rows]!=LS:
        raise RuntimeError('incomplete native search: '+str(path))
    return [dict(L=int(depth),recall=100*float(acc),latency_us=1e6*float(sec),qps=1/float(sec),
                 distance_count=int(dist),hop_count=int(hops)) for depth,sec,dist,hops,acc in rows]


def run():
    base.status('preparing_original_deg')
    target = prepare_data()
    source = prepare_code()
    monitor.CPUS = f.common.quiet_cores()
    base.save(WORK/'analysis/protocol.json',dict(source_commit=COMMIT, n=N, q=Q, dimensions=[D,D],
              degree=40, fallback_ef_construction=200, max_distances=[1,1], L=LS, repeats=3,
              fallback_build_threads=BUILD_THREADS, search_threads=1, cpus=monitor.CPUS, numa=3,
              query_alpha_and_gt={p.name:base.digest(p) for p in target.glob('range_*.?vecs')}))
    base.status('building_native_binary')
    build = WORK/'build'
    env = dict(PATH='/opt/gcc-11.4/bin:'+os.environ['PATH'])
    base.run(['cmake','-S',source,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
              '-DCMAKE_CXX_COMPILER=/usr/bin/g++-11','-DCMAKE_C_COMPILER=/usr/bin/gcc-11'],
             'configure-native',env,monitor.CPUS)
    base.run(['cmake','--build',build,'--target','main','-j','4'],'compile-native',env,monitor.CPUS)
    (WORK/'index').mkdir()
    binary = build/'test/main'
    prefix = [binary,'deg','openimage']
    graph = WORK/'index/deg_openimage.index'
    smoke_path = WORK/'reuse-smoke.json'
    smoke = base.read(smoke_path) if smoke_path.exists() else {}
    reused = (smoke.get('status')=='PASS' and smoke.get('candidate')==str(ARCHIVED_GRAPH)
              and ARCHIVED_GRAPH.is_file() and base.digest(ARCHIVED_GRAPH)==ARCHIVED_GRAPH_SHA256)
    if reused:
        base.status('reusing_native_graph',path=str(ARCHIVED_GRAPH))
        graph.symlink_to(ARCHIVED_GRAPH)
    else:
        build_cpus = f.common.quiet_cores(BUILD_THREADS)
        base.status('building_native_graph',cpus=build_cpus)
        build_env = dict(env,OMP_NUM_THREADS=str(BUILD_THREADS),OMP_THREAD_LIMIT=str(BUILD_THREADS))
        base.run(prefix+['0.5',1,1,'build'],'build-native-graph',build_env,build_cpus)
    if graph.stat().st_size <= N*8:
        raise RuntimeError('native graph is incomplete')
    sha = base.digest(graph)
    base.save(WORK/'analysis/graph.json',dict(path=str(graph),source=str(graph.resolve()),sha256=sha,
              reused=reused,build_threads=0 if reused else BUILD_THREADS))
    records = []
    for rep in (1,2,3):
        for alpha in ALPHAS:
            for variant in (('DEG','DEGStatic') if rep%2 else ('DEGStatic','DEG')):
                label = '{}-{}-r{}'.format(variant, 'overall' if alpha=='0.2' else alpha,rep)
                monitor.gate(label)
                base.status('searching_native',task=label,cpus=monitor.CPUS[:1])
                search_env = dict(env,OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1')
                if variant=='DEGStatic':
                    search_env['DEG_BYPASS_ALPHA'] = '1'
                base.run(prefix+[alpha,1,1,'search'],label,search_env,monitor.CPUS[:1])
                points = curve(WORK/'logs'/(label+'.log'))
                records.append(dict(label=label,repeat=rep,variant=variant,alpha_selection=alpha,
                                    points=points,recall_targets=[t for t in f.TARGETS if base.bracket(points,t)]))
                base.save(WORK/'analysis/curves.json',records)
    if sha!=base.digest(graph):
        raise RuntimeError('native graph changed during measurement')
    base.status('complete',curves=len(records))


def main():
    WORK.mkdir(exist_ok=True)
    for name in ('logs','analysis'):
        (WORK/name).mkdir(exist_ok=True)
    base.status('waiting_for_laion',reuse_candidate=str(ARCHIVED_GRAPH),
                fallback_build_threads=BUILD_THREADS,search_threads=1)
    laion = ROOT/'laion-fixed/status.json'
    while True:
        predecessor = base.read(laion)['status']
        if predecessor in ('complete','complete_with_partial_coverage'):
            break
        if predecessor in ('failed','interrupted','blocked_by_openimages'):
            base.status('blocked_by_laion',predecessor=predecessor)
            return
        time.sleep(20)
    with (base.PREV/'queue.lock').open('r') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        run()


if __name__ == '__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):
        signal.signal(sig,base.stop)
    try:
        main()
    except Exception as exc:
        base.status('interrupted' if isinstance(exc,InterruptedError) else 'failed',error=str(exc))
        traceback.print_exc()
        raise SystemExit(1)
