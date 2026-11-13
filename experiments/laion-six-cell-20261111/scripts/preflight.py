"""Required GT, memory, data-identity checks and 16/24-worker cost probes."""
import fcntl
import json
import math
from pathlib import Path
import re
import shutil
import struct
import sys
import time
import common as c


def read(path):
    return json.loads(Path(path).read_text())


def inventory():
    manifest = read(c.ROOT.parent/'concurrency-alpha/analysis/manifest.json')
    workloads = sorted([p for p in manifest['configurations'] if p['dataset']=='laion' and p['fixed_alpha'] in (.1,.5)],
                       key=lambda p:p['fixed_alpha'], reverse=True)
    if sorted(p['fixed_alpha'] for p in workloads) != [.1,.5]:
        raise RuntimeError('missing fixed-alpha workload')
    shapes = {}
    for name in ('base_img_emb.fvecs', 'base_text_emb.fvecs', 'query_img_emb.fvecs', 'query_text_emb.fvecs'):
        p = c.DATA/name
        with p.open('rb') as stream:
            n, d = struct.unpack('<II', stream.read(8))
        if p.stat().st_size != 8+n*d*4:
            raise RuntimeError('custom matrix size mismatch')
        shapes[name] = [n,d]
    if shapes['base_img_emb.fvecs'] != [10642155,512] or shapes['base_text_emb.fvecs'] != [10642155,512]:
        raise RuntimeError('base identity')
    for w in workloads:
        for path, expected in w['input_sha256'].items():
            if c.digest(path) != expected:
                raise RuntimeError('query/GT identity changed: '+path)
        for name, dims in [('gt',10),('alpha',1)]:
            with Path(w[name]).open('rb') as stream:
                if struct.unpack('<II',stream.read(8)) != (1000,dims):
                    raise RuntimeError('query count or GT width')
    memory = c.node_memory()
    c.save(c.ROOT/'analysis/inventory.json',dict(dataset='Laion',nodes=10642155,dimensions=[512,512],
        workloads=workloads,shapes=shapes,node3_memory=memory,
        free_bytes={str(p):shutil.disk_usage(str(p)).free for p in (c.ROOT,c.STORE)},
        coordinate=str(c.COORD),coordinate_bytes=c.COORD.stat().st_size,
        expected_dram_resident_upper_bytes=(65<<30),required_dram_free_bytes=(80<<30)))
    if memory['MemFree'] < 80*(1<<30):
        raise RuntimeError('NUMA3 lacks 80 GiB free headroom for resident query images')


def compile_tools():
    spec = read(c.PREV/'analysis/build.json')
    if c.digest(c.PREV/'bin/libpipeann-metric.a') != spec['library_sha256']:
        raise RuntimeError('corrected library changed')
    c.run(['g++-11']+spec['flags']+[c.ROOT/'scripts/build_graph.cpp',c.PREV/'bin/libpipeann-metric.a']+
          spec['link']+['-o',c.ROOT/'bin/build-graph'], 'compile-builder', c.CPUS[:2],timeout=1200)
    c.run(['g++-11','-std=c++17','-O2',c.ROOT/'scripts/full_layout.cpp','-o',c.ROOT/'bin/full-layout'],
          'compile-full-layout',c.CPUS[:2],timeout=1200)
    fixture = c.PREV/'scratch/fixture'
    dest = c.ROOT/'scratch/fixture'
    dest.mkdir(exist_ok=True)
    c.run([c.ROOT/'bin/full-layout','write',fixture/'float-graph.bin',fixture/'full.meta',dest/'topo.bin',dest/'write.json'],
          'fixture-materialize',c.CPUS[:2])
    if c.digest(dest/'topo.bin') != c.digest(fixture/'full.topo'):
        raise RuntimeError('Full materializer disagrees with OpenImages writer')
    c.run([c.ROOT/'bin/full-layout','check',fixture/'float-graph.bin',fixture/'full.meta',dest/'topo.bin',dest/'check.json'],
          'fixture-verify',c.CPUS[:2])
    broken = dest/'corrupt.bin'
    shutil.copyfile(str(dest/'topo.bin'),str(broken))
    with broken.open('r+b') as stream:
        stream.seek(4);stream.write(struct.pack('<I',2**32-1))
    c.run([c.ROOT/'bin/full-layout','check',fixture/'float-graph.bin',fixture/'full.meta',broken,dest/'bad.json'],
          'fixture-reject-corruption',c.CPUS[:2],expected=1)
    c.run([c.ROOT/'bin/build-graph','dual-prefix',fixture/'x.bin',fixture/'y.bin',512,8,16,2,dest],
          'fixture-prefix-build',c.CPUS[:2],dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',HVS_SOURCE_ONLY='1'))
    c.run([c.ROOT/'bin/full-layout','write',dest/'float-graph.bin',dest/'full.meta',dest/'new.topo',dest/'prefix.json'],
          'fixture-prefix-pack',c.CPUS[:2])
    c.save(c.ROOT/'analysis/tool-validation.json',dict(status='PASS',writer_reference_byte_identity=True,
        corruption_rejected=True,prefix_builder_checked=True,library_sha256=spec['library_sha256'],
        search_sha256=c.digest(c.ROOT/'bin/metric-search'),builder_sha256=c.digest(c.ROOT/'bin/build-graph'),
        layout_tool_sha256=c.digest(c.ROOT/'bin/full-layout')))


def probe(n, threads):
    label='probe-n{}-t{}'.format(n,threads)
    proof=c.ROOT/'analysis'/(label+'.json')
    if proof.exists(): return read(proof)
    folder=c.STORE/label
    folder.mkdir(exist_ok=True)
    started=time.time()
    c.run([c.ROOT/'bin/build-graph','dual-prefix',c.DATA/'base_img_emb.fvecs',c.DATA/'base_text_emb.fvecs',
           n,512,512,threads,folder],label,c.BUILD_CPUS[:threads],
          dict(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),HVS_SOURCE_ONLY='1'),timeout=10800)
    elapsed=time.time()-started
    log=(c.ROOT/'logs'/(label+'.log')).read_text()
    link=float(re.search(r'Link time: ([0-9.eE+-]+)s',log).group(1))
    timing=(c.ROOT/'logs'/(label+'.time')).read_text()
    rss=int(re.search(r'Maximum resident set size \(kbytes\): (\d+)',timing).group(1))*1024
    audit=read(folder/'graph-audit.json')
    if audit['status']!='SOURCE_COMPLETE' or audit['nodes']!=n:
        raise RuntimeError('incomplete probe source')
    result=dict(nodes=n,threads=threads,cpus=c.BUILD_CPUS[:threads],link_seconds=link,
        wall_seconds=elapsed,peak_rss_bytes=rss,source_audit=audit,
        float_graph_bytes=(folder/'float-graph.bin').stat().st_size,
        float_graph_sha256=c.digest(folder/'float-graph.bin'),subset='original ID prefix, cost estimate only')
    c.save(proof,result)
    # This is a reproducible cost probe, not a retained scientific index.
    (folder/'float-graph.bin').unlink()
    return result


def run_preflight():
    c.init()
    if not (c.ROOT/'analysis/inventory.json').exists(): inventory()
    if not (c.ROOT/'analysis/tool-validation.json').exists(): compile_tools()
    if not (c.ROOT/'analysis/gt-objective-audit.json').exists():
        c.run([sys.executable,c.ROOT/'scripts/gt_audit.py'],'gt-objective-audit',c.BUILD_CPUS[:16],
              dict(OPENBLAS_NUM_THREADS='16',MKL_NUM_THREADS='16',OMP_NUM_THREADS='16',OMP_THREAD_LIMIT='16'))
    if read(c.ROOT/'analysis/gt-objective-audit.json')['status']!='PASS':
        raise RuntimeError('GT semantics gate failed')
    if not (c.ROOT/'analysis/coordinate-identity.json').exists():
        c.run([sys.executable,c.ROOT/'scripts/coordinate_check.py'],'coordinate-identity',c.CPUS[:2])
    p16,p24=probe(200000,16),probe(200000,24)
    chosen=24 if p24['link_seconds'] < .95*p16['link_seconds'] else 16
    small=p24 if chosen==24 else p16
    large=probe(1000000,chosen)
    exponent=max(1.,math.log(large['link_seconds']/small['link_seconds'])/math.log(5))
    scale=10642155/1000000.
    linear=large['link_seconds']*scale
    projected=large['link_seconds']*scale**exponent
    estimate=dict(status='PASS',construction_threads=chosen,construction_cpus=c.BUILD_CPUS[:chosen],
        lower_link_seconds=linear,empirical_power_link_seconds=projected,
        planning_upper_seconds=max(projected,linear)*1.75+1800,
        growth_exponent=exponent,peak_rss_projection_bytes=int(large['peak_rss_bytes']*scale*1.3),
        input_loading_serialization_and_validation_additional=True,
        caveat='prefix probes estimate cost; they are not full-Laion search evidence',probes=[small,large])
    c.save(c.ROOT/'analysis/build-cost-estimate.json',estimate)
    c.status('preflight_complete',build_threads=chosen,estimated_link_hours=projected/3600,
             planning_upper_hours=estimate['planning_upper_seconds']/3600)


if __name__=='__main__':
    c.init()
    lock=open(str(c.ROOT.parent/'budget-reorder-20261103/queue.lock'),'a+')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try: run_preflight()
    except Exception as error:
        c.status('failed',error=str(error));raise
