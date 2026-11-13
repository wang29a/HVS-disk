"""Compile isolated I/O-only change and validate on 512-node functional fixture."""
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import numpy as np

ROOT=Path(__file__).resolve().parent.parent
EXPERIMENTS=ROOT.parent
OLD=EXPERIMENTS/'metric-consistent-20261111'
sys.path.insert(0,str(EXPERIMENTS/'laion-six-cell-20261111/scripts'))
import common as c
c.ROOT,c.ACTIVE=ROOT,ROOT
CPUS=[76,77]
COORD=Path('/mnt/nvme3/wz/Index/Openimage/disk_index_data')
PQ=Path('/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full')


def read(path):return json.loads(Path(path).read_text())


def matrix(path,dtype=np.float32):
    with Path(path).open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    return np.memmap(str(path),dtype=dtype,mode='r',offset=8,shape=(n,d))


def save_matrix(path,values):
    with Path(path).open('wb') as f:
        f.write(struct.pack('<II',*values.shape));values.tofile(f)


def compile_binary():
    c.init()
    spec=read(OLD/'analysis/build.json')
    if c.digest(OLD/'bin/libpipeann-metric.a')!=spec['library_sha256']:raise RuntimeError('corrected library changed')
    c.run(['g++-11']+spec['flags']+[ROOT/'scripts/layout_engine.cpp',OLD/'bin/libpipeann-metric.a']+
          spec['link']+['-o',ROOT/'bin/metric-search-new'],'compile-topology-resident',CPUS,
          dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'),timeout=1200)
    # c.init makes a legacy symlink. Publish only this directory's new binary.
    destination=ROOT/'bin/metric-search'
    if destination.is_symlink():destination.unlink()
    os.replace(str(ROOT/'bin/metric-search-new'),str(destination))


def regression():
    fixture=ROOT/'scratch/fixture';fixture.mkdir(exist_ok=True)
    for name in ('full.meta','full.topo','compact.meta','compact.topo'):
        shutil.copyfile(str(OLD/'scratch/fixture'/name),str(fixture/name))
    for name in ('full.meta','compact.meta'):
        m=bytearray((fixture/name).read_bytes());struct.pack_into('<II',m,4,768,768);(fixture/name).write_bytes(m)
    with COORD.open('rb') as f:(fixture/'coord').write_bytes(f.read(512*8192))
    prefix=fixture/'pq';prefix.mkdir(exist_ok=True)
    for name in ('emb_pq_pivots.bin','loc_pq_pivots.bin'):
        p=prefix/name
        if not p.exists():p.symlink_to(PQ/name)
    for name in ('emb_pq_compressed.bin','loc_pq_compressed.bin'):
        with (PQ/name).open('rb') as f:
            f.read(8);data=f.read(512*64)
        (prefix/name).write_bytes(struct.pack('<II',512,64)+data)
    split=read(OLD/'analysis/tuning-split-v2.json')['cohorts']['report']['files']
    x=np.array(matrix(split['query1']['path'])[:8]);y=np.array(matrix(split['query2']['path'])[:8])
    weights=np.array(matrix(split['alpha']['path'])[:8])
    for name,array in [('qx',x),('qy',y),('alpha',weights)]:save_matrix(fixture/name,array)
    coordinates=np.memmap(str(fixture/'coord'),dtype=np.float32,mode='r',shape=(512,2048))
    d1=np.sum((x.astype(np.float64)[:,None,:]-coordinates[None,:,:768])**2,axis=2)
    d2=np.sum((y.astype(np.float64)[:,None,:]-coordinates[None,:,768:1536])**2,axis=2)
    for name,scores in [('gt',weights*np.sqrt(d1)+(1-weights)*np.sqrt(d2)),('concat-gt',np.sqrt(d1+d2))]:
        save_matrix(fixture/name,np.asarray(np.argsort(scores,axis=1)[:,:10],dtype=np.uint32))
    checks=[]
    for arm in ('single','compact','full'):
        stem='full' if arm=='full' else 'compact'
        meta,topo=fixture/(stem+'.meta'),fixture/(stem+'.topo')
        for kind in ('TXY','T_XY'):
            folder=fixture/(arm+'-'+kind);folder.mkdir(exist_ok=True)
            c.run([ROOT/'bin/metric-search','build',meta,topo,fixture/'coord',folder,kind,folder/'identity.json'],
                  'fixture-build-'+arm+'-'+kind,CPUS)
            records={}
            for mode,binary in [('legacy',OLD/'bin/metric-search'),('off',ROOT/'bin/metric-search'),('on',ROOT/'bin/metric-search')]:
                label='fixture-query-{}-{}-{}'.format(arm,kind,mode)
                out=fixture/(label+'.json');trace=fixture/label
                env=dict(LAYOUT_ONE_PASS='1',LAYOUT_GT_DIAGNOSTIC='1',LAYOUT_TRACE=str(trace),LAYOUT_DUMP=str(trace),
                    METRIC_RERANK='40',OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2')
                if mode=='on':env['LAYOUT_TOPO_DRAM']='1'
                if arm=='single':env['METRIC_CONCAT']='1'
                c.run([binary,'query',meta,topo,fixture/'coord',folder,kind,out,str(prefix)+'/',fixture/'qx',fixture/'qy',
                       fixture/'alpha',fixture/('concat-gt' if arm=='single' else 'gt'),2,10,60],label,CPUS,env,timeout=120)
                records[mode]=(read(out)[0],trace)
            for mode in ('off','on'):
                for suffix in ('.L60','.L60.ids','.L60.dists'):
                    if Path(str(records[mode][1])+suffix).read_bytes()!=Path(str(records['legacy'][1])+suffix).read_bytes():
                        raise RuntimeError('topology residency changed trajectory or returned IDs/distances')
            point=records['on'][0]
            if point['topology_submitted_pages']!=0 or point['topology_memory_node_reads']!=point['expanded_nodes']:
                raise RuntimeError('topology is not purely DRAM served')
            if point['coordinate_medium']!='SSD' or point['coordinate_submitted_pages']!=point['reranked_nodes'] or point['reranked_nodes']!=40:
                raise RuntimeError('coordinates accidentally resident or not independently read')
            if kind=='TXY' and records['legacy'][0]['coordinate_submitted_pages']!=0:
                raise RuntimeError('fixture does not exercise coupled coordinate reuse')
            checks.append(dict(arm=arm,layout=kind,identical_to_legacy=True,topology_backend_pages=0,
                coordinate_backend_pages=point['coordinate_submitted_pages'],topology_resident_bytes=point['topology_resident_bytes']))
    c.save(ROOT/'analysis/preflight.json',dict(status='PASS',checks=checks,queries=8,L=60,r=40,
        fixture='functional test only: 512-node graph, real vector/PQ prefix, no performance or recall claim',
        binary_sha256=c.digest(ROOT/'bin/metric-search'),library_sha256=c.digest(OLD/'bin/libpipeann-metric.a'),
        source_sha256={name:c.digest(ROOT/'scripts'/name) for name in ('layout_store.h','layout_engine.cpp')}))


if __name__=='__main__':
    compile_binary();regression()
