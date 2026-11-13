"""Matched-recall comparison of X-only Vamana and the dual graph at alpha=1."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import statistics
import struct
import sys
import traceback

ROOT = Path(__file__).resolve().parent.parent
EXP = ROOT.parent
SINGLE = EXP/'openimage-single-space-20261110'
sys.path.insert(0,str(EXP/'second-space-cost-20261110'/'scripts'))
import endpoint as tools
base, monitor = tools.base, tools.monitor
base.ROOT = monitor.ROOT = tools.ROOT = ROOT
base.ENGINE = ROOT/'bin/compare-search'
os.environ.update(OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OMP_NUM_THREADS='4')
import numpy as np

TARGETS=(90,95,98)
SCREEN=[10,20,30,40,60,80,120,176,240,400]
N=456699
DIM=768
PAGE=8192
CACHE=256<<20
SLOTS=40
FIELDS=('qps','latency_us','pq_lookup_us','exact_us','pq_space_evals','exact_space_evals',
        'topology_submitted_pages','coordinate_submitted_pages','topology_submit_wait_us',
        'coordinate_submit_wait_us','inspected_edges','alpha_rejected_edges','expanded_nodes','reranked_nodes')


def verified_single(d):
    spec=base.read(SINGLE/'analysis/artifacts.json')['files']
    if base.read(SINGLE/'analysis/index-validation.json')['status']!='PASS':
        raise RuntimeError('source single-space graph was not validated')
    src=SINGLE/'index'
    index=src/'openimage_x_disk.index'
    prefix=src/'openimage_x'
    gt=src/'openimage_x_l2_top10.bin'
    for p in (index,gt,Path(str(prefix)+'_pq_compressed.bin'),Path(str(prefix)+'_pq_pivots.bin')):
        if base.digest(p)!=spec[p.name]['sha256']:
            raise RuntimeError('single input changed: '+p.name)
    for suffix in ('_pq_compressed.bin','_pq_pivots.bin'):
        if base.digest(Path(d['prefix'])/('emb'+suffix))!=spec['openimage_x'+suffix]['sha256']:
            raise RuntimeError('X-side PQ differs')
    truth=gt.read_bytes()
    if struct.unpack_from('<II',truth)!=(1000,10) or len(truth)!=120008:
        raise RuntimeError('unexpected L2 ground truth format')
    truth_ids=ROOT/'analysis/x-top10-ids.bin'
    truth_ids.write_bytes(truth[:40008])
    return index,prefix,truth_ids


def convert(index, d):
    base.status('converting_vamana_graph')
    raw=np.memmap(str(index),mode='r',dtype=np.uint8)
    nodes,dim,medoid,record,npp,frozen,_,size=struct.unpack_from('<8Q',raw,8)
    if (nodes,dim,record,npp,frozen,size)!=(N,DIM,3236,2,0,len(raw)) or medoid>=N:
        raise RuntimeError('unsupported source Vamana format')
    folder=ROOT/'scratch/canonical-x'
    folder.mkdir()
    coords_per_page=PAGE//(DIM*4)
    topo_per_page=PAGE//((SLOTS+1)*4)
    coord_pages=(N+coords_per_page-1)//coords_per_page
    topo_pages=(N+topo_per_page-1)//topo_per_page
    base.require_space((coord_pages+topo_pages)*PAGE)
    meta=bytearray(PAGE)
    struct.pack_into('<IIIIIQII',meta,0,N,DIM,0,SLOTS,0,topo_per_page,1,medoid)
    (folder/'meta.bin').write_bytes(meta)
    coord=np.memmap(str(folder/'coord.bin'),mode='w+',dtype=np.uint8,shape=(coord_pages*PAGE,))
    topo=np.memmap(str(folder/'topo.bin'),mode='w+',dtype=np.uint8,shape=(topo_pages*PAGE,))
    src=np.ndarray((coord_pages,npp,record//4),dtype='<u4',buffer=raw,offset=PAGE,strides=(PAGE,record,4))
    xy=np.ndarray((coord_pages,coords_per_page,DIM),dtype='<u4',buffer=coord,strides=(PAGE,DIM*4,4))
    graph=np.ndarray((topo_pages,topo_per_page,SLOTS+1),dtype='<u4',buffer=topo,
                     strides=(PAGE,(SLOTS+1)*4,4))
    for start in range(0,N,8192):
        ids=np.arange(start,min(N,start+8192))
        rows=src[ids//npp,ids%npp,:]
        xy[ids//coords_per_page,ids%coords_per_page]=rows[:,:DIM]
        graph[ids//topo_per_page,ids%topo_per_page]=rows[:,DIM:]
    coord.flush()
    topo.flush()
    for start in range(0,N,8192):
        ids=np.arange(start,min(N,start+8192))
        rows=src[ids//npp,ids%npp,:]
        if (not np.array_equal(xy[ids//coords_per_page,ids%coords_per_page],rows[:,:DIM]) or
                not np.array_equal(graph[ids//topo_per_page,ids%topo_per_page],rows[:,DIM:])):
            raise RuntimeError('canonical conversion changed a node')
    base.save(ROOT/'analysis/conversion.json',dict(status='PASS',nodes=N,entry=medoid,dimension=DIM,
              full_coordinates_and_neighbor_lists_preserved=True,alpha_slots=0))
    return dict(d,name='openimage-single',budget='single',meta=str(folder/'meta.bin'),
                topo=str(folder/'topo.bin'),coord=str(folder/'coord.bin'))


def query(spec,levels,label,one=False,dump=False):
    c,layout,folder,prefix,only_x=spec
    env=dict(LAYOUT_ORDER=c['order'],LAYOUT_SHARED_CACHE_BYTES=str(CACHE),
             LAYOUT_ENDPOINT_X_ONLY='1' if only_x else '0')
    if one: env['LAYOUT_ONE_PASS']='1'
    if dump: env['LAYOUT_DUMP']=str(ROOT/'analysis'/label)
    out=ROOT/'results'/(label+'.json')
    cmd=[base.ENGINE,'compare_x',c['meta'],c['topo'],c['coord'],folder,layout,out,
         prefix,c['query1'],c['query2'],'unused-alpha',TRUTH,4,base.SECONDS]+levels
    base.status('screening' if one else 'timing',task=label,levels=levels,cpus=monitor.CPUS)
    if one:base.run(cmd,label,env)
    else:monitor.monitored_run(cmd,label,env,monitor.CPUS)
    pts=base.read(out)
    if [p['L'] for p in pts]!=levels:
        raise RuntimeError('incomplete search curve')
    for p in pts:
        base.check_point(p,True)
        if (p['threads']!=4 or p['queries']!=1000 or not 0<=p['recall']<=100 or
                p['exact_space_evals'] != p['reranked_nodes']*(1 if only_x else 2) or
                (not one and p['seconds']<base.SECONDS)):
            raise RuntimeError('workload or per-space count mismatch')
    return pts


def select(spec,label):
    rows=query(spec,SCREEN,label+'-screen',one=True)
    pairs={t:base.bracket(rows,t) for t in TARGETS}
    if not pairs[90] or not pairs[95]:
        rows.extend(query(spec,[600,800,1200,1600],label+'-high',one=True))
        pairs={t:base.bracket(rows,t) for t in TARGETS}
    levels={p['L'] for pair in pairs.values() if pair for p in pair}
    for target,pair in pairs.items():
        if pair and pair[0]['L']!=pair[1]['L']:
            a,b=pair
            levels.add(round(a['L']+(b['L']-a['L'])*(target-a['recall'])/(b['recall']-a['recall'])))
    if not levels:
        raise RuntimeError('no covered recall target')
    base.save(ROOT/'analysis'/(label+'-screen.json'),
              dict(selected=sorted(levels),covered=[t for t,p in pairs.items() if p],screen=rows))
    return sorted(levels)


def validate_answers(points,label,x,q,truth):
    proof={}
    for p in points:
        L=p['L']
        prefix=ROOT/'analysis'/('{}.L{}'.format(label,L))
        ids_path=Path(str(prefix)+'.ids')
        dist_path=Path(str(prefix)+'.dists')
        ids=np.fromfile(str(ids_path),dtype='<u4').reshape(1000,10)
        ds=np.fromfile(str(dist_path),dtype='<f4').reshape(1000,10)
        if ids.max()>=N or np.any(np.diff(np.sort(ids,axis=1),axis=1)==0) or np.any(np.diff(ds,axis=1)<0):
            raise RuntimeError('invalid returned ids or distance order')
        v=x[ids].astype(np.float64)-q[:,None,:]
        exact=np.sqrt(np.einsum('qkd,qkd->qk',v,v))
        if not np.allclose(exact,ds,rtol=2e-5,atol=2e-5):
            raise RuntimeError('returned distance did not match X')
        recall=(ids[:,:,None]==truth[:,None,:]).any(axis=2).mean()*100
        if abs(recall-p['recall'])>1e-7:
            raise RuntimeError('reported recall disagrees with independent computation')
        proof[L]=dict(ids=base.digest(ids_path),dists=base.digest(dist_path),recall=p['recall'])
    return proof


def report(records):
    base.save(ROOT/'analysis/measurements.json',records)
    lines=['# OpenImages, independent X graph versus dual graph at alpha=1','',
           'Three-repeat means at matched Recall@10. p95 uses the nearest measured L.',
           'All methods share the query set, X PQ, physical ID order, search harness and cache budget.','',
           '| Recall@10 | Variant | L (interpolated) | QPS | Mean ms | p95 ms | Actual p95 recall | Topology pages | Coord pages | PQ evals |',
           '|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    matches=[]
    for label in ('single','dual','dual_x'):
        curves=[r for r in records if r['variant']==label and r['accepted']]
        for t in TARGETS:
            for r in curves:
                pair=base.bracket(r['points'],t)
                if not pair:continue
                a,b=pair
                f=0 if a['L']==b['L'] else (t-a['recall'])/(b['recall']-a['recall'])
                near=min(r['points'],key=lambda p:abs(p['recall']-t))
                matches.append(dict(target=t,variant=label,repeat=r['repeat'],
                                    p95_us=near['p95_us'],p95_recall=near['recall'],
                                    **{k:a[k]+f*(b[k]-a[k]) for k in FIELDS+('L',)}))
            group=[r for r in matches if r['variant']==label and r['target']==t]
            if len(group)==3:
                def avg(k):return statistics.mean(p[k] for p in group)
                lines.append('| {}% | {} | {:.1f} | {:.0f} | {:.3f} | {:.3f} | {:.2f}% | {:.2f} | {:.2f} | {:.0f} |'.format(
                    t,label,avg('L'),avg('qps'),avg('latency_us')/1000,avg('p95_us')/1000,avg('p95_recall'),
                    avg('topology_submitted_pages'),avg('coordinate_submitted_pages'),avg('pq_space_evals')))
    base.save(ROOT/'analysis/matched.json',matches)
    (ROOT/'RESULTS.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def experiment():
    global TRUTH
    d=next(c for c in base.read(monitor.OLD/'analysis/inputs.json') if c['name']=='openimage')
    monitor.verify_inputs(d)
    index,prefix,TRUTH=verified_single(d)
    s=convert(index,d)
    c=base.prepare_budget(d,'2')
    layouts=tools.layouts(c)
    sx=base.build_layout(s,'S_T_X')
    x1,x2=sx/'group-2.bin',layouts['S_T_X']/'group-2.bin'
    if base.digest(x1)!=base.digest(x2):
        raise RuntimeError('single and dual X-only coordinate files differ')
    shared=sx/'shared-x'
    os.link(str(x2),str(shared))
    os.replace(str(shared),str(x1))
    specs={'single':(s,'S_T_X',sx,prefix,True),
           'dual':(c,'T_XY',layouts['T_XY'],c['prefix']+'/',False),
           'dual_x':(c,'S_T_X',layouts['S_T_X'],c['prefix']+'/',True)}
    selected={key:select(specs[key],key) for key in ('single','dual')}
    selected['dual_x']=selected['dual']
    x_path=Path(d['query1']).with_name('base_img_emb.fvecs')
    x=np.memmap(str(x_path),mode='r',dtype='<f4',offset=8,shape=(N,DIM))
    q=np.memmap(d['query1'],mode='r',dtype='<f4',offset=8,shape=(1000,DIM)).astype(np.float64)
    truth=np.fromfile(str(TRUTH),dtype='<u4',offset=8).reshape(1000,10)
    proof,expected={},{}
    for label,spec in specs.items():
        tag='parity-'+label
        points=query(spec,selected[label],tag,one=True,dump=True)
        expected[label]={p['L']:p for p in points}
        proof[label]=validate_answers(points,tag,x,q,truth)
    if proof['dual']!=proof['dual_x']:
        raise RuntimeError('dual anchor changed fixed-alpha answers')
    base.save(ROOT/'analysis/validation.json',dict(status='PASS',independent_graph_top10_verified=True,
              dual_anchor_same_answers=True,output_hashes=proof,shared_x_coordinate_inode=x2.stat().st_ino))
    monitor.CPUS=tools.choose_cpus()
    base.save(ROOT/'analysis/protocol.json',dict(datasets=['openimage'],threads=4,cpus=monitor.CPUS,numa=3,
              targets=TARGETS,levels=selected,cache_bytes=CACHE,repeat=3,order='archived M=2 Starling',
              truth_sha256=base.digest(TRUTH),dual_alpha=1))
    records=[]
    orders=(('single','dual','dual_x'),('dual','dual_x','single'),('dual_x','single','dual'))
    for rep,order in enumerate(orders,1):
        for variant in order:
            for attempt in (1,2,3):
                label='{}-r{}-a{}'.format(variant,rep,attempt)
                monitor.gate(label)
                pts=query(specs[variant],selected[variant],label)
                base.compare_logical(pts,expected[variant],label)
                quality=[monitor.telemetry(label,p) for p in pts]
                accepted=all(x['quiet'] for x in quality)
                records.append(dict(label=label,variant=variant,repeat=rep,attempt=attempt,
                                    accepted=accepted,points=pts,telemetry=quality))
                report(records)
                base.save(ROOT/'analysis/progress.json',dict(accepted=sum(x['accepted'] for x in records),
                          total=9,attempts=len(records),task=label,updated_at=base.now()))
                if accepted:break
    shutil.rmtree(str(ROOT/'scratch'))
    return records


def main():
    for d in ('analysis','bin','results','scratch'):(ROOT/d).mkdir(exist_ok=True)
    with (ROOT/'queue.lock').open('a') as lock,(base.PREV/'queue.lock').open('r') as shared:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        base.status('waiting_for_serial_lock')
        fcntl.flock(shared,fcntl.LOCK_EX)
        tools.build()
        tools.prepare_window()
        records=experiment()
    accepted=sum(r['accepted'] for r in records)
    base.status('complete' if accepted==9 else 'complete_with_flagged_points',accepted=accepted,
                total=9,points=sum(len(r['points']) for r in records if r['accepted']),serial_lock_released=True)


if __name__=='__main__':
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGHUP):signal.signal(sig,base.stop)
    try:
        main()
    except Exception as e:
        base.status('interrupted' if isinstance(e,InterruptedError) else 'failed',error=str(e))
        traceback.print_exc()
        sys.exit(1)
