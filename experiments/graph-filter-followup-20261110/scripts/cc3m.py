"""Prepare the CC3M X-only graph and reproduce the three endpoint controls."""
import os
import re
from pathlib import Path
import shutil
import struct
import sys

import followup as common
from repair_single import repair

base, monitor = common.base, common.monitor
EXP = common.EXP
ROOT = common.ROOT / 'cc3m'
SOURCE = EXP / 'openimage-single-space-20261110'
sys.path.insert(0,str(SOURCE/'scripts'))
import prepare as prep
import numpy as np

N, DIM, PAGE = 1238875, 1024, 8192
TARGETS = (90,95,98)


def config():
    c = next(c for c in base.read(base.FIRST/'analysis/inputs.json') if c['name']=='cc3m')
    c.update(order=str(base.PREV/'inputs/cc3m/b2/order.bin'),
             b2_identity=base.read(base.PREV/'analysis/cc3m__b2-index-identity.json'))
    monitor.verify_inputs(c)
    return base.prepare_budget(c,'2')


def setup():
    for d in ('analysis','bin','results','scratch','logs','index'):
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    common.ROOT = common.tools.ROOT = base.ROOT = monitor.ROOT = common.prior.ROOT = ROOT
    root = ROOT.parent
    base.ENGINE = root/'bin/endpoint-search'
    if base.digest(base.ENGINE) != base.read(root/'analysis/build.json')['binary_sha256']:
        raise RuntimeError('archived search binary changed')


def candidate_truth(x_data, queries, target, excluded):
    inventory = []
    for path in sorted(Path(excluded).parent.parent.glob('top10*')):
        raw = path.read_bytes()
        try:
            if len(raw)==40008 and struct.unpack_from('<II',raw)==(1000,10):
                ids = np.frombuffer(raw,dtype='<u4',offset=8).reshape(1000,10)
            elif len(raw)==44000:
                rows = np.frombuffer(raw,dtype='<u4').reshape(1000,11)
                if not np.all(rows[:,0]==10):
                    raise ValueError('invalid ivecs rows')
                ids = rows[:,1:].copy()
            else:
                raise ValueError('unknown header')
            if ids.max()>=N:
                raise ValueError('out of range')
            delta = x_data[ids].astype(np.float64)-queries.astype(np.float64)[:,None,:]
            ds = np.einsum('qkd,qkd->qk',delta,delta).astype('<f4')
            check = target.with_name('candidate-x-gt.bin')
            check.write_bytes(struct.pack('<II',1000,10)+ids.tobytes()+ds.tobytes()+ids.tobytes())
            proof = prep.validate_gt(check,x_data,queries)
        except (ValueError,RuntimeError) as exc:
            inventory.append(dict(path=str(path),sha256=base.digest(path),reused=False,reason=str(exc)))
        else:
            os.replace(str(check),str(target))
            inventory.append(dict(path=str(path),sha256=base.digest(path),reused=True))
            return proof, inventory
    check = target.with_name('candidate-x-gt.bin')
    if check.exists():
        check.unlink()
    return None, inventory


def prepare():
    setup()
    prep.ROOT, prep.N, prep.DIM = ROOT,N,DIM
    if len(os.sched_getaffinity(0))!=2 or not os.sched_getaffinity(0)<=set(range(72,96)):
        raise RuntimeError('CC3M preparation requires two physical cores in NUMA 3')
    base.require_space(20<<30)
    c = config()
    prep.disk_gate('cc3m-inputs')
    base.status('validating_source_vectors')
    x,q,data,queries = prep.input_data(c)
    prefix = ROOT/'index/cc3m_x'
    for suffix in ('_pq_compressed.bin','_pq_pivots.bin'):
        src = Path(c['prefix'])/('emb'+suffix)
        target = Path(str(prefix)+suffix)
        if not target.exists():
            shutil.copyfile(str(src),str(target))
        if base.digest(src)!=base.digest(target):
            raise RuntimeError('X PQ changed')
    codes = Path(str(prefix)+'_pq_compressed.bin')
    with codes.open('rb') as f:
        if struct.unpack('<II',f.read(8))!=(N,64) or codes.stat().st_size!=8+N*64:
            raise RuntimeError('CC3M X PQ code header')
    gt = ROOT/'index/cc3m_x_l2_top10.bin'
    gt_proof = ROOT/'analysis/gt-validation.json'
    if not gt_proof.exists():
        base.status('checking_existing_X_gt')
        proof, candidates = candidate_truth(data,queries,gt,c['gt'])
        if proof is None:
            prep.disk_gate('cc3m-exact-gt')
            base.status('computing_X_only_gt')
            gt_tool = base.BUILD/'build-separated/tests/utils/compute_groundtruth'
            base.run([gt_tool,'float',x,q,10,gt],'groundtruth',
                     dict(prep.ENV,OPENBLAS_NUM_THREADS='2'))
            proof = prep.validate_gt(gt,data,queries)
        base.save(gt_proof,dict(status='PASS',candidates=candidates,excluded_mixed_gt=c['gt'],
                                computed=not any(p['reused'] for p in candidates),
                                sha256=base.digest(gt),**proof))
    elif base.digest(gt)!=base.read(gt_proof)['sha256']:
        raise RuntimeError('prepared CC3M X GT changed')
    builder = SOURCE/'bin/build-single'
    if base.digest(builder)!=base.read(SOURCE/'analysis/protocol.json')['builder_sha256']:
        raise RuntimeError('validated builder changed')
    index = Path(str(prefix)+'_disk.index')
    validation = ROOT/'analysis/index-validation.json'
    if not validation.exists():
        command = [builder,x,prefix]
        build_run = ROOT/'analysis/build-vamana-run.json'
        build_log = ROOT/'logs/build-vamana.log'
        if (not index.is_file() or not build_run.is_file() or not build_log.is_file()
                or base.read(build_run)['command'] != [str(arg) for arg in command]
                or 'SINGLE_SPACE_BUILD_COMPLETE' not in build_log.read_text().splitlines()):
            prep.disk_gate('cc3m-vamana-build')
            base.status('building_independent_X_graph',cpus=sorted(os.sched_getaffinity(0)))
            base.run(command,'build-vamana',prep.ENV)
        base.status('validating_independent_X_graph')
        repair_record = None
        try:
            proof = prep.validate_disk(index,data)
        except RuntimeError as exc:
            reached = re.fullmatch(r'entry can only reach (\d+) of (\d+) objects',str(exc))
            if not reached or int(reached.group(2))!=N:
                raise
            base.require_space(index.stat().st_size)
            prep.disk_gate('cc3m-connectivity-repair')
            base.status('repairing_independent_X_graph',unreachable=N-int(reached.group(1)))
            repaired = index.with_name('cc3m_x_disk_connected.index')
            repair_record = repair(index,repaired,data,N-int(reached.group(1)),PAGE)
            repair_record.update(source=index.name,source_sha256=base.digest(index))
            index = repaired
            proof = prep.validate_disk(index,data)
        base.save(validation,dict(status='PASS',filename=index.name,sha256=base.digest(index),
                                  connectivity_repair=repair_record,**proof))
    proof = base.read(validation)
    index = index.with_name(proof.get('filename',index.name))
    if base.digest(index)!=proof['sha256']:
        raise RuntimeError('prepared CC3M Vamana index changed')
    base.save(ROOT/'analysis/preparation.json',dict(status='PASS',source=c,
              base_sha256=base.digest(x),query_sha256=base.digest(q),builder_sha256=base.digest(builder),
              index_filename=index.name,index_sha256=proof['sha256'],gt_sha256=base.digest(gt),
              connectivity_repair=proof.get('connectivity_repair'),cores=sorted(os.sched_getaffinity(0)),
              R=40,L_build=100,numa=3))


def compare():
    setup()
    common.tools.prepare_window()
    c = config()
    preparation = base.read(ROOT/'analysis/preparation.json')
    prefix = ROOT/'index/cc3m_x'
    index = prefix.with_name(preparation.get('index_filename','cc3m_x_disk.index'))
    gt = ROOT/'index/cc3m_x_l2_top10.bin'
    for p,k in ((index,'index_sha256'),(gt,'gt_sha256')):
        if base.digest(p)!=preparation[k]:
            raise RuntimeError('verified CC3M input changed')
    truth = ROOT/'analysis/x-top10-ids.bin'
    truth.write_bytes(gt.read_bytes()[:40008])
    canon = ROOT/'scratch/canonical-x'
    if not (ROOT/'analysis/conversion.json').exists():
        raw = np.memmap(str(index),mode='r',dtype=np.uint8)
        nodes,dim,medoid,width,npp,frozen,_,size = struct.unpack_from('<8Q',raw,8)
        if (nodes,dim,width,npp,frozen,size)!=(N,DIM,4260,1,0,len(raw)) or medoid>=N:
            raise RuntimeError('unsupported CC3M Vamana packed index')
        base.status('converting_single_graph')
        base.require_space(6<<30)
        canon.mkdir(exist_ok=True)
        rows = np.ndarray((N,width//4),dtype='<u4',buffer=raw,offset=PAGE,strides=(PAGE,4))
        topo_pp = PAGE//164
        pages = (N+topo_pp-1)//topo_pp
        meta = bytearray(PAGE)
        struct.pack_into('<IIIIIQII',meta,0,N,DIM,0,40,0,topo_pp,1,medoid)
        (canon/'meta.bin').write_bytes(meta)
        coords = np.memmap(str(canon/'coord.bin'),mode='w+',dtype='<u4',shape=((N+1)//2*2,DIM))
        tops = np.memmap(str(canon/'topo.bin'),mode='w+',dtype=np.uint8,shape=(pages*PAGE,))
        topology = np.ndarray((pages,topo_pp,41),dtype='<u4',buffer=tops,strides=(PAGE,164,4))
        for lo in range(0,N,8192):
            ids = np.arange(lo,min(N,lo+8192))
            coords[ids] = rows[ids,:DIM]
            topology[ids//topo_pp,ids%topo_pp] = rows[ids,DIM:]
        coords.flush()
        tops.flush()
        for lo in range(0,N,8192):
            ids = np.arange(lo,min(N,lo+8192))
            if (not np.array_equal(coords[ids],rows[ids,:DIM]) or
                    not np.array_equal(topology[ids//topo_pp,ids%topo_pp],rows[ids,DIM:])):
                raise RuntimeError('canonical conversion changed a node')
        base.save(ROOT/'analysis/conversion.json',dict(status='PASS',nodes=N,entry=medoid,
                  coordinates_and_order_preserved=True))
    single = dict(c,meta=str(canon/'meta.bin'),coord=str(canon/'coord.bin'),topo=str(canon/'topo.bin'))
    dx = common.layout(c,'cc3m-dual-x','S_T_X')
    dy = common.layout(c,'cc3m-dual-xy','T_XY')
    sx = common.layout(single,'cc3m-single','S_T_X',dx/'group-2.bin')
    if (sx/'group-2.bin').stat().st_ino!=(dx/'group-2.bin').stat().st_ino:
        raise RuntimeError('single and dual X must share coordinate file')
    if base.digest(dx/'group-1.bin')!=base.digest(dy/'group-1.bin'):
        raise RuntimeError('dual path changed topology')
    p = c['prefix']+'/'
    specs = {'single':(single,sx,'S_T_X',prefix,truth,True,False),
             'dual':(c,dy,'T_XY',p,truth,False,False),
             'dual_x':(c,dx,'S_T_X',p,truth,True,False)}
    monitor.CPUS = common.quiet_cores()
    import edge_audit
    audit_result = edge_audit.ROOT/'analysis/edge-audit.json'
    if not audit_result.exists() or base.read(audit_result)['status']!='PASS':
        try:
            edge_audit.run(monitor.CPUS)
        except Exception as exc:
            base.save(audit_result,dict(status='failed',error=str(exc)))
            print('OpenImages trace audit failed:',exc,flush=True)
    base.save(ROOT/'analysis/protocol.json',dict(threads=4,numa=3,cpus=monitor.CPUS,alpha=1,
              targets=TARGETS,cache_bytes=common.CACHE,gt_sha256=base.digest(truth),order_sha256=base.digest(c['order']),
              single_index_sha256=preparation['index_sha256'],connectivity_repair=preparation.get('connectivity_repair')))
    grid = [10,20,30,40,60,80,120,176,240,320,400,500,600,800,1200,1600]
    screens = {}
    for v,spec in specs.items():
        if v=='dual_x':
            levels = sorted({p['L'] for p in screens['dual']})
        else:
            # Screen moderate L first and extend only if a requested target is uncovered.
            levels = grid[:11]
        screens[v] = common.run_search(spec,levels,v+'-screen',True)
        if v!='dual_x' and not base.bracket(screens[v],98):
            screens[v].extend(common.run_search(spec,grid[11:],v+'-extend',True))
    for v in ('single','dual','dual_x'):
        selected = common.levels_for(screens[v],TARGETS)
        if not selected:
            raise RuntimeError('no covered target for '+v)
    for L in [p['L'] for p in screens['dual']]:
        left = 'dual-screen' if L in grid[:11] else 'dual-extend'
        for ext in ('ids','dists'):
            if base.digest(ROOT/'analysis'/('{}{}.{}'.format(left+'.L',L,ext))) != base.digest(
                    ROOT/'analysis'/('dual_x-screen.L{}.{}'.format(L,ext))):
                raise RuntimeError('dual endpoint parity mismatch')
    common.prior.N, common.prior.DIM = N,DIM
    x = np.memmap(str(Path(c['query1']).with_name('base_img_emb.fvecs')),mode='r',dtype='<f4',
                  offset=8,shape=(N,DIM))
    q = np.memmap(c['query1'],mode='r',dtype='<f4',offset=8,shape=(1000,DIM)).astype(np.float64)
    gt_ids = np.fromfile(str(truth),dtype='<u4',offset=8).reshape(1000,10)
    checks = {}
    for v,rows in screens.items():
        checks[v] = {}
        for suffix in ('screen','extend'):
            subset = [p for p in rows if (ROOT/'analysis'/('{}-{}.L{}.ids'.format(v,suffix,p['L']))).exists()]
            if subset:
                checks[v].update(common.prior.validate_answers(subset,v+'-'+suffix,x,q,gt_ids))
    base.save(ROOT/'analysis/validation.json',dict(status='PASS',output_hashes=checks))
    base.save(ROOT/'analysis/diagnostics.json',screens)
    records = common.benchmark(specs,screens,TARGETS)
    accepted = sum(r['accepted'] for r in records)
    base.status('complete' if accepted==9 else 'complete_with_flagged_points',accepted=accepted,total=9)
