"""Read-only diagnostics plus an isolated, identity-checked 10k trace replay.

Independent of G1: waits for the same global lock, never holds G1 waiting for
its results, never repairs a graph or changes selection semantics.
"""
from pathlib import Path
import csv
import fcntl
import hashlib
import json
import os
import struct
import time
import run_g0 as c

AB = c.HOST_ROOT/'experiments/backlink-ab-20261115/runtime'
DATA = Path('/mnt/nvme2/wz/hvs-u-shape-20261113/subset-2m')


def matrix(p):
    import numpy as np
    with Path(p).open('rb') as f:
        n,d = struct.unpack('<II',f.read(8))
    return np.memmap(str(p),dtype='<f4',mode='r',offset=8,shape=(n,d))


def save_status(stage, **values):
    c.save('status.json',dict(stage=stage,updated_unix=time.time(),**values))


def capture():
    src = c.ROOT/'implementation'
    out = c.RUN/'capture10k'
    out.mkdir(exist_ok=True)
    original = AB/'build/hybrid_index.cpp'
    proof = json.loads((AB/'validation/build.json').read_text())
    assert c.sha(original) == proof['variant']
    text = original.read_text(encoding='utf-8')
    for needle,before,after in [
        ('prune_neighbors_no_opt(pool, parameters, pruned_list);',
         'hvs_diag_candidates(node,node,"forward",pool);',
         'hvs_diag_selected(node,node,"forward",pruned_list);'),
        ('prune_neighbors_no_opt(candidates,parameter,selected);',
         'hvs_diag_candidates(des.id_,n,"reverse",candidates);',
         'hvs_diag_selected(des.id_,n,"reverse",selected);')]:
        assert text.count(needle) == 1, needle
        text = text.replace(needle,before+'\n'+needle+'\n'+after)
    text = '#include "connectivity_capture.h"\n'+text
    unit = c.RUN/'capture_hybrid.cpp'
    unit.write_text(text,encoding='utf-8')
    flags = proof['binaries']['B']['flags']+['-I'+str(src)]
    obj = c.RUN/'capture_hybrid.o'
    binary = c.RUN/'capture_builder'
    c.command(['g++-11']+flags+['-c',unit,'-o',obj],'compile-capture-object',300)
    # All HybridIndex symbols come from this object. The archive supplies other
    # unchanged components, without modifying or copying the official library.
    c.command(['g++-11']+flags+[AB/'build/build_graph.cpp',obj,
               AB/'build/B/libpipeann.a']+proof['link']+['-o',binary],
              'compile-capture-builder',180)
    previous = dict(c.ENV)
    c.ENV.update(OMP_NUM_THREADS='1',OMP_THREAD_LIMIT='1',HVS_SOURCE_ONLY='1',
                 HVS_FULL_ONLY='1',HVS_CONNECTIVITY_DIAG_DIR=str(out))
    try:
        c.command([binary,'dual',AB/'data-10k/base_img_emb.fvecs',
                   AB/'data-10k/base_text_emb.fvecs',10000,512,512,1,out],
                  'capture-build-10k',300)
    finally:
        c.ENV.clear();c.ENV.update(previous)
    equal = c.sha(out/'float-graph.bin') == c.sha(AB/'gate-graphs/B/float-graph.bin')
    result = dict(graph_byte_equal=equal,source_sha256=c.sha(original),
        capture_source_sha256=c.sha(unit),binary_sha256=c.sha(binary),flags=flags,
        original_graph_sha256=c.sha(AB/'gate-graphs/B/float-graph.bin'),
        captured_graph_sha256=c.sha(out/'float-graph.bin'),
        scope='only logging around unchanged forward/reverse selection; one-thread same-input replay')
    c.save('capture-identity.json',result)
    if not equal:
        raise RuntimeError('diagnostic capture did not reproduce the original graph')
    import numpy as np
    x = matrix(AB/'data-10k/base_img_emb.fvecs')
    y = matrix(AB/'data-10k/base_text_emb.fvecs')
    candidates = list(csv.DictReader((out/'construction-candidates.csv').open()))
    with (out/'candidate-distances-float64.csv').open('w') as f:
        writer = csv.writer(f)
        writer.writerow(['owner','origin','kind','candidate','builder_dx','builder_dy',
                         'dx_to_8947','dy_to_8947','dx_to_118','dy_to_118'])
        for row in candidates:
            v = int(row['candidate']);ds=[]
            for u in (8947,118):
                ds += [float(np.linalg.norm(a[u].astype('float64')-a[v].astype('float64'))) for a in (x,y)]
            writer.writerow([row[k] for k in ('owner','origin','kind','candidate','dist_x','dist_y')]+ds)
    duplicate_ids = []
    # Exact full-10k scan for the motivating point, no graph-neighbor shortcut.
    for begin in range(0,len(x),2048):
        end = min(len(x),begin+2048)
        dx = np.asarray(x[begin:end],dtype='float64')-x[8947].astype('float64')
        dy = np.asarray(y[begin:end],dtype='float64')-y[8947].astype('float64')
        ids = np.flatnonzero(np.einsum('ij,ij->i',dx,dx)+np.einsum('ij,ij->i',dy,dy)==0)+begin
        duplicate_ids.extend(map(int,ids))
    chosen = list(csv.DictReader((out/'construction-selected.csv').open()))
    forward = {str(u):[int(v['selected']) for v in chosen if int(v['owner'])==u and v['kind']=='forward'] for u in (118,8947)}
    value = dict(pair_8947_118=[float(np.linalg.norm(a[8947].astype('float64')-a[118].astype('float64'))) for a in (x,y)],
        exact_duplicate_group_including_self=duplicate_ids,
        forward_candidates={str(u):sum(int(v['owner'])==u and v['kind']=='forward' for v in candidates) for u in (118,8947)},
        forward_selected=forward,
        captured_reverse_118_from_8947=[v for v in chosen if v['kind']=='reverse' and int(v['owner'])==118 and int(v['origin'])==8947],
        graph_identity_proven=equal,
        distance_table=str(out/'candidate-distances-float64.csv'))
    c.save('motivating-pair.json',value)
    return value


def radius_matches(x,y,target_ids):
    """Exact radius candidates via a necessary coordinate bound, then full L2.

    Every qualifying pair satisfies the bound on this first-space coordinate;
    no graph adjacency, ANN lookup or probabilistic sampling excludes a match.
    """
    import numpy as np
    targets = sorted(target_ids)
    sample = np.linspace(0,len(x)-1,min(2048,len(x)),dtype=np.int64)
    dim = int(np.argmax(np.std(np.asarray(x[sample],dtype='float64'),axis=0)))
    values = np.asarray(x[:,dim]).copy()
    assert np.isfinite(values).all()
    order = np.argsort(values,kind='mergesort')
    sorted_values = values[order].astype('float64')
    eps = 1e-6
    flags = {}
    exact_groups = {}
    with (c.RUN/'low-degree-duplicate-evidence.jsonl').open('w') as evidence:
        for at,u in enumerate(targets):
            qx = x[u].astype('float64');qy = y[u].astype('float64')
            key = hashlib.sha256(qx.tobytes()+qy.tobytes()).digest()
            cached = exact_groups.get(key)
            if cached is not None:
                mate = cached[0] if cached[0] != u else cached[1]
                zero = bool(np.array_equal(x[mate],x[u]) and np.array_equal(y[mate],y[u]))
                if zero:
                    record = dict(id=u,exact_zero_both=True,lt_1e_6_both=True,mate=int(mate),dx=0.,dy=0.,verified_cached_group=True)
                    flags[u]=record;evidence.write(json.dumps(record)+'\n');continue
            lo = np.searchsorted(sorted_values,np.nextafter(float(values[u])-eps,-np.inf),'left')
            hi = np.searchsorted(sorted_values,np.nextafter(float(values[u])+eps,np.inf),'right')
            choices = order[lo:hi]
            zero,near,mate,dist = False,False,None,None
            for begin in range(0,len(choices),4096):
                ids = choices[begin:begin+4096]
                ids = ids[ids != u]
                if not len(ids):continue
                dx = np.asarray(x[ids],dtype='float64')-qx
                dy = np.asarray(y[ids],dtype='float64')-qy
                xx = np.einsum('ij,ij->i',dx,dx);yy = np.einsum('ij,ij->i',dy,dy)
                z = np.flatnonzero((xx==0)&(yy==0))
                if len(z):
                    zero=near=True;mate=int(ids[z[0]]);dist=(0.,0.)
                    exact_groups[key]=(u,mate)
                    break
                z = np.flatnonzero((xx<eps*eps)&(yy<eps*eps))
                if len(z) and not near:
                    near=True;mate=int(ids[z[0]]);dist=(float(np.sqrt(xx[z[0]])),float(np.sqrt(yy[z[0]])))
            record=dict(id=u,exact_zero_both=zero,lt_1e_6_both=near,mate=mate,
                        dx=dist[0] if dist else None,dy=dist[1] if dist else None,
                        projection_candidate_count=int(hi-lo),projection_dimension=dim)
            flags[u]=record
            evidence.write(json.dumps(record)+'\n')
            if at%100==0:
                evidence.flush();save_status('radius_diagnostics',completed=at+1,total=len(targets))
    c.save('radius-method.json',dict(epsilon=eps,projection_dimension=dim,
        complete=True,targets=len(targets),definition='point count with at least one other point; strict full L2<1e-6 in both spaces; exact zero is a subset',
        method='necessary first-space coordinate bound over all rows, followed by float64 full-vector distances in both spaces; zero groups reused only after exact vector equality check'))
    return flags


def main():
    c.RUN=c.ROOT/'runtime/revision3-diagnostics'
    c.RUN.mkdir(parents=True,exist_ok=True)
    lock=(c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    save_status('waiting_global_lock')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        save_status('capturing_10k_candidates')
        pair=capture()
        src=c.ROOT/'implementation'
        flags=['-std=c++17','-O3','-fno-fast-math','-ffp-contract=off']
        c.command(['clang-format','-style=LLVM','-i',src/'connectivity_audit.cpp'],'format-connectivity')
        c.command(['g++-11']+flags+[src/'connectivity_audit.cpp','-o',c.RUN/'connectivity_audit'],'compile-connectivity')
        rows=[];targets=set()
        for label in ('A1','B1','A2','B2'):
            packed=AB/'formal-ab-v2/packed'/label
            graph=packed/'compact.topo'
            for policy,meta in [('native',AB/'formal-ab-v2/graphs'/label/'full.meta'),
                                ('shared-search-entries',packed/'compact.meta')]:
                name=label+'-'+policy;out=c.RUN/name
                save_status('graph_connectivity',graph=label,entry_policy=policy)
                c.command([c.RUN/'connectivity_audit',meta,graph,out],'audit-'+name,180)
                result=json.loads((c.RUN/('audit-'+name+'.log')).read_text())
                low=list(csv.DictReader(Path(str(out)+'-low-degree.csv').open()))
                target={int(v['id']) for v in low};targets.update(target)
                result.update(graph=label,entry_policy=policy,meta_sha256=c.sha(meta),topology_sha256=c.sha(graph),
                              low_ids=sorted(target))
                rows.append(result)
        x=matrix(DATA/'base_img_emb.fvecs');y=matrix(DATA/'base_text_emb.fvecs')
        assert x.shape==y.shape==(2000000,512)
        flags=radius_matches(x,y,targets)
        for row in rows:
            ids=row.pop('low_ids')
            row['low_degree_exact_zero_both']=sum(flags[v]['exact_zero_both'] for v in ids)
            row['low_degree_lt_1e_6_both']=sum(flags[v]['lt_1e_6_both'] for v in ids)
        openimages=dict(status='PENDING_VERIFIED_A_B_INPUTS',
            reason='No OpenImages graph with verified always-reprune B identity located in the A/B artifact catalog; legacy b2 names are layouts, not proof of policy B. Requested paths from user.',
            no_new_openimages_build_started=True)
        result=dict(status='LAION_DIAGNOSTICS_COMPLETE_OPENIMAGES_PENDING',motivating_pair=pair,
                    Laion2M=rows,OpenImages=openimages,graph_repair_applied=False,
                    pruning_changes_applied=False,G1_independent=True)
        c.save('diagnostic-results.json',result)
        save_status(result['status'])
    except Exception as e:
        save_status('DIAGNOSTIC_ERROR',reason=str(e))
        raise


if __name__=='__main__':main()
