"""Serial four-dataset budget, locality, recall and field-group experiments."""
import collections
import itertools
import json
import os
from pathlib import Path
import shutil
import statistics
import struct
import subprocess
import time
from run_all import ROOT,BUILD,PREVIOUS,STAR,ENV,run,save,status,log,digest,now

ENGINE=ROOT/'bin/layout-engine'
PLAIN=ROOT/'bin/layout-engine-no-timing'
REFERENCE=PREVIOUS/'bin/layout-engine'
PARTITIONER=STAR/'release-gcc11-abi0/graph_partition/partitioner'
BASE_LS=[20,40,60,80,100,120,160,200,300,400]
EXTRA_LS=[600,800,1200,1600]
LAYOUTS=['T_XY','TXY','T_X_Y']
SECONDS=10

def invoke(cmd,label,output=None,env=None,observe=False,cost=False):
    cmd=[str(x) for x in cmd];e=dict(ENV);e.update(env or {})
    record=ROOT/'analysis'/(label+'-run.json')
    if output and Path(output).exists() and record.exists():return json.loads(Path(output).read_text())
    pending=None
    if output:
        output=Path(output);pending=output.with_name(output.name+'.partial')
        cmd=[str(pending) if x==str(output) else x for x in cmd]
    status(status='running',stage='budget-reorder',task=label)
    log('RUN '+label)
    launch=cmd
    if cost:launch=['/usr/bin/time','-v','-o',str(ROOT/'analysis'/(label+'-time.txt'))]+cmd
    with open(str(ROOT/'logs'/(label+'.log')),'w') as out,open(str(ROOT/'logs'/(label+'.pidstat')),'w') as telemetry:
        started=time.time();p=subprocess.Popen(launch,env=e,stdout=out,stderr=subprocess.STDOUT)
        monitor=subprocess.Popen(['pidstat','-h','-u','-r','-w','-p',str(p.pid),'1'],stdout=telemetry,stderr=subprocess.STDOUT) if observe else None
        try:code=p.wait()
        finally:
            if monitor:monitor.terminate();monitor.wait()
        if code:raise RuntimeError('{} exit {}'.format(label,code))
    data=None
    if pending:
        data=json.loads(pending.read_text());os.replace(str(pending),str(output))
        for f in pending.parent.glob(pending.name+'.L*.queries.json'):
            os.replace(str(f),str(f.with_name(f.name.replace(pending.name,output.name,1))))
    save(record,dict(command=cmd,started_unix=started,finished_unix=time.time(),
         environment={k:v for k,v in e.items() if k.startswith(('LAYOUT_','OMP_','OPENBLAS_','MKL_'))}))
    log('DONE '+label)
    return data

def args(c,layout,folder,op,out,binary=ENGINE):
    return [binary,op,c['meta'],c['topo'],c['coord'],folder,layout,out]
def query(c,layout,folder,out,threads,ls,binary=ENGINE):
    return args(c,layout,folder,'query',out,binary)+[c['prefix']+'/',c['query1'],c['query2'],c['alpha'],c['gt'],threads,SECONDS]+ls
def order_env(c,ordering):
    return dict(LAYOUT_ORDER=c['order_file']) if ordering=='starling' else {}
def key(c):return c['name']+'__b'+c['budget']

def build():
    spec=json.loads((BUILD/'experiments/concurrency-alpha/analysis/build.json').read_text())['separated']
    if digest(spec['library'])!=spec['library_sha256']:raise RuntimeError('Original library changed')
    cpp=['layout_engine.cpp','layout_store.h','convert_budget.cpp','test_layout_io.cpp']
    run(['clang-format','-i','-style=LLVM']+[ROOT/'scripts'/n for n in cpp],'format-code')
    source_hashes={p.name:digest(p) for p in (ROOT/'scripts').iterdir() if p.is_file() and p.suffix in ['.py','.h','.cpp','.sh']}
    manifest=ROOT/'analysis/build.json'
    if manifest.exists():
        old=json.loads(manifest.read_text())
        if old['sources']!=source_hashes or old['library_sha256']!=spec['library_sha256']:raise RuntimeError('Source change requires a new experiment identity')
        if all(digest(ROOT/'bin'/n)==h for n,h in old['binaries'].items()):
            if not (ROOT/'analysis/layout-unit-tests.json').exists():
                subprocess.run(['python3',str(ROOT/'scripts/test_layout.py')],env=ENV,check=True)
            return
    commands=[]
    for name,extra in [('layout-engine',[]),('layout-engine-no-timing',['-DLAYOUT_NO_TIMING'])]:
        cmd=['g++-11']+spec['flags']+extra+[ROOT/'scripts/layout_engine.cpp',spec['library'],
            '-L/mnt/nvme2/wz/disk/local/openblas/lib','-laio','-ljemalloc','-lopenblas','-o',ROOT/'bin'/name]
        run(cmd,'build-'+name);commands.append([str(x) for x in cmd])
    cmd=['g++-11','-std=c++17','-O3',ROOT/'scripts/convert_budget.cpp','-laio','-o',ROOT/'bin/convert-budget']
    run(cmd,'build-converter');commands.append([str(x) for x in cmd])
    save(manifest,dict(source_commit='d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e',
        sources=source_hashes,library_sha256=spec['library_sha256'],commands=commands,
        binaries={n:digest(ROOT/'bin'/n) for n in ['layout-engine','layout-engine-no-timing','convert-budget']},
        reference_binary_sha256=digest(REFERENCE),starling_binary_sha256=digest(PARTITIONER)))
    subprocess.run(['python3',str(ROOT/'scripts/test_layout.py')],env=ENV,check=True)

def provenance(cs):
    save(ROOT/'analysis/inputs.json',cs)
    for label,folder in [('hvs-source',BUILD/'source'),('starling',STAR),('starling-partitioner',STAR/'graph_partition')]:
        head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=str(folder)).decode().strip()
        dirty=subprocess.check_output(['git','status','--short'],cwd=str(folder)).decode()
        diff=subprocess.check_output(['git','diff'],cwd=str(folder)).decode()
        if label=='hvs-source' and (head!='d2a39e2bd50e6d03716c4268f7b90f931e5a6a1e' or dirty):
            raise RuntimeError('HVS source identity changed')
        save(ROOT/'analysis'/(label+'-source.json'),dict(path=str(folder),commit=head,status=dirty,diff=diff))
    save(ROOT/'analysis/protocol.json',dict(datasets=[c['name'] for c in cs],serial=True,max_threads=64,
        cpus='0-47,96-111',numa_memory=[0,1],blas_threads=1,budgets=['full','2','0'],
        layouts=LAYOUTS,orders=['original','starling'],main_threads=[1,32],diagnostic_threads=[4,64],
        main_query_runs=432,diagnostic_query_runs=432,laion_additional_alpha_runs=144,
        query_seconds_per_point=SECONDS,repeats=3,L=BASE_LS,high_recall_probe_L=EXTRA_LS,
        graph_partition_threads=32,graph_partition_LDG_passes=8,
        same_permutation_across_layouts_within_budget=True,partition_recomputed_per_budget=True,
        cache_bytes_per_query=16<<20,shared_order_arrays_bytes_per_node=8,
        coord_batch_nodes=256,max_AIO_batch=128,new_GT=False,
        patch_operations=10000,patch_batch=128,patch_fixed_buffer_bytes=2<<20,
        patch_acknowledgement='after per-batch fdatasync in both modes, no WAL',
        complete_dynamic_ANN_updates=False,single_space_ANN_GT='not supplied; storage-only controls'))

def alpha_conditions(c):
    out=[dict(c,condition='default')]
    if c['name']=='laion':
        data=Path(c['query1']).parent
        for i in [1,3,7,9]:
            a=data/('alpha%d.bin'%i);g=data/('gt/base_gt_round_00/top10_results_alpha_0.%d.ivecs'%i)
            if not a.is_file() or not g.is_file():raise RuntimeError('Supplied Laion alpha/GT missing')
            if struct.unpack('<II',a.read_bytes()[:8])!=(1000,1) or struct.unpack('<II',g.read_bytes()[:8])!=(1000,10):raise RuntimeError('alpha/GT header')
            out.append(dict(c,alpha=str(a),gt=str(g),condition='a%d'%i))
    return out

def prepare_budget(c,budget):
    c=dict(c,budget=budget);folder=ROOT/'inputs'/c['name']/('b'+budget);folder.mkdir(parents=True,exist_ok=True)
    if budget!='full':
        output=ROOT/'analysis'/(key(c)+'-conversion.json')
        meta=folder/'meta.bin';topo=folder/'topo.bin'
        invoke([ROOT/'bin/convert-budget',c['meta'],c['topo'],c['coord'],meta,topo,budget,output],key(c)+'-conversion',output,cost=True)
        c.update(meta=str(meta),topo=str(topo))
    raw=Path(c['meta']).read_bytes();n,d1,d2,degree,slots,npp,entries=struct.unpack_from('<IIIIIQI',raw)
    c.update(nodes=n,d1=d1,d2=d2,degree=degree,slots=slots)
    partition=folder/'partition.bin';order=folder/'order.bin';valid=ROOT/'analysis'/(key(c)+'-partition-validation.json')
    if not order.exists() or not valid.exists():
        invoke([PARTITIONER,'--data_type','float','--gp_file',partition,'--custom_graph','1',
                '--graph_file',c['topo'],'--meta_file',c['meta'],'--ep_size',entries,'-T','32','-L','8'],key(c)+'-starling',cost=True)
        invoke(['python3',ROOT/'scripts/partition_order.py',partition,order,valid],key(c)+'-partition-check',valid,cost=True)
    c['order_file']=str(order)
    save(ROOT/'analysis'/(key(c)+'-index-identity.json'),dict(meta_sha256=digest(c['meta']),
        topo_sha256=digest(c['topo']),coord_sha256=c['source_hashes']['coord'],
        order_sha256=digest(order),partition_sha256=digest(partition),slots=slots,topo_nodes_per_page=npp))
    return c

def expected_bytes(c,layout):
    masks={'TXY':[7],'T_XY':[1,6],'T_X_Y':[1,2,4],'S_TX':[3],'S_T_X':[1,2]}[layout]
    widths=[4+c['degree']*(4+2*c['slots']),c['d1']*4,c['d2']*4]
    total=0
    for mask in masks:
        width=sum(w for k,w in enumerate(widths) if mask&(1<<k));npp=8192//width
        if not npp:raise RuntimeError('Record overflow')
        total+=(c['nodes']+npp-1)//npp*8192
    return total

def build_layout(c,layout,ordering,label):
    folder=ROOT/'scratch'/label
    if folder.exists():shutil.rmtree(str(folder))
    folder.mkdir()
    if shutil.disk_usage(str(folder)).free<expected_bytes(c,layout)+(24<<30):raise RuntimeError('Insufficient nvme3 headroom')
    out=ROOT/'analysis'/(label+'-build.json')
    # A generated layout can never be resumed after an interrupted patch.
    if out.exists():out.unlink()
    invoke(args(c,layout,folder,'build',out),label+'-build',out,order_env(c,ordering),cost=True)
    save(ROOT/'analysis'/(label+'-physical.json'),dict(layout=layout,ordering=ordering,
        group_hashes={p.name:digest(p) for p in folder.glob('group-*.bin')},physical_bytes=expected_bytes(c,layout),
        mapping_resident_bytes=c['nodes']*8))
    return folder

def ensure_reference(c,layout,folder):
    label=key(c)+'__'+c['condition']+'__reference';out=ROOT/'analysis'/(label+'.json')
    dump=ROOT/'analysis'/label
    rows=invoke(query(c,layout,folder,out,1,[40,100,200],REFERENCE),label,out,
        dict(LAYOUT_ONE_PASS='1',LAYOUT_DUMP=str(dump),LAYOUT_TRACE=str(dump)+'.trace'))
    return rows,dump

def parity(c,layout,folder,ordering,label,binary=ENGINE):
    out=ROOT/'analysis'/(label+'-parity.json');dump=ROOT/'analysis'/(label+'-parity')
    e=order_env(c,ordering);e.update(LAYOUT_ONE_PASS='1',LAYOUT_DUMP=str(dump),LAYOUT_TRACE=str(dump)+'.trace')
    actual=invoke(query(c,layout,folder,out,1,[40,100,200],binary),label+'-parity',out,e)
    ref=ROOT/'analysis'/(key(c)+'__'+c['condition']+'__reference')
    expected=json.loads(Path(str(ref)+'.json').read_text())
    if len(actual)!=3 or len(expected)!=3:raise RuntimeError('Incomplete parity curve '+label)
    for a,b in zip(actual,expected):
        for metric in ['L','recall','pq_comparisons','expanded_nodes','reranked_nodes','inspected_edges','alpha_rejected_edges']:
            if abs(a[metric]-b[metric])>1e-7:raise RuntimeError('Logical parity '+metric+' '+label)
        for suffix in ['ids','dists']:
            if digest(str(dump)+'.L%d.%s'%(a['L'],suffix))!=digest(str(ref)+'.L%d.%s'%(a['L'],suffix)):
                raise RuntimeError('IDs/distances parity '+label)
    if digest(str(dump)+'.trace')!=digest(str(ref)+'.trace'):raise RuntimeError('Search trace changed '+label)
    save(ROOT/'analysis'/(label+'-parity-check.json'),dict(status='PASS',reference=str(ref),
        IDs_and_distances_byte_equal=True,search_trace_equal=True))

def levels(c,layout,folder):
    path=ROOT/'analysis'/(key(c)+'__'+c['condition']+'-levels.json')
    if path.exists():return json.loads(path.read_text())['L']
    label=key(c)+'__'+c['condition']+'-recall-screen';out=ROOT/'analysis'/(label+'.json')
    rows=invoke(query(c,layout,folder,out,1,[400]+EXTRA_LS),label,out,dict(LAYOUT_ONE_PASS='1'))
    stop=next((r['L'] for r in rows if r['recall']>=98),1600)
    ls=BASE_LS+[x for x in EXTRA_LS if x<=stop]
    save(path,dict(L=ls,highest_screen_recall=max(r['recall'] for r in rows),
        covered_targets=[t for t in [90,95,98] if max(r['recall'] for r in rows)>=t],
        rule='At most L=1600 in this batch; do not extrapolate missing recall targets.'))
    return ls

def check_points(rows,ls):
    if [r['L'] for r in rows]!=ls or any(r['seconds']<SECONDS or r['samples']%1000 for r in rows):raise RuntimeError('Incomplete/short timed curve')

def bracket(rows,target=90):
    for a,b in zip(rows,rows[1:]):
        if a['recall']<=target<=b['recall']:return sorted(set([a['L'],b['L']]))
    return [100]

def measure(c,layout,folder,ordering,rep,label,ls,diagnostic=True):
    e=order_env(c,ordering);threads=[1,32] if rep%2 else [32,1]
    for t in threads:
        out=ROOT/'results'/(label+'__main__t%d.json'%t)
        no=ROOT/'results'/(label+'__timingoff__t%d.json'%t)
        if rep%2==0:invoke(query(c,layout,folder,no,t,[100],PLAIN),no.stem,no,e,observe=True)
        rows=invoke(query(c,layout,folder,out,t,ls),out.stem,out,e,observe=True);check_points(rows,ls)
        if rep%2:invoke(query(c,layout,folder,no,t,[100],PLAIN),no.stem,no,e,observe=True)
        off=json.loads(no.read_text())[0];on=next(r for r in rows if r['L']==100)
        for metric in ['recall','pq_comparisons','expanded_nodes','reranked_nodes','submitted_pages']:
            if abs(off[metric]-on[metric])>1e-7:raise RuntimeError('Timer switch changes results')
        trace=ROOT/'analysis'/(key(c)+'__'+c['condition']+'__reference.trace')
        replay=ROOT/'results'/(label+'__replay__t%d.json'%t)
        invoke(args(c,layout,folder,'replay',replay)+[trace,t,SECONDS],replay.stem,replay,e,observe=True)
    if diagnostic:
        ref=json.loads((ROOT/'results'/(label+'__main__t1.json')).read_text());ls2=bracket(ref)
        for t in ([4,64] if rep%2 else [64,4]):
            out=ROOT/'results'/(label+'__concurrency__t%d.json'%t)
            rows=invoke(query(c,layout,folder,out,t,ls2),out.stem,out,e,observe=True);check_points(rows,ls2)
            for row in rows:
                r=next(x for x in ref if x['L']==row['L'])
                if row['recall']!=r['recall']:raise RuntimeError('Thread count changes recall')

def patches(c,layout,folder,ordering,rep,label):
    e=order_env(c,ordering)
    for pattern,mask in itertools.product(['uniform','hot'],[2,4,6,1]):
        modes=[0,1] if rep%2 else [1,0]
        for mode in modes:
            out=ROOT/'results'/(label+'__patch_{}_m{}_c{}.json'.format(pattern,mask,mode))
            env=dict(e,LAYOUT_PATCH_COALESCE=str(mode),LAYOUT_PATCH_BATCH='128')
            r=invoke(args(c,layout,folder,'patch',out)+[20261103+rep,10000,mask,pattern],out.stem,out,env,observe=True)
            if r['status']!='PASS' or r['modified_page_mismatches'] or r['restored_page_mismatches']:raise RuntimeError('Patch restore failure')

def single_controls(c,layout,folder,ordering,rep,label):
    if c['budget']!='full' or ordering!='original' or layout not in ['TXY','T_XY']:return
    trace=ROOT/'analysis'/(key(c)+'__default__reference.trace')
    alias='S_TD' if layout=='TXY' else 'S_T_D'
    for t in [1,32]:
        out=ROOT/'results'/(label+'__singleD_replay__t%d.json'%t)
        r=invoke(args(c,alias,folder,'replay',out)+[trace,t,SECONDS],out.stem,out,observe=True)
        dual=json.loads((ROOT/'results'/(label+'__replay__t%d.json'%t)).read_text())
        for metric in ['submitted_pages','submitted_bytes','logical_field_bytes']:
            if r[metric]!=dual[metric]:raise RuntimeError('Equal-byte single/dual control mismatch')

def smaller_controls(c,rep):
    if c['budget']!='full':return
    trace=ROOT/'analysis'/(key(c)+'__default__reference.trace')
    for layout in ['S_TX','S_T_X']:
        label=key(c)+'__original__'+layout+'__r%d__Sd'%rep
        marker=ROOT/'analysis'/(label+'-completed.json')
        if marker.exists():continue
        folder=build_layout(c,layout,'original',label)
        for t in [1,32]:
            out=ROOT/'results'/(label+'__replay__t%d.json'%t)
            invoke(args(c,layout,folder,'replay',out)+[trace,t,SECONDS],out.stem,out,
                   dict(LAYOUT_REPLAY_MASK='2'),observe=True)
        shutil.rmtree(str(folder));save(marker,dict(status='complete',scope='S-d capacity control on dual-space trace, not ANN query'))

def summarize():
    counts=collections.Counter();matched=[]
    for p in (ROOT/'results').glob('*.json'):
        if '.queries.' in p.name or '.partial' in p.name:continue
        if '__main__' in p.name:
            counts['main_runs']+=1;rows=json.loads(p.read_text());counts['main_points']+=len(rows)
            for target in [90,95,98]:
                for a,b in zip(rows,rows[1:]):
                    if a['recall']<=target<=b['recall']:
                        f=(target-a['recall'])/(b['recall']-a['recall']) if b['recall']!=a['recall'] else 0
                        r=dict(run=p.stem,target_recall=target)
                        for metric in ['L','qps','latency_us','submitted_pages','topology_submitted_pages','coordinate_submitted_pages','pq_comparisons','reranked_nodes']:
                            r[metric]=a[metric]+f*(b[metric]-a[metric])
                        matched.append(r);break
        elif '__patch_' in p.name:counts['patch_runs']+=1
        elif '__concurrency__' in p.name:counts['concurrency_runs']+=1
        elif '__timingoff__' in p.name:counts['timingoff_runs']+=1
        elif '__replay__' in p.name or '__singleD_replay__' in p.name:counts['replay_runs']+=1
    save(ROOT/'analysis/progress-summary.json',dict(counts=dict(counts),matched_recall=matched,
        updated_at=now(),scope='Controlled layout harness; static GT only; storage patches are not dynamic ANN updates.'))

def main(cs):
    provenance(cs);build()
    if not (ROOT/'analysis/layout-unit-tests.json').exists():raise RuntimeError('Missing unit validation')
    for source in cs:
        c=dict(source)
        c['source_hashes']={k:digest(c[k]) for k in ['meta','topo','coord']}
        save(ROOT/'analysis'/(c['name']+'-source-identity.json'),dict(hashes=c['source_hashes'],
             PQ_hashes={n:digest(Path(c['prefix'])/n) for n in ['emb_pq_pivots.bin','loc_pq_pivots.bin','emb_pq_compressed.bin','loc_pq_compressed.bin']}))
        for budget in ['full','2','0']:
            index=prepare_budget(c,budget)
            conditions=alpha_conditions(index)
            save(ROOT/'analysis'/(key(index)+'-query-identities.json'),[
                dict(condition=q['condition'],alpha_sha256=digest(q['alpha']),gt_sha256=digest(q['gt'])) for q in conditions])
            for rep in [1,2,3]:
                orders=['original','starling'] if rep%2 else ['starling','original']
                order_layouts=LAYOUTS[rep-1:]+LAYOUTS[:rep-1]
                for ordering in orders:
                    for layout in order_layouts:
                        label=key(index)+'__'+ordering+'__'+layout+'__r%d'%rep
                        marker=ROOT/'analysis'/(label+'-completed.json')
                        if marker.exists():continue
                        folder=build_layout(index,layout,ordering,label)
                        # Establish budget-specific references using the previous validated
                        # harness on identity files before the first reordered layout.
                        if rep==1 and ordering=='original':
                            for q in conditions:
                                ensure_reference(q,layout,folder);levels(q,layout,folder)
                        selected=conditions if ordering=='starling' and layout in ['TXY','T_XY'] else conditions[:1]
                        for q in selected:
                            case=label+'__'+q['condition']
                            parity(q,layout,folder,ordering,case)
                            if rep==1:parity(q,layout,folder,ordering,case+'-timingoff',PLAIN)
                            ls=json.loads((ROOT/'analysis'/(key(q)+'__'+q['condition']+'-levels.json')).read_text())['L']
                            measure(q,layout,folder,ordering,rep,case,ls,diagnostic=q['condition']=='default')
                            if q['condition']=='default':single_controls(q,layout,folder,ordering,rep,case)
                        patches(index,layout,folder,ordering,rep,label)
                        shutil.rmtree(str(folder))
                        save(marker,dict(status='complete',finished_at=now(),generated_files_removed=True,
                            query_conditions=[q['condition'] for q in selected]))
                        summarize()
                smaller_controls(index,rep)
        if any(digest(c[k])!=h for k,h in c['source_hashes'].items()):raise RuntimeError('Original index changed')
        save(ROOT/'analysis'/(c['name']+'-completed.json'),dict(status='complete',finished_at=now(),original_source_hashes_preserved=True))
    summarize()
