"""Build the explicitly authorized OpenImages B, preserving historical A."""
from pathlib import Path
import csv
import fcntl
import json
import os
import shutil
import struct
import subprocess
import time
import run_g0 as c
import run_revision3_diagnostics as d

E=c.HOST_ROOT/'experiments'
AB=E/'backlink-ab-20261115/runtime'
U=E/'u-shape-20261113'
A=U/'review-20261113/graphs/openimage'
PA=E/'edbt-evidence-20261115/runtime/openimages/packed/u'
DATA=Path('/mnt/nvme3/wz/dataset/Openimage_disk')
OUTPUT=Path('/mnt/nvme2/wz/hvs-update-basic-2026-11-16/openimages-B')


def status(stage,**extra):
    c.save('status.json',dict(stage=stage,updated_unix=time.time(),**extra))


def run(args,label,threads=2,timeout=300):
    args=list(map(str,args))
    cpus='72-79' if threads==8 else '76,77'
    command=['nice','-n','15','ionice','-c','2','-n','7','numactl',
             '--physcpubind='+cpus,'--membind=3']+args
    env={k:v for k,v in c.ENV.items() if not k.startswith(('HVS_','LAYOUT_','PTR_','METRIC_'))}
    env.update(OMP_NUM_THREADS=str(threads),OMP_THREAD_LIMIT=str(threads),
               OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',
               HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1')
    started=time.time()
    with (c.RUN/(label+'.log')).open('w') as log:
        child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        samples=[]
        try:
            while child.poll() is None:
                if time.time()-started>timeout:
                    child.terminate()
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:child.kill();child.wait()
                    raise RuntimeError(label+' timeout')
                sample=dict(unix=time.time(),pid=child.pid)
                try:
                    lines=Path('/proc/'+str(child.pid)+'/status').read_text().splitlines()
                    sample.update({v.split(':',1)[0]:v.split(':',1)[1].strip() for v in lines
                                   if v.startswith(('Cpus_allowed_list:','Mems_allowed_list:','VmRSS:','Threads:'))})
                    sample['anonymous_numa_pages']={}
                    for line in Path('/proc/'+str(child.pid)+'/numa_maps').read_text().splitlines():
                        if 'anon=' in line:
                            for cell in line.split():
                                if cell.startswith('N') and '=' in cell:
                                    node,value=cell.split('=')
                                    sample['anonymous_numa_pages'][node]=sample['anonymous_numa_pages'].get(node,0)+int(value)
                except (OSError,ValueError):pass
                samples.append(sample)
                c.save(label+'-telemetry.json',samples)
                status(label,active_pid=child.pid,elapsed_s=time.time()-started,
                       progress_log=str(c.RUN/(label+'.log')))
                time.sleep(5)
        finally:
            if child.poll() is None:
                child.terminate();child.wait(timeout=15)
    record=dict(command=command,returncode=child.returncode,wall_s=time.time()-started,
                environment={k:env[k] for k in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','HVS_SOURCE_ONLY','HVS_FULL_ONLY','LD_LIBRARY_PATH')})
    c.save(label+'.json',record)
    if child.returncode:raise RuntimeError(label+' failed '+str(child.returncode))
    return record


def main():
    c.RUN=c.ROOT/'runtime/revision3-openimages-ab'
    c.RUN.mkdir(parents=True,exist_ok=True)
    lock=(c.HOST_ROOT/'experiments/budget-reorder-20261103/queue.lock').open('a+')
    status('waiting_global_lock')
    fcntl.flock(lock,fcntl.LOCK_EX)
    try:
        status('verifying_identities')
        proof=json.loads((AB/'validation/build.json').read_text())
        semantic=json.loads((U/'stage2-v2/validation/semantic-proof.json').read_text())
        a_identity=json.loads((A/'identity.json').read_text())
        packed_a_identity=json.loads((PA/'identity.json').read_text())
        b_binary=AB/'bin/build-B';b_library=AB/'build/B/libpipeann.a';b_source=AB/'build/hybrid_index.cpp'
        assert c.sha(b_binary)==proof['binaries']['B']['binary']
        assert c.sha(b_library)==proof['binaries']['B']['library']
        assert c.sha(b_source)==proof['variant']
        assert c.sha(A/'float-graph.bin')==a_identity['graph_sha256']==packed_a_identity['identity']['source']
        assert c.sha(PA/'compact.topo')==packed_a_identity['sha256']
        expected=semantic['binaries']['unified']
        assert c.sha(U/'stage2-v2/bin/build-unified')==expected['builder_sha256']==a_identity['builder_sha256']
        assert c.sha(U/'stage2-v2/bin/libpipeann-unified.a')==expected['library_sha256']
        assert c.sha(U/'stage2-v2/variant/unified/hybrid_index.cpp')==expected['source_sha256']
        files=[DATA/'base_img_emb.fvecs',DATA/'base_text_emb.fvecs']
        for p in files:
            with p.open('rb') as f:assert struct.unpack('<II',f.read(8))==(456699,768)
        input_hashes={str(p):c.sha(p) for p in files}
        parameters=dict(nodes=456699,dimensions=[768,768],R=40,L=200,C=750,
            normalization='none',threads=8,numa_node=3,cpus='72-79',policy='B: always reverse reprune')
        identity=dict(parameters=parameters,inputs=input_hashes,
            B=dict(binary=str(b_binary),binary_sha256=c.sha(b_binary),library=str(b_library),
                   library_sha256=c.sha(b_library),source=str(b_source),source_sha256=c.sha(b_source),
                   formal_source_sha256=proof['formal_source'],flags=proof['binaries']['B']['flags'],
                   link=proof['link'],wrapper_sha256=c.sha(AB/'build/build_graph.cpp')),
            A=dict(scope='historical A semantics, not a newly rebuilt counted A1',graph=str(A/'float-graph.bin'),
                   graph_sha256=a_identity['graph_sha256'],identity=c.sha(A/'identity.json'),
                   build_command=c.sha(U/'review-20261113/analysis/openimage-build-command.json'),
                   semantic_proof=c.sha(U/'stage2-v2/validation/semantic-proof.json')),
            storage=dict(physical_B_directory=str(OUTPUT),reason='nvme3 has about 2 GiB free; use a new owned nvme2 directory and a link in this experiment'))
        prebuild_path=c.RUN/'prebuild-identity.json'
        previous_identity=json.loads(prebuild_path.read_text()) if prebuild_path.exists() else None
        completed_build_path=c.RUN/'build-openimages-B.json'
        resume=OUTPUT.exists() and any(OUTPUT.iterdir())
        if resume:
            assert previous_identity==identity, 'existing B build identity differs'
            assert completed_build_path.exists() and json.loads(completed_build_path.read_text())['returncode']==0, 'existing B build not completed'
        else:
            c.save('prebuild-identity.json',identity)
        OUTPUT.parent.mkdir(parents=True,exist_ok=True)
        OUTPUT.mkdir(exist_ok=True)
        if shutil.disk_usage(str(OUTPUT)).free<2*2**30:
            raise RuntimeError('insufficient space for new B artifacts plus margin')
        link=c.RUN/'openimages-B'
        if not link.exists():link.symlink_to(OUTPUT,target_is_directory=True)
        else:assert link.resolve()==OUTPUT
        if resume:
            record=json.loads(completed_build_path.read_text())
        else:
            record=run([b_binary,'dual',files[0],files[1],456699,768,768,8,OUTPUT],
                       'build-openimages-B',threads=8,timeout=7200)
        stats=json.loads((OUTPUT/'reverse-insert-stats.json').read_text())
        assert stats['reverse_always_prune']==1 and stats['reverse_targets']==stats['B_reprunes'], 'B did not reprune every reverse target'
        assert all(c.sha(p)==input_hashes[str(p)] for p in files), 'input vectors changed during build'
        assert c.sha(A/'float-graph.bin')==a_identity['graph_sha256'],'A graph changed'
        packed=OUTPUT/'packed';packed.mkdir(exist_ok=True)
        packer=E/'multigraph-20261113/runtime/bin/pack'
        assert c.sha(packer)==packed_a_identity['identity']['packer']
        run([packer,OUTPUT/'full.meta',OUTPUT/'float-graph.bin',packed/'compact.meta',
             packed/'compact.topo',packed/'ranges.bin',OUTPUT/'full.meta'],'pack-openimages-B')
        auditor=AB/'function-v2/bin/graph-audit'
        run([auditor,OUTPUT/'float-graph.bin',packed/'compact.meta',packed/'compact.topo',
             packed/'ranges.bin'],'validate-openimages-B',timeout=300)
        final=dict(identity,build=record,reverse_counts=stats,
            graph_audit=json.loads((OUTPUT/'graph-audit.json').read_text()),
            binary_identity_unchanged=c.sha(b_binary)==identity['B']['binary_sha256'],
            outputs={str(p.relative_to(OUTPUT)):c.sha(p) for p in OUTPUT.rglob('*') if p.is_file()})
        c.save('build-identity.json',final)
        tool=c.ROOT/'runtime/revision3-diagnostics/connectivity_audit'
        rows=[];targets=set();sets={}
        for name,meta,graph in [('historical-A',A/'full.meta',PA/'compact.topo'),
                                ('new-B',OUTPUT/'full.meta',packed/'compact.topo')]:
            prefix=c.RUN/name
            run([tool,meta,graph,prefix],'connectivity-'+name)
            row=json.loads((c.RUN/('connectivity-'+name+'.log')).read_text())
            low={int(v['id']) for v in csv.DictReader(Path(str(prefix)+'-low-degree.csv').open())}
            unreachable={int(v) for v in Path(str(prefix)+'-unreachable.txt').read_text().split()}
            sets[name]=(low,unreachable);targets.update(low|unreachable)
            row.update(graph=name,entry_policy='own metadata entries',meta_sha256=c.sha(meta),topology_sha256=c.sha(graph))
            rows.append(row)
        status('openimages_radius_diagnostics',targets=len(targets))
        flags=d.radius_matches(d.matrix(files[0]),d.matrix(files[1]),targets)
        for row in rows:
            low,unreachable=sets[row['graph']]
            row.update(low_degree_exact_zero_both=sum(flags[v]['exact_zero_both'] for v in low),
                       low_degree_lt_1e_6_both=sum(flags[v]['lt_1e_6_both'] for v in low),
                       unreachable_exact_zero_both=sum(flags[v]['exact_zero_both'] for v in unreachable),
                       unreachable_lt_1e_6_both=sum(flags[v]['lt_1e_6_both'] for v in unreachable))
        result=dict(status='OPENIMAGES_A_HISTORICAL_B_NEW_DIAGNOSTICS_COMPLETE',graphs=rows,
            comparison_scope='one historical A and one newly counted B, same 456699-row input; structural/duplicate diagnostics only, not formal repeated A/B performance',
            inputs=input_hashes,graph_repair_applied=False,pruning_changes_applied=False)
        c.save('diagnostic-results.json',result)
        parent=c.ROOT/'runtime/revision3-diagnostics/diagnostic-results.json'
        old=json.loads(parent.read_text());old['OpenImages']=result
        old['status']='REQUESTED_DIAGNOSTICS_COMPLETE'
        parent.write_text(json.dumps(old,indent=2)+'\n')
        (parent.parent/'status.json').write_text(json.dumps(dict(
            stage='REQUESTED_DIAGNOSTICS_COMPLETE',updated_unix=time.time(),
            OpenImages=result['status']),indent=2)+'\n')
        status(result['status'])
    except Exception as e:
        status('OPENIMAGES_DIAGNOSTIC_STOPPED',reason=str(e))
        raise


if __name__=='__main__':main()
