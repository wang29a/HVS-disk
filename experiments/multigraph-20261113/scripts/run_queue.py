"""One serial Laion-only queue. Every correctness failure stops before timing."""
import fcntl
import json
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import common as c
import prepare_inputs as inp
import protocol as p
import validate as v
sys.path.insert(0,str(c.E/'metric-consistent-20261111/scripts'))
import monitor as m

SOURCE={'u':c.U/'stage2-v2/graphs/unified','a05':c.U/'stage2-v2/graphs/fixed-0.5'}
ALPHAS={'a01':.1,'a03':.3,'a05':.5,'a07':.7,'a09':.9}

def build(name):
    if name in SOURCE:return SOURCE[name]
    folder=c.RUN/'graphs'/name;folder.mkdir(parents=True,exist_ok=True);stamp=folder/'build-identity.json'
    binary=c.RUN/'bin/build-vamana' if name in ('x','y') else c.U/'stage2-v2/bin/build-fixed'
    if stamp.exists():
        data=c.load(stamp)
        if data['builder_sha256']!=c.sha(binary) or data['graph_sha256']!=c.sha(folder/'float-graph.bin'):raise RuntimeError('built graph identity changed '+name)
        return folder
    c.capacity(5*2**30,'multigraph build '+name)
    env=dict(OMP_NUM_THREADS='8',OMP_THREAD_LIMIT='8',HVS_SOURCE_ONLY='1',HVS_FULL_ONLY='1')
    if name in ALPHAS:env['HVS_FIXED_ALPHA']=str(ALPHAS[name])
    base=c.D/('base_text_emb.fvecs' if name=='y' else 'base_img_emb.fvecs')
    start=time.time();c.run([binary,'single' if name in ('x','y') else 'dual',base,c.D/'base_text_emb.fvecs',2000000,512,512,8,folder],'build-'+name,cpus='72-79',timeout=6*3600,extra=env)
    audit=c.load(folder/'graph-audit.json')
    if audit['nodes']!=2000000 or audit['quantized_lost_grid_points']!=0:raise RuntimeError('graph audit '+name)
    if name in ('x','y') and not (folder/'full.meta').exists():
        (folder/'full.meta').symlink_to('compact.meta')
    c.save(stamp,dict(builder_sha256=c.sha(binary),graph_sha256=c.sha(folder/'float-graph.bin'),wall_seconds=time.time()-start,audit=audit,implementation='native Vamana squared L2, saturation=true' if name in ('x','y') else 'v2 fixed weighted Euclidean LE'))
    return folder

def pack(name,source):
    folder=c.RUN/'packed'/name;folder.mkdir(parents=True,exist_ok=True);stamp=folder/'identity.json'
    identity={k:c.sha(source/k) for k in ('float-graph.bin','full.meta')}
    identity['common_entry_meta_sha256']=c.sha(SOURCE['u']/'full.meta');identity['packer_sha256']=c.sha(c.RUN/'bin/pack')
    if stamp.exists():
        old=c.load(stamp)
        if old['source']!=identity:raise RuntimeError('packed source identity changed '+name)
        for k,h in old['files'].items():
            if c.sha(folder/k)!=h:raise RuntimeError('packed file changed '+name)
        return old
    slots=struct.unpack('<I',(source/'full.meta').read_bytes()[16:20])[0] if name=='u' else 0
    side=folder/'ranges.bin' if name=='u' else '-'
    log=c.run([c.RUN/'bin/pack',source/'full.meta',source/'float-graph.bin',folder/'compact.meta',folder/'compact.topo',side,SOURCE['u']/'full.meta'],'pack-'+name,cpus='72,73',timeout=600)
    stats=json.loads(log.read_text());files={f:c.sha(folder/f) for f in (['compact.meta','compact.topo','ranges.bin'] if name=='u' else ['compact.meta','compact.topo'])}
    if name in SOURCE:
        buildmeta=c.load(source/'identity.json');seconds=buildmeta['wall_seconds']
    else:seconds=c.load(source/'build-identity.json')['wall_seconds']
    data=dict(source=identity,files=files,stats=stats,range_slots=slots,build_seconds=seconds,
        topology_bytes=(folder/'compact.topo').stat().st_size+(folder/'compact.meta').stat().st_size,
        topology_data_bytes=(folder/'compact.topo').stat().st_size,metadata_bytes=(folder/'compact.meta').stat().st_size,
        sidecar_bytes=(folder/'ranges.bin').stat().st_size if name=='u' else 0,
        entry_policy='common unified entries; native source entries retained in immutable source meta')
    c.save(stamp,data);return data

def manifest(names):
    lines=['HVS_MULTI_V1 2000000 512 512 '+str(len(names)),str(c.RUN/'data/xy.pages'),str(c.RUN/'data')+'/']
    for name in names:
        f=c.RUN/'packed'/name;d=c.load(f/'identity.json')
        lines.append(' '.join(map(str,[name,f/'compact.meta',f/'compact.topo',f/'ranges.bin' if name=='u' else '-',d['range_slots']])))
    (c.RUN/'manifest.txt').write_text('\n'.join(lines)+'\n')

def verify_sources(plan_name=None):
    frozen=c.load(c.ROOT/'FREEZE.json');data=(c.ROOT/'README.md').read_bytes()
    import hashlib
    if hashlib.sha256(data[:frozen['bytes']]).hexdigest()!=frozen['sha256']:raise RuntimeError('frozen README prefix overwritten')
    proof=c.load(c.RUN/'validation/build.json')
    for name,h in proof['binaries'].items():
        if c.sha(c.RUN/'bin'/name)!=h:raise RuntimeError('compiled binary changed '+name)
    for name,h in proof['sources'].items():
        if c.sha(c.ROOT/'scripts'/name)!=h:raise RuntimeError('C++ source differs from compiled version '+name)
    if c.load(c.RUN/'validation/tiny-gates.json')['status']!='PASS':raise RuntimeError('tiny gate not passed')
    plan=dict(protocol_prefix=frozen,readme_sha256=c.sha(c.ROOT/'README.md'),binary_sha256=c.sha(c.RUN/'bin/multigraph'),scripts={s.name:c.sha(s) for s in (c.ROOT/'scripts').iterdir() if s.suffix in ('.py','.cpp')},levels=p.LEVELS,budgets=p.BUDGETS,weights=inp.WEIGHTS,stage='Laion-only; no OpenImages execution')
    path=c.RUN/'plan.json'
    if plan_name is not None:
        if plan_name!='three-layer-20261114':raise RuntimeError('unknown amendment plan')
        plan['parent_plan_sha256']=c.sha(path)
        plan['amendment_sha256']=c.sha(c.ROOT/'AMENDMENT-20261114.json')
        path=c.RUN/'plans'/(plan_name+'.json')
    if path.exists() and c.load(path)!=plan:raise RuntimeError('queue implementation changed after launch; new namespace/amendment required')
    c.save(path,plan)

def wait_predecessors():
    """Do not overtake the three already queued review jobs at an unfair flock."""
    directory=c.U/'review-20261113'
    predecessors=[(directory/'status.json','run_priority_review.py'),
        (directory/'explanation-status.json','run_centre_explanation.py'),
        (directory/'cpu-recheck/status.json','run_review_recheck.py')]
    deadline=time.monotonic()+24*3600
    while True:
        live=[]
        for path,script in predecessors:
            if not path.exists():continue
            pid=c.load(path).get('pid')
            if not pid:continue
            try:cmd=(Path('/proc')/str(pid)/'cmdline').read_bytes().replace(b'\0',b' ').decode()
            except OSError:continue
            if script in cmd:live.append(dict(status=str(path),pid=pid,script=script))
        if not live:return
        c.status('waiting_predecessors',jobs=live)
        if time.monotonic()>deadline:raise RuntimeError('predecessor wait >24h; deferred, no overtaking')
        time.sleep(10)

def tune():
    curves={};contribution=c.load(c.RUN/'data/contribution.json')
    for a in inp.WEIGHTS:
        for method in p.METHODS:
            label=str(a)+'-'+method;folder=c.RUN/'tune';folder.mkdir(exist_ok=True);out=folder/(label+'.json');jobs=p.jobs(method,a,contribution[str(a)]['tune']['mean']);jp=folder/(label+'.jobs');v.write_jobs(jp,jobs)
            if not out.exists():c.run(v.command(c.RUN/'manifest.txt',inp.alphapath(a),inp.gtpath(a),c.RUN/'data/tune-ids.txt',out,'dram',jp), 'tune-'+label,timeout=3600)
            points=c.load(out)
            if [x['label'] for x in points]!=[x['label'] for x in jobs]:raise RuntimeError('training grid changed')
            curves[label]=dict(alpha=a,method=method,jobs={x['label']:x for x in jobs},frontiers=p.frontiers(points),training_sha256=c.sha(out))
    c.save(c.RUN/'frozen.json',dict(selection='training workload only; report timing not inspected',curves=curves,manifest_sha256=c.sha(c.RUN/'manifest.txt'),gate_sha256=c.sha(c.RUN/'validation/real-gates.json')))
    return curves

def timed(args,label,out):
    env=dict(c.ENV);cmd=['nice','-n','10','ionice','-c','2','-n','7','numactl','--all','--physcpubind='+','.join(map(str,m.CPUS)),'--membind=3']+list(map(str,args))
    c.save(c.RUN/'commands'/(label+'.json'),dict(command=cmd,start=time.time(),timeout=240));c.status('timing',task=label)
    with (c.RUN/'logs'/(label+'.log')).open('w') as log:
        c.CHILD=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);samples=[];begin=time.monotonic()
        while c.CHILD.poll() is None:
            if time.monotonic()-begin>240:c.stop();return dict(accepted=False,reason='bounded measurement timeout')
            samples.append(m.sample(c.CHILD.pid));time.sleep(.5)
        code=c.CHILD.returncode;c.CHILD=None
    if code:raise RuntimeError('timed executable error '+label)
    point=c.load(out)[0]
    guard=m.validate(samples,dict(point_start_unix=point['start_unix'],measured_seconds=point['seconds']))
    return guard

def measure(curves):
    started=time.monotonic()
    while True:
        choice=m.choose()
        if choice['choice']['external_busy_fraction']<.10:break
        if time.monotonic()-started>180:raise RuntimeError('no quiet NUMA3 block after bounded wait; deferred')
        time.sleep(5)
    c.save(c.RUN/'affinity.json',choice);pending=[];consecutive=0
    names=sorted(curves)
    for rep in range(3):
        for label in names[rep:]+names[:rep]:
            info=curves[label];a=info['alpha'];chosen=sorted(set(x for front in info['frontiers'].values() for x in front))
            for medium in ('ssd','topo'):
                for name in chosen:
                    key=label+'-'+name+'-'+medium+'-rep'+str(rep+1);accepted=c.RUN/'timings'/(key+'.json')
                    if accepted.exists():continue
                    jp=c.RUN/'jobs'/(key+'.txt');v.write_jobs(jp,[info['jobs'][name]])
                    for attempt in range(1,4):
                        tag=key+'-attempt'+str(attempt);out=c.RUN/'results'/(tag+'.json');args=v.command(c.RUN/'manifest.txt',inp.alphapath(a),inp.gtpath(a),c.RUN/'data/report-ids.txt',out,medium,jp,seconds=10)
                        guard=timed(args,tag,out);c.save(c.RUN/'guards'/(tag+'.json'),guard)
                        if guard['accepted']:
                            c.save(accepted,dict(point=c.load(out)[0],guard=guard,raw=str(out)));consecutive=0;break
                        time.sleep(5)
                    else:
                        pending.append(key);consecutive+=1;c.save(c.RUN/'pending.json',pending)
                        if consecutive>=3:raise RuntimeError('three consecutive timing points rejected; deferred, no busy retry loop')
    c.save(c.RUN/'pending.json',pending)

def main():
    for sub in ('graphs','packed','logs','commands','validation','data','timings','results','jobs','guards'):(c.RUN/sub).mkdir(parents=True,exist_ok=True)
    own=(c.RUN/'coordinator.lock').open('a+');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    verify_sources();wait_predecessors();queue_lock=c.lock()
    c.run([sys.executable,c.ROOT/'scripts/prepare_inputs.py','shared'],'shared-inputs',cpus='72,73',timeout=3600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    for name in ('u','a05'):pack(name,SOURCE[name])
    manifest(['u','a05'])
    # Fail cheaply on existing-centre compatibility before spending hours on GT
    # and extra graphs. This is a gate, not a tuning or timed measurement.
    selected=c.load(c.D/'identity.json')['query_cohorts']['tune'][:8]+c.load(c.D/'identity.json')['query_cohorts']['report'][:8]
    (c.RUN/'data/gate-ids.txt').write_text('\n'.join(map(str,sorted(selected)))+'\n')
    c.save(c.RUN/'validation/early-centre.json',v.centre(c.RUN/'manifest.txt'))
    c.run([sys.executable,c.ROOT/'scripts/prepare_inputs.py','gt'],'missing-gt-and-oracle',cpus='72,73',timeout=12*3600,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'))
    for name in ('x','y'):pack(name,build(name))
    manifest(['x','y','a05','u']);v.coverage()
    for name in ('a01','a03','a07','a09'):pack(name,build(name))
    manifest(['x','y','a01','a03','a05','a07','a09','u'])
    v.real()
    if c.load(c.RUN/'validation/real-gates.json')['status']!='PASS':raise RuntimeError('all real gates required before timing')
    curves=tune();measure(curves)
    c.run([sys.executable,c.ROOT/'scripts/report.py'],'report-laion',cpus='72,73',timeout=300,extra=dict(OMP_NUM_THREADS='2',OMP_THREAD_LIMIT='2'))
    c.status('awaiting_user_review',dataset='Laion-2M',next='stop; OpenImages requires user confirmation',pending=c.load(c.RUN/'pending.json'))
if __name__=='__main__':
    signal.signal(signal.SIGTERM,c.stop);signal.signal(signal.SIGINT,c.stop)
    try:main()
    except Exception as error:c.stop();c.status('stopped',error=str(error),action='inspect failure; no automatic parameter changes or dataset expansion');raise
