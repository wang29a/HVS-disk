"""Use the existing no-sidecar loader branch; never change graph or binary."""
from pathlib import Path
import json,shutil,struct,subprocess
import numpy as np
import common as c
import measure

OUT=c.RUN/'no-sidecar'
def matrix(path,dtype='<f4'):
    with path.open('rb') as f:n,d=struct.unpack('<II',f.read(8))
    if path.stat().st_size!=8+n*d*4:raise RuntimeError('matrix size')
    return np.memmap(str(path),dtype=dtype,mode='r',offset=8,shape=(n,d))
def gt(a):return (c.MG_RUN/'data' if a in (.3,.7) else c.D)/('gt-alpha-'+str(a)+'.bin')
def alpha(a):return (c.MG_RUN/'data' if a in (.3,.7) else c.D)/('query-alpha-'+str(a)+'.bin')
def jobs(path,values):path.write_text(''.join('{label} {method} {L} {kx} {ky} {weight}\n'.format(**v) for v in values))
def command(manifest,a,jp,out,seconds=0,trace='-',ids=None):
    return [c.MG_RUN/'bin/multigraph',manifest,c.D/'query_img_emb.fvecs',c.D/'query_text_emb.fvecs',alpha(a),gt(a),ids or c.MG_RUN/'data/report-ids.txt',out,'ssd',4,seconds,'pq',jp,trace,0]
def byte_equal(a,b,labels):
    for label in labels:
        for suffix in ('','.ids','.dists'):
            if Path(str(a)+'.'+label+suffix).read_bytes()!=Path(str(b)+'.'+label+suffix).read_bytes():raise RuntimeError('no-sidecar changed trajectory/results '+label+suffix)
def distances(points,a):
    x,y=(matrix(c.D/name) for name in ('base_img_emb.fvecs','base_text_emb.fvecs'))
    qx,qy=(matrix(c.D/name) for name in ('query_img_emb.fvecs','query_text_emb.fvecs'))
    af=float(np.float32(a));maximum=0.
    for p in points:
        for row in p['queries']:
            qi=row['query_id'];ids=row['ids'];dx=np.linalg.norm(x[ids].astype(np.float64)-qx[qi].astype(np.float64),axis=1);dy=np.linalg.norm(y[ids].astype(np.float64)-qy[qi].astype(np.float64),axis=1)
            ref=af*dx+(1-af)*dy;got=np.asarray(row['distances']);zero=ref==0
            if np.any(got[zero]!=0):raise RuntimeError('zero return distance changed')
            if np.any(~zero):maximum=max(maximum,float(np.max(np.abs(got[~zero]-ref[~zero])/ref[~zero])))
    if maximum>5e-6:raise RuntimeError('independent return distance mismatch')
    return maximum
def main():
    for name in ('diagnostic','timings','results'):(OUT/name).mkdir(parents=True,exist_ok=True)
    binary=c.MG_RUN/'bin/multigraph';build=c.load(c.MG_RUN/'validation/build.json')
    if c.sha(binary)!=build['binaries']['multigraph']:raise RuntimeError('original executor binary changed')
    original=c.MG_RUN/'manifest.txt';lines=original.read_text().splitlines();changed=0
    for i,line in enumerate(lines):
        fields=line.split()
        if fields and fields[0]=='u':
            if len(fields)!=5:raise RuntimeError('manifest entry shape')
            fields[3]='-';fields[4]='0';lines[i]=' '.join(fields);changed+=1
    if changed!=1:raise RuntimeError('unified manifest entry missing/duplicate')
    manifest=OUT/'manifest-no-sidecar.txt';manifest.write_text('\n'.join(lines)+'\n')
    frozen=c.load(c.MG_RUN/'frozen.json');prior=c.load(c.MG_RUN/'report/summary.json')['records'];plans={}
    for a in (.1,.3,.5,.7,.9):
        plans[str(a)]={}
        for case,method in [('loaded','DEG_off'),('no-sidecar','DEG_off'),('specialized','Or_ideal')]:
            row=next(v for v in prior if v['medium']=='ssd' and v['rule']=='weighted_pages' and v['method']==method and v['target']==.95 and v['alpha']==a)
            labels=list(dict.fromkeys(row['repeats'][0]['labels']));curve=frozen['curves'][str(a)+'-'+method]
            plans[str(a)][case]=dict(jobs=[curve['jobs'][name] for name in labels],labels=row['repeats'][0]['labels'],weight=row['repeats'][0]['interpolation_weight'])
    identity=dict(binary_sha256=c.sha(binary),original_manifest_sha256=c.sha(original),derived_manifest_sha256=c.sha(manifest),frozen_sha256=c.sha(c.MG_RUN/'frozen.json'),plans=plans)
    stamp=OUT/'plan.json'
    if stamp.exists() and c.load(stamp)!=identity:raise RuntimeError('no-sidecar test inputs changed')
    c.save(stamp,identity)
    strace=shutil.which('strace');trace_ok=False;max_error=0.
    if strace:
        probe=subprocess.run([strace,'-o',str(OUT/'strace-probe.log'),'true'],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        if probe.returncode:strace=None
    for a in (.1,.3,.5,.7,.9):
        prefixes={}
        for case in ('loaded','no-sidecar'):
            info=plans[str(a)][case];jp=OUT/'diagnostic'/(str(a)+'-'+case+'.jobs');jobs(jp,info['jobs']);out=OUT/'diagnostic'/(str(a)+'-'+case+'.json');prefix=OUT/'diagnostic'/(str(a)+'-'+case)
            args=command(manifest if case=='no-sidecar' else original,a,jp,out,trace=prefix)
            if strace and case=='no-sidecar' and a==.1:args=[strace,'-f','-e','trace=open,openat','-o',OUT/'no-sidecar-open.log']+args
            c.run(args,'no-sidecar-check-'+str(a)+'-'+case,cpus='80-83',timeout=600)
            points=c.load(out);max_error=max(max_error,distances(points,a));prefixes[case]=prefix
            if case=='no-sidecar' and any(p['sidecar_bytes']!=0 for p in points):raise RuntimeError('sidecar still allocated')
        byte_equal(prefixes['loaded'],prefixes['no-sidecar'],[v['label'] for v in plans[str(a)]['loaded']['jobs']])
    if strace:
        if 'ranges.bin' in (OUT/'no-sidecar-open.log').read_text():raise RuntimeError('no-sidecar path opened range file')
        trace_ok=True
    c.save(OUT/'correctness.json',dict(status='PASS',all_500_report_queries_byte_equal=True,weights=[.1,.3,.5,.7,.9],sidecar_bytes=0,strace_no_range_open=trace_ok,strace_available=bool(strace),max_return_relative_error=max_error))
    choice=measure.quiet();cpus=choice['choice']['cpus'];c.save(OUT/'affinity.json',choice);summary=[];pending=[]
    for i,a in enumerate((.1,.3,.5,.7,.9)):
        cases=['loaded','no-sidecar','specialized'];cases=cases[i%3:]+cases[:i%3]
        for case in cases:
            info=plans[str(a)][case];jp=OUT/'timings'/(str(a)+'-'+case+'.jobs');jobs(jp,info['jobs']);out=OUT/'results'/(str(a)+'-'+case+'.json');saved=OUT/'timings'/(str(a)+'-'+case+'.json')
            if saved.exists():data=c.load(saved)
            else:
                data=measure.run(command(manifest if case=='no-sidecar' else original,a,jp,out,seconds=10), 'no-sidecar-time-'+str(a)+'-'+case,out,cpus)
                if data is not None:c.save(saved,data)
            if data is None:pending.append([a,case]);continue
            by={p['label']:p for p in data['points']};v1,v2=[by[name] for name in info['labels']];w=info['weight']
            row=dict(alpha=a,case=case,latency_us=(1-w)*v1['latency_us']+w*v2['latency_us'],qps=(1-w)*v1['qps']+w*v2['qps'],
                peak_rss_kib=max(p['peak_rss_kib'] for p in data['points']),sidecar_bytes=max(p['sidecar_bytes'] for p in data['points']),
                recall_bracket=[v1['recall'],v2['recall']],labels=info['labels'],weight=w,raw_p95_us={p['label']:p['p95_us'] for p in data['points']})
            summary.append(row)
    for a in (.1,.3,.5,.7,.9):
        found={v['case']:v for v in summary if v['alpha']==a}
        if all(k in found for k in ('loaded','no-sidecar','specialized')):
            n=found['no-sidecar'];n['RSS_over_specialized']=n['peak_rss_kib']/found['specialized']['peak_rss_kib'];n['memory_comparable']=.90<=n['RSS_over_specialized']<=1.10
            n['latency_change_vs_loaded_percent']=100*(n['latency_us']/found['loaded']['latency_us']-1)
    c.save(OUT/'summary.json',dict(results=summary,pending=pending,repetitions=1,selection='unchanged prior weighted-page-cost 95% brackets; no retuning',scope='single-pass confirmation, not a three-repeat performance claim'))
    text='# 不加载区间旁表的统一图\n\n复用同一二进制、拓扑、PQ和坐标，全部500条报告查询的路径／结果字节相同；输出旁表字节数为0。正式仅一轮，使用此前冻结的95%夹点，不重新调参。\n\n| α | 加载版ms | 不加载版ms | 专用图ms | 不加载版RSS MiB | 专用图RSS MiB | RSS比值 |\n|---:|---:|---:|---:|---:|---:|---:|\n'
    for a in (.1,.3,.5,.7,.9):
        f={v['case']:v for v in summary if v['alpha']==a}
        if all(k in f for k in ('loaded','no-sidecar','specialized')):
            l,n,s=(f[k] for k in ('loaded','no-sidecar','specialized'));text+='| %.1f | %.4f | %.4f | %.4f | %.2f | %.2f | %.3f |\n'%(a,l['latency_us']/1000,n['latency_us']/1000,s['latency_us']/1000,n['peak_rss_kib']/1024,s['peak_rss_kib']/1024,n['RSS_over_specialized'])
    text+='\nRSS相当阈值事前设为±10%；时间只有一次确认测量，不把细小波动包装为加速。峰值RSS来自独立进程，精排深度和线程暂存也计入；不是仅用减去旁表字节数作估计。p95只列原始点，不插值。\n'
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8');return dict(complete=not pending,summary=str(OUT/'summary.json'))
