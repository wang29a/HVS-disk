"""Complete the five missing points without overwriting historical runs."""
from pathlib import Path
import statistics
import common as c
import measure

OLD=c.U/'review-20261113/cpu-recheck'
OUT=c.RUN/'openimage'
METRICS=['latency_us','qps','submitted_pages','topology_submitted_pages','coordinate_submitted_pages','topology_submit_wait_us','coordinate_submit_wait_us','expanded_nodes','pq_space_evals']
def config(label,L,r,rep):return label+'-L'+str(L)+'-r'+str(r)+'-rep'+str(rep)
def command(label,L,r,out):
    _,version,bucket,filtering=label.split('-');s=OLD/'ssd'/version;w=OLD/'ssd'/bucket/'report'
    args=[c.E/'metric-consistent-20261111/bin/metric-search','query',s/'query.meta',(s/'group-1.bin').resolve(),(s/'group-6.bin').resolve(),s,'T_XY',out,
        '/mnt/nvme3/wz/hvs-disk-paper-20260920/experiments/formal-32/openimage/full/',w/'qx.bin',w/'qy.bin',w/'alpha.bin',w/'gt.bin',4,10,L]
    env=dict(METRIC_RERANK=str(r))
    if filtering=='f0':env['LAYOUT_BYPASS_ALPHA']='1'
    return args,env
def one(root,label,L,r,rep,cpus,reuse=None):
    key=config(label,L,r,rep);accepted=root/'accepted'/(key+'.json');accepted.parent.mkdir(parents=True,exist_ok=True)
    if accepted.exists():return c.load(accepted)
    if reuse is not None:
        src=reuse/'timings'/(key+'-accepted.json')
        if src.exists():accepted.symlink_to(src);return c.load(src)
    out=root/'results'/(key+'.json');out.parent.mkdir(exist_ok=True)
    args,env=command(label,L,r,out);result=measure.run(args,root.name+'-'+key,out,cpus,extra=env)
    if result is None:return None
    saved=dict(point=result['points'][0],guard=result['guards'][0],result=result['result'],new_measurement=True)
    c.save(accepted,saved);return saved
def summarize(root,plans):
    result={};pending=[]
    for label,rules in plans.items():
        result[label]={}
        for rule,p in rules.items():
            repeated=[]
            if not p['covered']:result[label][rule]=dict(complete=False,covered=False);continue
            for rep in range(1,4):
                points=[]
                for L,r in p['bracket']:
                    f=root/'accepted'/(config(label,L,r,rep)+'.json')
                    if not f.exists():pending.append(config(label,L,r,rep));break
                    points.append(c.load(f)['point'])
                if len(points)==2:
                    a,b=points;w=p['weight'];repeated.append({k:(1-w)*a[k]+w*b[k] for k in METRICS})
            result[label][rule]=dict(complete=len(repeated)==3,repeats=repeated,mean={k:statistics.mean(v[k] for v in repeated) for k in METRICS} if len(repeated)==3 else {},bracket=p['bracket'])
    c.save(root/'summary.json',dict(results=result,pending=sorted(set(pending)),scope='95% mixed-alpha buckets; complete repetitions only'))
    return result,sorted(set(pending))
def run_set(root,plans,cpus,reuse=None):
    for name in ('results','accepted'): (root/name).mkdir(parents=True,exist_ok=True)
    for rep in range(1,4):
        labels=list(plans);labels=labels[rep-1:]+labels[:rep-1]
        for label in labels:
            pairs=sorted(set(tuple(v) for p in plans[label].values() if p['covered'] for v in p['bracket']))
            for L,r in pairs:one(root,label,L,r,rep,cpus,reuse)
    return summarize(root,plans)
def main():
    OUT.mkdir(exist_ok=True);plans=c.load(OLD/'ssd/report-brackets.json');cpus=c.load(OLD/'affinity-choice.json')['choice']['cpus']
    source=dict(plans_sha256=c.sha(OLD/'ssd/report-brackets.json'),binary_sha256=c.sha(c.E/'metric-consistent-20261111/bin/metric-search'),cpus=cpus,original_pending=c.load(OLD/'timings/pending.json'))
    plan=OUT/'plan.json'
    if plan.exists() and c.load(plan)!=source:raise RuntimeError('OpenImages supplement identity changed')
    c.save(plan,source);root=OUT/'supplement';results,pending=run_set(root,plans,cpus,reuse=OLD)
    anchors=[]
    for version in ('old','new'):
        label='ssd-'+version+'-side-f1';p=plans[label]['pages'];points=[]
        for L,r in p['bracket']:
            data=one(OUT/'anchors',label,L,r,1,cpus)
            if data:points.append(data['point'])
        if len(points)==2:
            w=p['weight'];value=(1-w)*points[0]['latency_us']+w*points[1]['latency_us'];prior=c.load(OLD/'ssd/matched95-summary.json')['results'][label]['pages']['mean']['latency_us']
            anchors.append(dict(version=version,current_us=value,prior_us=prior,relative_change=value/prior-1))
    c.save(OUT/'anchors.json',anchors)
    if len(anchors)!=2 or any(abs(a['relative_change'])>.10 for a in anchors):
        choice=measure.quiet();c.save(OUT/'fresh-affinity.json',choice);root=OUT/'fresh';results,pending=run_set(root,plans,choice['choice']['cpus'])
    comparisons=[]
    for version in ('old','new'):
        for bucket in ('side','centre'):
            for rule in ('pages','compute'):
                a=results['ssd-'+version+'-'+bucket+'-f0'][rule];b=results['ssd-'+version+'-'+bucket+'-f1'][rule]
                item=dict(version=version,bucket=bucket,rule=rule,complete=a['complete'] and b['complete'])
                if item['complete']:
                    item.update(off=a['mean'],on=b['mean'],filter_latency_change_percent=100*(b['mean']['latency_us']/a['mean']['latency_us']-1))
                comparisons.append(item)
    c.save(OUT/'comparisons.json',dict(cohort=str(root),comparisons=comparisons,pending=pending,anchors=anchors))
    text='# OpenImages 新旧图过滤配对补齐\n\n95% Recall@10；mixed-alpha偏侧与居中桶，四线程NUMA3。缺点与原测量均保留来源；只在锚点稳定时复用原独立核组，否则使用完整新批次。\n\n| 图 | 查询桶 | 冻结规则 | 关过滤ms | 开过滤ms | 开过滤变化 |\n|---|---|---|---:|---:|---:|\n'
    for x in comparisons:
        text+='| %s | %s | %s | %s | %s | %s |\n'%(x['version'],x['bucket'],x['rule'],'%.4f'%(x['off']['latency_us']/1000) if x['complete'] else '未完成','%.4f'%(x['on']['latency_us']/1000) if x['complete'] else '未完成','%+.2f%%'%x['filter_latency_change_percent'] if x['complete'] else '不下结论')
    text+='\n总提交页、拓扑／坐标分项等待和PQ评估数在comparisons.json中。范围结论仅适用于本轮图、查询桶和调参规则；不能把提交页等同于设备物理读取。\n'
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8');return dict(complete=not pending,summary=str(OUT/'comparisons.json'))
