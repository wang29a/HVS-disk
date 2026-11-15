"""Only frozen report curves; no interpolation of tail latency or extrapolation."""
import csv
import math
from pathlib import Path
import statistics
import common as c
import protocol as p

NAMES={'DEG_off':'DEG off','DEG_on':'DEG on','Or_ideal':'Or (fixed)','F':'F (0.5 graph)','M_equal':'M equal','M_weighted':'M weighted'}

def decisions(records,rule,baseline):
    def selected(method):return {r['alpha']:r for r in records if r['rule']==rule and r['medium']=='ssd' and r['target']==.95 and r['method']==method and r['covered']}
    base,oracle,fixed=selected(baseline),selected('Or_ideal'),selected('F')
    out=dict(rule=rule,baseline=baseline,scope='Laion 2M; SSD PQ rerank; no shared cache; four NUMA3 threads; disk Vamana/DEG adaptations')
    alphas=[.1,.3,.5,.7,.9]
    if not all(a in base and a in oracle and a in fixed for a in alphas):return dict(out,status='insufficient matched-recall coverage')
    ratios={a:dict(latency=oracle[a]['latency_us']/base[a]['latency_us'],expanded=oracle[a]['expanded']/base[a]['expanded']) for a in alphas}
    fast={a:(ratios[a]['latency']<=.9 or ratios[a]['expanded']<=.9) for a in alphas}
    close={a:abs(ratios[a]['latency']-1)<.1 and abs(ratios[a]['expanded']-1)<.1 for a in alphas}
    out.update(status='complete',oracle_ratios=ratios,
        oracle_all_weights_fast=all(fast.values()),oracle_endpoints_fast_center_close=fast[.1] and fast[.9] and close[.5],oracle_all_close=all(close.values()),
        fixed_endpoint_degradation={a:dict(latency_over_oracle=fixed[a]['latency_us']/oracle[a]['latency_us'],expansion_over_oracle=fixed[a]['expanded']/oracle[a]['expanded']) for a in (.1,.9)})
    multi={}
    for method in ('M_equal','M_weighted'):
        rows=selected(method)
        if not all(a in rows for a in alphas):multi[method]=dict(covered=False);continue
        ratio={a:dict(latency=rows[a]['latency_us']/base[a]['latency_us'],expanded=rows[a]['expanded']/base[a]['expanded']) for a in alphas}
        multi[method]=dict(covered=True,ratios=ratio,not_slower_weights=sum(ratio[a]['latency']<=1 for a in alphas),necessity_alert=sum(ratio[a]['latency']<=1 for a in alphas)>=3,
            endpoints_faster_center_slower=ratio[.1]['latency']<=.9 and ratio[.9]['latency']<=.9 and ratio[.5]['latency']>=1.1)
    out['multi']=multi;return out

def plot(records,out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for medium in ('ssd','topo'):
        for rule in p.RULES:
            for actual in (True,False):
                fig,ax=plt.subplots(figsize=(6.6,4.1))
                for method in p.METHODS:
                    rows=sorted([r for r in records if r['medium']==medium and r['rule']==rule and r['method']==method and r['target']==.95],key=lambda r:r['alpha'])
                    x=[r['image_contribution'] if actual else r['alpha'] for r in rows]
                    y=[r['latency_us']/1000 if r['covered'] else float('nan') for r in rows]
                    ax.plot(x,y,marker='o',linewidth=1.4,markersize=4,label=NAMES[method])
                ax.set_xlabel('Actual image contribution on GT top-10' if actual else 'Nominal alpha')
                ax.set_ylabel('Mean query latency (ms) at Recall@10 = 95%');ax.grid(alpha=.2);ax.legend(fontsize=8,ncol=2)
                ax.set_title('Laion 2M | '+medium+' | '+rule+' | 4 threads',fontsize=10);fig.tight_layout()
                name=medium+'-'+rule+('-actual' if actual else '-nominal')
                for ext in ('svg','pdf','png'):fig.savefig(str(out/(name+'.'+ext)),dpi=180)
                plt.close(fig)

def main():
    frozen=c.load(c.RUN/'frozen.json');out=c.RUN/'report';out.mkdir(exist_ok=True)
    contribution=c.load(c.RUN/'data/contribution.json');rows=[];tails=[]
    for key,info in frozen['curves'].items():
        for medium in ('ssd','topo'):
            for rule,labels in info['frontiers'].items():
                for target in (.9,.95):
                    repeated=[]
                    for rep in range(1,4):
                        points=[]
                        for label in labels:
                            path=c.RUN/'timings'/(key+'-'+label+'-'+medium+'-rep'+str(rep)+'.json')
                            if not path.exists():continue
                            pt=c.load(path)['point'];points.append(pt);tails.append(dict(alpha=info['alpha'],method=info['method'],medium=medium,rule=rule,rep=rep,label=label,recall=pt['recall'],p95_us=pt['p95_us']))
                        repeated.append(p.matched(points,target))
                    row=dict(alpha=info['alpha'],method=info['method'],medium=medium,rule=rule,target=target,covered=all(r['covered'] for r in repeated),repeats=repeated,image_contribution=contribution[str(info['alpha'])]['report']['mean'])
                    if row['covered']:
                        for field in ('latency_us','qps','expanded','pq_space_evals','reranked','topology_pages','topology_submitted_pages','topology_wait_us','coordinate_pages','coordinate_submitted_pages','coordinate_wait_us','candidates_before_union','candidates_after_union'):
                            values=[r[field] for r in repeated];row[field]=statistics.mean(values);row[field+'_std']=statistics.stdev(values)
                    raw=[]
                    for rep in range(1,4):
                        for label in labels:
                            path=c.RUN/'timings'/(key+'-'+label+'-'+medium+'-rep'+str(rep)+'.json')
                            if path.exists():raw.append(c.load(path)['point'])
                    row['peak_rss_kib']=max((v['peak_rss_kib'] for v in raw),default=None)
                    row['resident_page_bytes']=max((v['resident_page_bytes'] for v in raw),default=None)
                    rows.append(row)
    tables={name:c.load(c.RUN/'packed'/name/'identity.json') for name in ('x','y','a01','a03','a05','a07','a09','u')}
    groups={'M_equal':['x','y'],'M_weighted':['x','y'],'M_endpoint':['x','y'],'Or_ideal':['a01','a03','a05','a07','a09'],'Or_nearest':['a01','a03','a05','a07','a09'],'F':['a05'],'DEG_off':['u'],'DEG_on':['u']}
    storage={name:dict(topology_bytes=sum(tables[g]['topology_bytes'] for g in graphs),sidecar_bytes=sum(tables[g]['sidecar_bytes'] for g in graphs),build_seconds=sum(tables[g]['build_seconds'] for g in graphs)) for name,graphs in groups.items()}
    judgement=[decisions(rows,rule,base) for rule in p.RULES for base in ('DEG_off','DEG_on')]
    result=dict(status='awaiting_user_review',scope='Laion-only; no OpenImages timing authorised yet',records=rows,storage=storage,graphs=tables,decision_checks=judgement,
        aliases='On the five fixed non-endpoint weights, Or_nearest=Or_ideal. M_endpoint=selected M_weighted; no independent endpoint advantage has been measured.',
        memory='Per-point peak_rss_kib includes process-owned PQ/books/sidecar, query scratch and resident topology; all fields remain in raw accepted timing JSON. Or total storage counts five graphs even when a fixed-alpha process loads one.',
        limits='Adapted graph architectures, not native HNSW; <= fixed pruning; no universal argument from a 2M subset; p95 only at raw points, never interpolated.')
    c.save(out/'summary.json',result)
    with (out/'storage.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['method','topology_bytes','sidecar_bytes','build_seconds']);w.writerows([name]+[d[k] for k in ('topology_bytes','sidecar_bytes','build_seconds')] for name,d in storage.items())
    with (out/'graphs.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['graph','nodes','edges','mean_degree','topology_bytes','sidecar_bytes','build_seconds'])
        for name,d in tables.items():w.writerow([name,d['stats']['nodes'],d['stats']['edges'],d['stats']['edges']/d['stats']['nodes'],d['topology_bytes'],d['sidecar_bytes'],d['build_seconds']])
    with (out/'raw-p95.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=['alpha','method','medium','rule','rep','label','recall','p95_us']);w.writeheader();w.writerows(tails)
    fields=['alpha','method','medium','rule','target','covered','image_contribution','latency_us','qps','expanded','pq_space_evals','reranked','topology_pages','coordinate_pages','peak_rss_kib','resident_page_bytes']
    with (out/'matched.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    plot(rows,out)
    alerts=[x for x in judgement if any(v.get('necessity_alert') for v in x.get('multi',{}).values())]
    if alerts:c.save(out/'NECESSITY-ALERT.json',dict(message='HNSW_M is not slower at >=3/5 weights against the named unified baseline. Report now; do not expand data sets.',cases=alerts))
    text='# Laion 2M 多图对照结果\n\n这是 Vamana／固定权重 DEG 的磁盘架构适配，不是原生 HNSW 复现。全部配置由调参集工作量冻结；报告集不参与挑配置。三条规则、两种驻留条件分别报告，不按结果选最佳规则。\n\n'
    text+='95%召回的实际贡献比例主图为 `ssd-*-actual.svg`，名义权重补充图为 `ssd-*-nominal.svg`。`matched.csv` 保存90%／95%均值，`summary.json`保留逐轮插值区间和判读表，`raw-p95.csv`只列原始工作点尾延迟。未覆盖的点为空缺，不能外推。\n\n'
    text+='| 方法 | 拓扑 MiB | 区间旁表 MiB | 建图秒数 |\n|---|---:|---:|---:|\n'
    for name,d in storage.items():text+='| %s | %.2f | %.2f | %.1f |\n'%(name,d['topology_bytes']/2**20,d['sidecar_bytes']/2**20,d['build_seconds'])
    text+='\n所有方法共用同一份XY坐标、两套PQ码本和编码；表中不重复计入。已有图建图时间来自其原始记录，不冒充本轮隔离计时。旁表算进DEG内存预算。实际运行内存见每配置原始RSS，而非只报向量大小。\n\n'
    text+='预先声明的六条判读均在 `summary.json` 的 `decision_checks` 分规则、分统一图开关列出。若部分权重未覆盖95%，对应判断必须保留“证据不足”；单空间图端点版在本轮五个非端点上与其双索引配置相同，不据此宣称端点优化成立。\n\n'
    text+='Laion 子集完成后停止，等待用户确认。没有自动扩展到 OpenImages 的步骤。\n'
    (out/'RESULTS.md').write_text(text,encoding='utf-8')
if __name__=='__main__':main()
