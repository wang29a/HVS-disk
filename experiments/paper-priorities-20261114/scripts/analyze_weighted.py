"""Offline explanation of frozen alpha=.3 choices; never retune on report time."""
import sys
import common as c
sys.path.insert(0,str(c.MG/'scripts'))
import protocol as P

OUT=c.RUN/'weighted-anomaly'
def main():
    OUT.mkdir(exist_ok=True);frozen=c.load(c.MG_RUN/'frozen.json')['curves'];summary=c.load(c.MG_RUN/'report/summary.json')['records'];result=[]
    for rule in ('pages','compute','weighted_pages'):
        for method in ('M_equal','M_weighted'):
            key='0.3-'+method;info=frozen[key];train=c.load(c.MG_RUN/'tune'/(key+'.json'));by={p['label']:p for p in train}
            reconstructed=P.frontiers(train)
            if reconstructed[rule]!=info['frontiers'][rule]:raise RuntimeError('frozen frontier differs from training-only rule '+key+' '+rule)
            row=next(r for r in summary if r['medium']=='ssd' and r['rule']==rule and r['target']==.95 and r['method']==method and r['alpha']==.3)
            endpoints=[]
            for label in row['repeats'][0]['labels']:
                path=c.MG_RUN/'timings'/(key+'-'+label+'-ssd-rep1.json');point=c.load(path)['point'];job=info['jobs'][label];training=by[label];cost=P.work(training)
                dominators=[]
                for candidate in train:
                    work=P.work(candidate)
                    if candidate['recall']>=training['recall'] and all(work[k]<=cost[k]+1e-9 for k in ('pages','compute')) and any(work[k]<cost[k]-1e-9 for k in ('pages','compute')):
                        dominators.append(dict(label=candidate['label'],recall=candidate['recall'],cost=work))
                endpoints.append(dict(job=job,train_recall=training['recall'],train_cost=cost,report_recall=point['recall'],
                    measured_latency_us=point['latency_us'],expanded=sum(g['expanded'] for g in point['graphs']),pq_space_evals=sum(g['pq_space_evals'] for g in point['graphs']),
                    reranked=point['reranked'],candidate_union=point['candidates_after_union'],topology_pages=sum(g['topology_pages'] for g in point['graphs']),coordinate_pages=point['coordinate_pages'],training_dominators=dominators))
            result.append(dict(rule=rule,method=method,latency_us=row['latency_us'],expanded=row['expanded'],topology_pages=row['topology_pages'],coordinate_pages=row['coordinate_pages'],
                interpolation_weight=row['repeats'][0]['interpolation_weight'],recall_bracket=row['repeats'][0]['recall_bracket'],endpoints=endpoints,training_rule_reproduced=True))
    contribution=c.load(c.MG_RUN/'data/contribution.json')['0.3']['tune'];c.save(OUT/'summary.json',dict(records=result,training_contribution=contribution,
        scope='existing frozen experiment only; no new GT/search or grid extension',paper_policy='Use equal M as primary comparison; weighted M is reported as sensitivity, not claimed as an optimized upper baseline'))
    text='# α=0.3 加权双索引异常：已有配置核对\n\n所有前沿均由原训练数据和工作量规则重建且与冻结文件一致；没有按报告延迟重挑参数。下表列95%召回报告夹点，原始配对和插值权重见summary.json。\n\n| 规则 | 方法 | L | kx | ky | 分配 | 训练召回 | 报告召回 | 展开 | 精排 | 拓扑页 | 坐标页 |\n|---|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|\n'
    for row in result:
        for r in row['endpoints']:
            j=r['job'];text+='| %s | %s | %d | %d | %d | %s | %.3f%% | %.3f%% | %.1f | %.1f | %.1f | %.1f |\n'%(row['rule'],row['method'],j['L'],j['kx'],j['ky'],j.get('allocation','label encoded'),r['train_recall']*100,r['report_recall']*100,r['expanded'],r['reranked'],r['topology_pages'],r['coordinate_pages'])
    text+='\n加权规则的名义权重／GT贡献权重与等量分配不是嵌套的候选集合；把更多预算给某一侧可能减少另一侧对融合近邻的覆盖，进而需要更大L或候选预算。只有本表确实出现这种配置变化时，才用它解释本轮差距，不能把机制猜测写成已经证明。即使冻结规则执行正确，有限网格与页数／计算代理也不是最低实测延迟的保证。\n\n论文主对照采用等量版，加权版完整报告为该分配网格的敏感性结果；本次不补网格、不用报告集延迟改选配置，不宣称加权索引架构天然更差。\n'
    (OUT/'RESULTS.md').write_text(text,encoding='utf-8');return dict(complete=True,summary=str(OUT/'summary.json'))
